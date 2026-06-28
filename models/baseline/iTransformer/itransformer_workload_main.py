import argparse
import os
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from model.iTransformer import Model as ITransformer


def load_workload(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = np.load(open(path, "rb"), allow_pickle=True)
    return np.asarray(data, dtype=np.float32)


def use_dyn_azure_scale(*paths):
    names = " ".join(Path(path).name.lower() for path in paths if path)
    return "azure" in names or "tr_vm" in names or "te_vm" in names or "_vm" in names


def split_by_time(data, train_ratio=0.8, test_ratio=0.2):
    periods = data.shape[1]
    train_periods = int(periods * train_ratio)
    test_periods = int(periods * test_ratio)
    return data[:, :train_periods, :], data[:, -test_periods:, :]


def inverse_scale(values, scaler):
    values = np.asarray(values)
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def get_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))


class WorkloadDataset(Dataset):
    def __init__(self, x_scaled, y_raw, seq_len=48, pred_len=24, step=12):
        x_hist = []
        y_future_scaled = []
        y_future_raw = []

        for series_id in range(x_scaled.shape[0]):
            for pos in range(seq_len, x_scaled.shape[1] - pred_len, step):
                x_hist.append(x_scaled[series_id, pos - seq_len:pos, :])
                y_future_scaled.append(x_scaled[series_id, pos:pos + pred_len, 0])
                y_future_raw.append(y_raw[series_id, pos:pos + pred_len])

        if not x_hist:
            raise RuntimeError(
                "No windows built from data shape {}. Need more than seq_len + pred_len points.".format(
                    x_scaled.shape
                )
            )

        self.x_hist = np.asarray(x_hist, dtype=np.float32)
        self.y_future_scaled = np.asarray(y_future_scaled, dtype=np.float32)
        self.y_future_raw = np.asarray(y_future_raw, dtype=np.float32)

    def __len__(self):
        return self.x_hist.shape[0]

    def __getitem__(self, index):
        return self.x_hist[index], self.y_future_scaled[index], self.y_future_raw[index]


def build_model(args, input_size, device):
    config = SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        output_attention=False,
        use_norm=args.use_norm,
        d_model=args.d_model,
        embed=args.embed,
        freq=args.freq,
        dropout=args.dropout,
        class_strategy=args.class_strategy,
        factor=args.factor,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        activation=args.activation,
        enc_in=input_size,
    )
    return ITransformer(config).float().to(device)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_raw = load_workload(args.train_data)
    if args.test_data:
        test_raw = load_workload(args.test_data)
    else:
        train_raw, test_raw = split_by_time(train_raw)

    if use_dyn_azure_scale(args.train_data, args.test_data):
        train_raw = train_raw.copy()
        test_raw = test_raw.copy()
        train_raw[:, :, 0] = train_raw[:, :, 0] / 10.0
        test_raw[:, :, 0] = test_raw[:, :, 0] / 10.0

    input_size = min(args.input_size, train_raw.shape[-1])
    train_selected = train_raw[:, :, :input_size]
    test_selected = test_raw[:, :, :input_size]
    y_train_raw = train_selected[:, :, 0]
    y_test_raw = test_selected[:, :, 0]

    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    n_train, t_train, f_train = train_selected.shape
    train_scaled = x_scaler.fit_transform(train_selected.reshape(-1, f_train)).reshape(n_train, t_train, f_train)
    n_test, t_test, _ = test_selected.shape
    test_scaled = x_scaler.transform(test_selected.reshape(-1, f_train)).reshape(n_test, t_test, f_train)

    train_loader = DataLoader(
        WorkloadDataset(train_scaled, y_train_raw, args.seq_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        WorkloadDataset(test_scaled, y_test_raw, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("train data:", args.train_data, train_selected.shape)
    print("test data:", args.test_data or "time split", test_selected.shape)
    print("input size:", input_size)

    model = build_model(args, input_size, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)

    best_loss = float("inf")
    best_mse = None
    best_mae = None
    best_smape = None
    best_epoch = None

    for epoch in tqdm(range(args.train_epochs)):
        model.train()
        train_losses = []
        for x_hist, y_future_scaled, _ in train_loader:
            x_hist = x_hist.float().to(device)
            y_future_scaled = y_future_scaled.float().to(device)

            pred = model(x_hist, None, None, None)[:, :, 0]
            loss = criterion(pred, y_future_scaled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        test_mses = []
        test_maes = []
        test_smapes = []
        with torch.no_grad():
            for x_hist, y_future_scaled, y_future_raw in test_loader:
                x_hist = x_hist.float().to(device)
                y_future_scaled = y_future_scaled.float().to(device)

                pred = model(x_hist, None, None, None)[:, :, 0]
                loss = criterion(pred, y_future_scaled)

                pred_raw = inverse_scale(pred.cpu().numpy(), y_scaler)
                true_raw = y_future_raw.numpy()
                test_losses.append(loss.item())
                test_mses.append(np.mean((pred_raw.reshape(-1) - true_raw.reshape(-1)) ** 2))
                test_maes.append(np.mean(np.abs(pred_raw.reshape(-1) - true_raw.reshape(-1))))
                test_smapes.append(get_smape(true_raw, pred_raw))

        epoch_loss = float(np.mean(test_losses))
        epoch_mse = float(np.mean(test_mses))
        epoch_mae = float(np.mean(test_maes))
        epoch_smape = float(np.mean(test_smapes))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_mse = epoch_mse
            best_mae = epoch_mae
            best_smape = epoch_smape
            best_epoch = epoch
            torch.save(model, os.path.join(args.save_dir, "iTransformer_workload_best.pt"))
            pickle.dump(
                [x_scaler, y_scaler],
                open(os.path.join(args.save_dir, "itransformer_workload_scaler.pkl"), "wb"),
            )

        print(
            "epoch {}, train loss: {}, test loss: {}, mse: {}, mae: {}, smape(%): {}".format(
                epoch,
                float(np.mean(train_losses)),
                epoch_loss,
                epoch_mse,
                epoch_mae,
                epoch_smape,
            )
        )

    print(
        "best epoch: {}, best mse: {}, best mae: {}, best smape(%): {}".format(
            best_epoch, best_mse, best_mae, best_smape
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--train_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=12)
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--class_strategy", type=str, default="projection")
    parser.add_argument("--use_norm", action="store_true", default=True)
    parser.add_argument("--no_use_norm", action="store_false", dest="use_norm")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    main(parser.parse_args())
