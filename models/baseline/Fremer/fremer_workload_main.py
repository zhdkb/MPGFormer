import argparse
import copy
import os
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from protocol import split_workload, report_test_metrics, mean_by_batch
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ts_benchmark.baselines.fremer.fremer_model import FremerModel


def load_workload(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = np.load(open(path, "rb"), allow_pickle=True)
    return np.asarray(data, dtype=float)


def use_dyn_azure_scale(*paths):
    names = " ".join(Path(path).name.lower() for path in paths if path)
    return "azure" in names or "tr_vm" in names or "te_vm" in names or "_vm" in names


def split_by_time(X, train_ratio=0.8, test_ratio=0.2):
    num_periods = X.shape[1]
    train_periods = int(num_periods * train_ratio)
    test_periods = int(num_periods * test_ratio)
    return X[:, :train_periods, :], X[:, -test_periods:, :]


def inverse_scale(values, scaler):
    values = np.asarray(values)
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


def get_mse(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = inverse_scale(y_true, scaler)
        y_pred = inverse_scale(y_pred, scaler)
    return np.mean((y_true - y_pred) ** 2)


def get_mae(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = inverse_scale(y_true, scaler)
        y_pred = inverse_scale(y_pred, scaler)
    return np.mean(np.abs(y_true - y_pred))




def get_smape(y_true, y_pred, scaler=None):
    if scaler is not None:
        try:
            y_true = inverse_scale(y_true, scaler)
            y_pred = inverse_scale(y_pred, scaler)
        except NameError:
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))


class WorkloadDataset(Dataset):
    def __init__(self, X_scaled, y_raw, seq_len=48, pred_len=24, step=12):
        x_hist = []
        y_future_scaled = []
        y_future_raw = []

        for i in range(X_scaled.shape[0]):
            for j in range(seq_len, X_scaled.shape[1] - pred_len + 1, step):
                x_hist.append(X_scaled[i, j - seq_len:j, :])
                y_future_scaled.append(X_scaled[i, j:j + pred_len, 0])
                y_future_raw.append(y_raw[i, j:j + pred_len])

        if not x_hist:
            raise RuntimeError(
                "No windows built from data shape {}. Need more than seq_len + pred_len points.".format(
                    X_scaled.shape
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
    max_freq_len = int((args.seq_len + args.pred_len) / 2) + 1
    if args.low_cut >= max_freq_len:
        raise ValueError("--low_cut must be smaller than {} for seq_len + pred_len = {}".format(
            max_freq_len,
            args.seq_len + args.pred_len,
        ))

    config = SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        low_cut=args.low_cut,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        dropout=args.dropout,
        enc_in=input_size,
    )
    return FremerModel(config).float().to(device)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    Xtr_raw, Xva_raw, Xte_raw = split_workload(
        load_workload(args.train_data), args.train_data,
        load_workload(args.test_data) if args.test_data else None,
        load_workload(args.val_data) if args.val_data else None,
    )

    if use_dyn_azure_scale(args.train_data, args.test_data):
        Xtr_raw[:, :, 0] = Xtr_raw[:, :, 0] / 10.0
        Xva_raw[:, :, 0] = Xva_raw[:, :, 0] / 10.0
        Xte_raw[:, :, 0] = Xte_raw[:, :, 0] / 10.0

    input_size = min(args.input_size, Xtr_raw.shape[-1])
    Xtr_selected = Xtr_raw[:, :, :input_size]
    Xva_selected = Xva_raw[:, :, :input_size]
    Xte_selected = Xte_raw[:, :, :input_size]
    ytr_raw = Xtr_selected[:, :, 0]
    yva_raw = Xva_selected[:, :, 0]
    yte_raw = Xte_selected[:, :, 0]

    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    yscaler.fit(ytr_raw.reshape(-1, 1))

    num_ts, num_periods, num_features = Xtr_selected.shape
    Xtr_scaled = xscaler.fit_transform(Xtr_selected.reshape(-1, num_features)).reshape(
        num_ts, num_periods, num_features
    )
    num_ts_te, num_periods_te, _ = Xte_selected.shape
    num_ts_va, num_periods_va, _ = Xva_selected.shape
    Xva_scaled = xscaler.transform(Xva_selected.reshape(-1, num_features)).reshape(
        num_ts_va, num_periods_va, num_features
    )
    Xte_scaled = xscaler.transform(Xte_selected.reshape(-1, num_features)).reshape(
        num_ts_te, num_periods_te, num_features
    )

    train_loader = DataLoader(
        WorkloadDataset(Xtr_scaled, ytr_raw, args.seq_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WorkloadDataset(Xva_scaled, yva_raw, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WorkloadDataset(Xte_scaled, yte_raw, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("train data:", args.train_data, Xtr_selected.shape)
    print("test data:", args.test_data or "time split", Xte_selected.shape)
    print("input size:", input_size)

    model = build_model(args, input_size, device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)

    def evaluate(loader):
        model.eval()
        test_losses = []
        test_mses = []
        test_maes = []
        test_smapes = []
        batch_sizes = []
        with torch.no_grad():
            for x_hist, y_future_scaled, y_future_raw in loader:
                x_hist = x_hist.float().to(device)
                y_future_scaled = y_future_scaled.float().to(device)

                pred = model(x_hist)[:, :, 0]
                loss = criterion(pred, y_future_scaled)

                batch_sizes.append(len(y_future_scaled))
                test_losses.append(loss.item())
                pred_raw = inverse_scale(pred.cpu().numpy(), yscaler)
                true_raw = y_future_raw.numpy()
                test_mses.append(np.mean((pred_raw.reshape(-1) - true_raw.reshape(-1)) ** 2))
                test_maes.append(np.mean(np.abs(pred_raw.reshape(-1) - true_raw.reshape(-1))))
                test_smapes.append(get_smape(true_raw, pred_raw))

        epoch_loss = float(mean_by_batch(test_losses, batch_sizes))
        epoch_mse = float(mean_by_batch(test_mses, batch_sizes))
        epoch_mae = float(mean_by_batch(test_maes, batch_sizes))
        epoch_smape = float(mean_by_batch(test_smapes, batch_sizes))
        return epoch_loss, epoch_mse, epoch_mae, epoch_smape

    best_loss = float("inf")
    best_state = None
    best_mse = None
    best_mae = None
    best_smape = None
    best_epoch = None

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_losses = []
        for x_hist, y_future_scaled, _ in train_loader:
            x_hist = x_hist.float().to(device)
            y_future_scaled = y_future_scaled.float().to(device)

            pred = model(x_hist)[:, :, 0]
            loss = criterion(pred, y_future_scaled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        epoch_loss, epoch_mse, epoch_mae, epoch_smape = evaluate(val_loader)
        if epoch_mse < best_loss:
            best_loss = epoch_mse
            best_state = copy.deepcopy(model.state_dict())
            best_mse = epoch_mse
            best_mae = epoch_mae
            best_smape = epoch_smape
            best_epoch = epoch
            torch.save(model, os.path.join(args.save_dir, "Fremer_workload_best.pt"))
            pickle.dump([xscaler, yscaler], open(os.path.join(args.save_dir, "fremer_workload_scaler.pkl"), "wb"))

        print(
            "epoch {}, train loss: {}, val loss: {}, mse: {}, mae: {}, smape(%): {}".format(
                epoch, float(np.mean(train_losses)), epoch_loss, epoch_mse, epoch_mae, epoch_smape
            )
        )

    print(
        "best epoch: {}, best mse: {}, best mae: {}, best smape(%): {}".format(
            best_epoch, best_mse, best_mae, best_smape
        )
    )

    model.load_state_dict(best_state)
    test_loss, test_mse, test_mae, test_smape = evaluate(test_loader)
    print("test loss: {}, test mse: {}, test mae: {}, test smape(%): {}".format(
        test_loss, test_mse, test_mae, test_smape
    ))

    report_test_metrics(Path(__file__).stem, args.seed, args.train_data, args.test_data, test_mse, test_mae, test_smape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fremer workload reproduction entry for ECW, Azure, and Google data.")
    parser.add_argument("--train_data", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--low_cut", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    main(parser.parse_args())
