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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from pdf_model import Model as PDFModel


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
            for pos in range(seq_len, x_scaled.shape[1] - pred_len + 1, step):
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
    if not (len(args.period) == len(args.patch_len) == len(args.stride)):
        raise ValueError("--period, --patch_len, and --stride must have the same length.")

    config = SimpleNamespace(
        enc_in=input_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        head_dropout=args.head_dropout,
        individual=args.individual,
        add=args.add,
        wo_conv=args.wo_conv,
        serial_conv=args.serial_conv,
        kernel_list=args.kernel_list,
        patch_len=args.patch_len,
        period=args.period,
        stride=args.stride,
        padding_patch=args.padding_patch,
        revin=args.revin,
        affine=args.affine,
        subtract_last=args.subtract_last,
    )
    return PDFModel(config).float().to(device)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_raw, val_raw, test_raw = split_workload(
        load_workload(args.train_data), args.train_data,
        load_workload(args.test_data) if args.test_data else None,
        load_workload(args.val_data) if args.val_data else None,
    )

    if use_dyn_azure_scale(args.train_data, args.test_data):
        train_raw = train_raw.copy()
        val_raw = val_raw.copy()
        test_raw = test_raw.copy()
        train_raw[:, :, 0] = train_raw[:, :, 0] / 10.0
        val_raw[:, :, 0] = val_raw[:, :, 0] / 10.0
        test_raw[:, :, 0] = test_raw[:, :, 0] / 10.0

    input_size = min(args.input_size, train_raw.shape[-1])
    train_selected = train_raw[:, :, :input_size]
    val_selected = val_raw[:, :, :input_size]
    test_selected = test_raw[:, :, :input_size]
    y_train_raw = train_selected[:, :, 0]
    y_val_raw = val_selected[:, :, 0]
    y_test_raw = test_selected[:, :, 0]

    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    n_train, t_train, f_train = train_selected.shape
    train_scaled = x_scaler.fit_transform(train_selected.reshape(-1, f_train)).reshape(n_train, t_train, f_train)
    n_test, t_test, _ = test_selected.shape
    n_val, t_val, _ = val_selected.shape
    val_scaled = x_scaler.transform(val_selected.reshape(-1, f_train)).reshape(n_val, t_val, f_train)
    test_scaled = x_scaler.transform(test_selected.reshape(-1, f_train)).reshape(n_test, t_test, f_train)

    train_loader = DataLoader(
        WorkloadDataset(train_scaled, y_train_raw, args.seq_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WorkloadDataset(val_scaled, y_val_raw, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WorkloadDataset(test_scaled, y_test_raw, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("train data:", args.train_data, train_selected.shape)
    print("test data:", args.test_data or "time split", test_selected.shape)
    print("input size:", input_size)
    print("periods:", args.period)

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

                pred_raw = inverse_scale(pred.cpu().numpy(), y_scaler)
                true_raw = y_future_raw.numpy()
                batch_sizes.append(len(y_future_scaled))
                test_losses.append(loss.item())
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
            torch.save(model, os.path.join(args.save_dir, "PDF_workload_best.pt"))
            pickle.dump([x_scaler, y_scaler], open(os.path.join(args.save_dir, "pdf_workload_scaler.pkl"), "wb"))

        print(
            "epoch {}, train loss: {}, val loss: {}, mse: {}, mae: {}, smape(%): {}".format(
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

    model.load_state_dict(best_state)
    test_loss, test_mse, test_mae, test_smape = evaluate(test_loader)
    print("test loss: {}, test mse: {}, test mae: {}, test smape(%): {}".format(
        test_loss, test_mse, test_mae, test_smape
    ))

    report_test_metrics(Path(__file__).stem, args.seed, args.train_data, args.test_data, test_mse, test_mae, test_smape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF workload reproduction entry for project datasets.")
    parser.add_argument("--train_data", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=12)
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--period", type=int, nargs="+", default=[24, 12, 6])
    parser.add_argument("--patch_len", type=int, nargs="+", default=[4, 4, 3])
    parser.add_argument("--stride", type=int, nargs="+", default=[4, 4, 3])
    parser.add_argument("--kernel_list", type=int, nargs="+", default=[3, 7, 11])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc_dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--padding_patch", type=str, default="end")
    parser.add_argument("--revin", action="store_true", default=True)
    parser.add_argument("--no_revin", action="store_false", dest="revin")
    parser.add_argument("--affine", action="store_true", default=False)
    parser.add_argument("--subtract_last", action="store_true", default=False)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--add", action="store_true", default=False)
    parser.add_argument("--wo_conv", action="store_true", default=False)
    parser.add_argument("--serial_conv", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    main(parser.parse_args())
