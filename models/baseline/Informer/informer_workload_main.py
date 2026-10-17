import argparse
import copy
import os
import pickle
import random
from pathlib import Path

import numpy as np
from protocol import split_workload, report_test_metrics, mean_by_batch
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.model import Informer


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
    def __init__(self, X, seq_len=48, label_len=12, pred_len=24, step=12):
        x_all = []
        y_all = []
        x_mark_all = []
        y_mark_all = []

        for i in range(X.shape[0]):
            for j in range(seq_len, X.shape[1] - pred_len + 1, step):
                x_all.append(X[i, j - seq_len:j, 0])
                y_all.append(X[i, j - label_len:j + pred_len, 0])
                x_mark_all.append(X[i, j - seq_len:j, 1:4])
                y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

        if not x_all:
            raise RuntimeError(
                "No windows built from data shape {}. Need more than seq_len + pred_len points.".format(X.shape)
            )
        self.x = np.asarray(x_all).reshape(-1, seq_len, 1).astype(float)
        self.y = np.asarray(y_all).reshape(-1, label_len + pred_len, 1).astype(float)
        self.x_mark = np.asarray(x_mark_all).reshape(-1, seq_len, 3).astype(float)
        self.y_mark = np.asarray(y_mark_all).reshape(-1, label_len + pred_len, 3).astype(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_mark[index], self.y_mark[index]


def build_model(args, device):
    return Informer(
        enc_in=1,
        dec_in=1,
        c_out=1,
        seq_len=args.seq_len,
        label_len=args.label_len,
        out_len=args.pred_len,
        factor=args.factor,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn=args.attn,
        embed=args.embed,
        freq=args.freq,
        activation=args.activation,
        output_attention=False,
        distil=args.distil,
        mix=args.mix,
        device=device,
    ).float().to(device)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    Xtr, Xva, Xte = split_workload(
        load_workload(args.train_data), args.train_data,
        load_workload(args.test_data) if args.test_data else None,
        load_workload(args.val_data) if args.val_data else None,
    )

    if use_dyn_azure_scale(args.train_data, args.test_data):
        Xtr[:, :, 0] = Xtr[:, :, 0] / 10.0
        Xva[:, :, 0] = Xva[:, :, 0] / 10.0
        Xte[:, :, 0] = Xte[:, :, 0] / 10.0

    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    yscaler.fit(Xtr[:, :, 0].reshape(-1, 1))

    num_ts, num_periods, num_features = Xtr.shape
    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    num_ts_te, num_periods_te, _ = Xte.shape
    num_ts_va, num_periods_va, _ = Xva.shape
    Xva = xscaler.transform(Xva.reshape(-1, num_features)).reshape(num_ts_va, num_periods_va, num_features)
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts_te, num_periods_te, num_features)

    train_loader = DataLoader(
        WorkloadDataset(Xtr, args.seq_len, args.label_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WorkloadDataset(Xva, args.seq_len, args.label_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WorkloadDataset(Xte, args.seq_len, args.label_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("train data:", args.train_data, Xtr.shape)
    print("test data:", args.test_data or "time split", Xte.shape)

    model = build_model(args, device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss().to(device)

    def evaluate(loader):
        model.eval()
        test_losses = []
        test_mses = []
        test_maes = []
        test_smapes = []
        batch_sizes = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -args.pred_len:, -1]
                target = batch_y[:, -args.pred_len:, -1]
                loss = criterion(outputs, target)

                batch_sizes.append(len(target))
                test_losses.append(loss.item())
                test_mses.append(get_mse(target.cpu(), outputs.cpu(), yscaler))
                test_maes.append(get_mae(target.cpu(), outputs.cpu(), yscaler))
                test_smapes.append(get_smape(target.cpu(), outputs.cpu(), yscaler))

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

    for epoch in tqdm(range(args.train_epochs)):
        model.train()
        train_losses = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, -1]
            target = batch_y[:, -args.pred_len:, -1]
            loss = criterion(outputs, target)

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
            torch.save(model, os.path.join(args.save_dir, "Informer2020_workload_best.pt"))
            pickle.dump([xscaler, yscaler], open(os.path.join(args.save_dir, "informer_workload_scaler.pkl"), "wb"))

        print(
            "epoch {}, train loss: {}, val loss: {}, mse: {}, mae: {}, smape(%): {}".format(
                epoch, float(np.mean(train_losses)), epoch_loss, epoch_mse, epoch_mae, epoch_smape
            )
        )

    print("best epoch: {}, best mse: {}, best mae: {}, best smape(%): {}".format(best_epoch, best_mse, best_mae, best_smape))

    model.load_state_dict(best_state)
    test_loss, test_mse, test_mae, test_smape = evaluate(test_loader)
    print("test loss: {}, test mse: {}, test mae: {}, test smape(%): {}".format(
        test_loss, test_mse, test_mae, test_smape
    ))

    report_test_metrics('Informer', args.seed, args.train_data, args.test_data, test_mse, test_mae, test_smape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Informer workload reproduction entry for ECW and Azure data.")
    parser.add_argument("--train_data", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attn", type=str, default="prob")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--distil", type=bool, default=True)
    parser.add_argument("--mix", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    main(parser.parse_args())
