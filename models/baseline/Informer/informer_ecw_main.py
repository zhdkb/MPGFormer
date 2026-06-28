import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.model import Informer


class ECWDataset(Dataset):
    def __init__(self, X, seq_len=48, label_len=12, pred_len=24, step=12):
        num_ts, num_periods, _ = X.shape
        x_all = []
        y_all = []
        x_mark_all = []
        y_mark_all = []

        for i in range(num_ts):
            for j in range(seq_len, num_periods - pred_len, step):
                x_all.append(X[i, j - seq_len:j, 0])
                y_all.append(X[i, j - label_len:j + pred_len, 0])
                x_mark_all.append(X[i, j - seq_len:j, 1:4])
                y_mark_all.append(X[i, j - label_len:j + pred_len, 1:4])

        self.x = np.asarray(x_all).reshape(-1, seq_len, 1).astype(float)
        self.y = np.asarray(y_all).reshape(-1, label_len + pred_len, 1).astype(float)
        self.x_mark = np.asarray(x_mark_all).reshape(-1, seq_len, 3).astype(float)
        self.y_mark = np.asarray(y_mark_all).reshape(-1, label_len + pred_len, 3).astype(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_mark[index], self.y_mark[index]


def train_test_split(X, y, train_ratio=0.8, test_ratio=0.2):
    _, num_periods, _ = X.shape
    train_periods = int(num_periods * train_ratio)
    test_periods = int(num_periods * test_ratio)
    return X[:, :train_periods, :], y[:, :train_periods], X[:, -test_periods:, :], y[:, -test_periods:]


def get_mape(y_true, y_pred, scaler=None):
    if scaler:
        y_true = inverse_scale(y_true, scaler)
        y_pred = inverse_scale(y_pred, scaler)
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)


def get_mse(y_true, y_pred, scaler=None):
    if scaler:
        y_true = inverse_scale(y_true, scaler)
        y_pred = inverse_scale(y_pred, scaler)
    return np.mean((y_true - y_pred) ** 2)


def get_mae(y_true, y_pred, scaler=None):
    if scaler:
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
def inverse_scale(values, scaler):
    values = np.asarray(values)
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


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

    X_all = np.load(open(args.data_path, "rb"), allow_pickle=True)
    y_all = X_all[:, :, 0]
    Xtr, ytr, Xte, _ = train_test_split(X_all, y_all)

    num_ts, num_periods, num_features = Xtr.shape
    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    yscaler.fit(ytr.reshape(-1, 1))
    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    num_ts_te, num_periods_te, _ = Xte.shape
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts_te, num_periods_te, num_features)

    train_loader = DataLoader(
        ECWDataset(Xtr, args.seq_len, args.label_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ECWDataset(Xte, args.seq_len, args.label_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args, device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss().to(device)

    scaler_path = os.path.join(args.save_dir, "informer_data_scaler.pkl")
    pickle.dump([xscaler, yscaler], open(scaler_path, "wb"))

    best_loss = float("inf")
    best_mse = None
    best_mae = None
    best_mape = None
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

        model.eval()
        test_losses = []
        test_mses = []
        test_mapes = []
        test_maes = []
        test_smapes = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
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

                test_losses.append(loss.item())
                test_mses.append(get_mse(target.cpu(), outputs.cpu(), yscaler))
                test_mapes.append(get_mape(target.cpu(), outputs.cpu(), yscaler))
                test_maes.append(get_mae(target.cpu(), outputs.cpu(), yscaler))
                test_smapes.append(get_smape(target.cpu(), outputs.cpu(), yscaler))

        epoch_loss = float(np.mean(test_losses))
        epoch_mse = float(np.mean(test_mses))
        epoch_mape = float(np.mean(test_mapes))
        epoch_mae = float(np.mean(test_maes))
        epoch_smape = float(np.mean(test_smapes))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_mse = epoch_mse
            best_mape = epoch_mape
            best_mae = epoch_mae
            best_smape = epoch_smape
            best_epoch = epoch
            torch.save(model, os.path.join(args.save_dir, "Informer2020_best.pt"))

        print(
            "epoch {}, train loss: {}, test loss: {}, mse: {}, mape: {}, mae: {}, smape(%): {}".format(
                epoch,
                float(np.mean(train_losses)),
                epoch_loss,
                epoch_mse,
                epoch_mape,
                epoch_mae,
                epoch_smape,
            )
        )

    print("best epoch: {}, best mse: {}, best mape: {}, best mae: {}, best smape(%): {}".format(best_epoch, best_mse, best_mape, best_mae, best_smape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Informer2020 adapted to ECW workload data.")
    parser.add_argument("--data_path", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
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
