import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def batch_generator_padding(X, seq_len=48, label_len=12, pred_len=24, step=2):
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

    x_all = np.asarray(x_all).reshape(-1, seq_len, 1)
    y_all = np.asarray(y_all).reshape(-1, label_len + pred_len, 1)
    x_mark_all = np.asarray(x_mark_all).reshape(-1, seq_len, 3)
    y_mark_all = np.asarray(y_mark_all).reshape(-1, label_len + pred_len, 3)
    return x_all, y_all, x_mark_all, y_mark_all


def main(args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path).to(device)
    model.eval()
    criterion = nn.MSELoss().to(device)

    X = pickle.load(open(args.data_path, "rb"))
    num_ts, _, num_features = X.shape
    xscaler, yscaler = pickle.load(open(args.scaler_path, "rb"))
    Xte = xscaler.transform(X.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    X, Y, X_mark, Y_mark = batch_generator_padding(
        Xte,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        step=args.step,
    )
    print("test windows:", X.shape[0])

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X.astype(float)).float(),
            torch.from_numpy(Y.astype(float)).float(),
            torch.from_numpy(X_mark.astype(float)).float(),
            torch.from_numpy(Y_mark.astype(float)).float(),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    preds = []
    trues = []
    norm_losses = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, -1]
            target = batch_y[:, -args.pred_len:, -1]

            norm_losses.append(criterion(outputs, target).item())
            preds.append(outputs.cpu().numpy())
            trues.append(target.cpu().numpy())

            del batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    print("The Mean Squared Error of forecasts is {}".format(float(np.mean(norm_losses))))
    print("The Mean Squared Error of forecasts is {} (raw)".format(get_mse(y_true, y_pred, yscaler)))
    print("The MAPE of forecasts is {} (raw)".format(get_mape(y_true, y_pred, yscaler)))
    print("The Mean Absolute Error of forecasts is {} (raw)".format(get_mae(y_true, y_pred, yscaler)))
    print("The SMAPE of forecasts is {}% (raw)".format(get_smape(y_true, y_pred, yscaler)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched official Informer2020 inference for ECW dynamic scenarios.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="saved_model/Informer2020_best.pt")
    parser.add_argument("--scaler_path", type=str, default="saved_model/informer_data_scaler.pkl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--gpu", type=int, default=0)
    main(parser.parse_args())
