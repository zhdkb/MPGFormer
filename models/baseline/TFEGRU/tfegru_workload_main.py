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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from tfegru_model import TFEGRU


def load_workload(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = np.load(open(path, "rb"), allow_pickle=True)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected [num_series, num_periods, num_features], got {}".format(data.shape))
    return data


def use_dyn_azure_scale(*paths):
    names = " ".join(Path(path).name.lower() for path in paths if path)
    return "azure" in names or "tr_vm" in names or "te_vm" in names or "_vm" in names


def split_by_time(x, train_ratio=0.8):
    split = int(x.shape[1] * train_ratio)
    return x[:, :split, :], x[:, split:, :]


def inverse_scale(values, scaler):
    values = np.asarray(values)
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def smape_percent(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))


class WorkloadDataset(Dataset):
    def __init__(self, x_scaled, y_scaled, seq_len=48, pred_len=24, step=12):
        self.x = []
        self.y = []
        for series_id in range(x_scaled.shape[0]):
            for pos in range(seq_len, x_scaled.shape[1] - pred_len + 1, step):
                self.x.append(x_scaled[series_id, pos - seq_len:pos, :])
                self.y.append(y_scaled[series_id, pos:pos + pred_len])
        if not self.x:
            raise RuntimeError(
                "No windows built from shape {}. Need more than seq_len + pred_len points.".format(
                    x_scaled.shape
                )
            )
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()


def prepare_data(args):
    train_raw, val_raw, test_raw = split_workload(
        load_workload(args.train_data), args.train_data,
        load_workload(args.test_data) if args.test_data else None,
        load_workload(args.val_data) if args.val_data else None,
    )

    if train_raw.shape[-1] != test_raw.shape[-1]:
        raise ValueError("Train/test feature count mismatch: {} vs {}".format(train_raw.shape, test_raw.shape))

    train_raw = train_raw.copy()
    val_raw = val_raw.copy()
    test_raw = test_raw.copy()
    if use_dyn_azure_scale(args.train_data, args.test_data):
        train_raw[:, :, 0] = train_raw[:, :, 0] / 10.0
        val_raw[:, :, 0] = val_raw[:, :, 0] / 10.0
        test_raw[:, :, 0] = test_raw[:, :, 0] / 10.0

    input_size = min(args.input_size, train_raw.shape[-1]) if args.input_size else train_raw.shape[-1]
    train_selected = train_raw[:, :, :input_size]
    val_selected = val_raw[:, :, :input_size]
    test_selected = test_raw[:, :, :input_size]

    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()
    y_scaler.fit(train_selected[:, :, 0].reshape(-1, 1))

    n_train, t_train, f_train = train_selected.shape
    train_scaled = x_scaler.fit_transform(train_selected.reshape(-1, f_train)).reshape(n_train, t_train, f_train)
    n_val, t_val, _ = val_selected.shape
    val_scaled = x_scaler.transform(val_selected.reshape(-1, f_train)).reshape(n_val, t_val, f_train)
    n_test, t_test, _ = test_selected.shape
    test_scaled = x_scaler.transform(test_selected.reshape(-1, f_train)).reshape(n_test, t_test, f_train)

    y_train_scaled = y_scaler.transform(train_selected[:, :, 0].reshape(-1, 1)).reshape(n_train, t_train)
    y_val_scaled = y_scaler.transform(val_selected[:, :, 0].reshape(-1, 1)).reshape(n_val, t_val)
    y_test_scaled = y_scaler.transform(test_selected[:, :, 0].reshape(-1, 1)).reshape(n_test, t_test)

    return train_scaled, y_train_scaled, val_scaled, y_val_scaled, test_scaled, y_test_scaled, x_scaler, y_scaler


def evaluate(model, loader, criterion, device, y_scaler):
    model.eval()
    losses = []
    mses = []
    maes = []
    smapes = []
    batch_sizes = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred_all = model(x)
            pred = pred_all[:, :, 0]
            loss = criterion(pred, y)

            true_raw = inverse_scale(y.cpu().numpy(), y_scaler)
            pred_raw = inverse_scale(pred.cpu().numpy(), y_scaler)
            batch_sizes.append(len(y))
            losses.append(loss.item())
            mses.append(np.mean((pred_raw.reshape(-1) - true_raw.reshape(-1)) ** 2))
            maes.append(np.mean(np.abs(pred_raw.reshape(-1) - true_raw.reshape(-1))))
            smapes.append(smape_percent(true_raw, pred_raw))

    return float(mean_by_batch(losses, batch_sizes)), float(mean_by_batch(mses, batch_sizes)), float(mean_by_batch(maes, batch_sizes)), float(mean_by_batch(smapes, batch_sizes))


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_scaled, y_train_scaled, val_scaled, y_val_scaled, test_scaled, y_test_scaled, x_scaler, y_scaler = prepare_data(args)
    print("train data:", args.train_data, train_scaled.shape)
    print("validation data:", val_scaled.shape)
    print("test data:", args.test_data or "time split", test_scaled.shape)

    train_loader = DataLoader(
        WorkloadDataset(train_scaled, y_train_scaled, args.seq_len, args.pred_len, args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WorkloadDataset(val_scaled, y_val_scaled, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WorkloadDataset(test_scaled, y_test_scaled, args.seq_len, args.pred_len, args.test_step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = TFEGRU(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        input_size=train_scaled.shape[-1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        decomp_kernel=args.decomp_kernel,
    ).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=args.decay_rate)

    best_loss = float("inf")
    best_state = None
    best_mse = None
    best_mae = None
    best_smape = None
    best_epoch = None
    epochs_no_improve = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)[:, :, 0]
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        val_loss, val_mse, val_mae, val_smape = evaluate(model, val_loader, criterion, device, y_scaler)
        if val_mse < best_loss:
            best_loss = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_mse = val_mse
            best_mae = val_mae
            best_smape = val_smape
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "TFEGRU_workload_best.pt"))
            pickle.dump([x_scaler, y_scaler], open(os.path.join(args.save_dir, "tfegru_workload_scaler.pkl"), "wb"))
        else:
            epochs_no_improve += 1

        print(
            "epoch {}, train loss: {}, val loss: {}, mse: {}, mae: {}, smape(%): {}".format(
                epoch, float(np.mean(train_losses)), val_loss, val_mse, val_mae, val_smape
            )
        )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print("early stopping at epoch {}".format(epoch))
            break

    print(
        "best epoch: {}, best mse: {}, best mae: {}, best smape(%): {}".format(
            best_epoch, best_mse, best_mae, best_smape
        )
    )
    model.load_state_dict(best_state)
    test_loss, test_mse, test_mae, test_smape = evaluate(model, test_loader, criterion, device, y_scaler)
    print("test loss: {}, test mse: {}, test mae: {}, test smape(%): {}".format(
        test_loss, test_mse, test_mae, test_smape
    ))

    report_test_metrics('TFEGRU', args.seed, args.train_data, args.test_data, test_mse, test_mae, test_smape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFEGRU workload reproduction entry.")
    parser.add_argument("--train_data", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=12)
    parser.add_argument("--input_size", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--decomp_kernel", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
