import argparse
import copy
import os
import pickle
import random
from pathlib import Path

import numpy as np
from protocol import split_workload, report_test_metrics, mean_by_batch
import torch
from sklearn import preprocessing
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dyneformer_tune import DynEformer
from models.baseline.pickle_compat import register_globalpool_aliases

register_globalpool_aliases()


class GoogleDataset(Dataset):
    def __init__(self, X, enc_len=48, pred_len=24, step=12):
        self.pred_len = pred_len
        num_ts, num_periods, num_features = X.shape
        seq_len = enc_len + pred_len
        samples = []

        for i in range(num_ts):
            for j in range(seq_len, num_periods + 1, step):
                samples.append(X[i, j - seq_len:j, :])

        if not samples:
            raise RuntimeError(
                "No windows built from data shape {}. Need num_periods > enc_len + pred_len.".format(X.shape)
            )
        self.X = np.stack(samples).reshape(-1, seq_len, num_features).astype(float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.X[index, -self.pred_len:, 0]


def inverse_scale(values, scaler):
    values = np.asarray(values)
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


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
def load_global_pool(args, device):
    pool_path = os.path.join(
        "..",
        "GlobalPooling",
        "pools",
        "global_pool_c{}_s{}_s{}.pkl".format(args.pool_clusters, args.enc_seq_len, args.pool_step),
    )
    with open(pool_path, "rb") as f:
        global_pool = pickle.load(f)

    seasonal_pool = np.stack(global_pool.seasonal_pool).T
    return torch.tensor(seasonal_pool).float().to(device)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    Xtr, Xva, Xte = split_workload(
        np.load(args.train_data, allow_pickle=True).astype(float), args.train_data,
        np.load(args.test_data, allow_pickle=True).astype(float),
        np.load(args.val_data, allow_pickle=True).astype(float) if args.val_data else None,
    )
    ytr = Xtr[:, :, 0]

    num_ts, num_periods, num_features = Xtr.shape
    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()

    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    num_ts_te, num_periods_te, _ = Xte.shape
    num_ts_va, num_periods_va, _ = Xva.shape
    Xva = xscaler.transform(Xva.reshape(-1, num_features)).reshape(num_ts_va, num_periods_va, num_features)
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts_te, num_periods_te, num_features)
    yscaler.fit(ytr.reshape(-1, 1))

    train_loader = DataLoader(
        GoogleDataset(Xtr, enc_len=args.enc_seq_len, pred_len=args.out_seq_len, step=args.step),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        GoogleDataset(Xva, enc_len=args.enc_seq_len, pred_len=args.out_seq_len, step=args.step),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        GoogleDataset(Xte, enc_len=args.enc_seq_len, pred_len=args.out_seq_len, step=args.step),
        batch_size=args.batch_size,
        shuffle=False,
    )

    global_pool = load_global_pool(args, device)
    time_feature_len = global_pool.shape[-1]
    print("Google train data:", args.train_data, Xtr.shape)
    print("Google test data:", args.test_data, Xte.shape)
    print("Global pool:", global_pool.shape)

    model = DynEformer(
        args.d_model,
        args.d_model,
        args.dim_feedforward,
        num_features,
        args.enc_seq_len,
        args.dec_seq_len,
        args.out_seq_len,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        n_heads=args.nhead,
        dropout=args.dropout,
        activation=args.activation,
        time_feature_len=time_feature_len,
        if_padding=args.if_padding,
    ).to(device)

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
            for x, y in loader:
                x = x.float().to(device)
                y = y.float().to(device)

                y_pred = model(x[:, :, :num_features], global_pool, None)
                loss = criterion(y_pred, y)

                batch_sizes.append(len(y))
                test_losses.append(loss.item())
                test_mses.append(get_mse(y.cpu(), y_pred.cpu(), yscaler))
                test_maes.append(get_mae(y.cpu(), y_pred.cpu(), yscaler))
                test_smapes.append(get_smape(y.cpu(), y_pred.cpu(), yscaler))

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

    for epoch in tqdm(range(args.num_epoches)):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)

            y_pred = model(x[:, :, :num_features], global_pool, None)
            loss = criterion(y_pred, y)

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
            if args.save_model:
                torch.save(
                    model,
                    os.path.join(args.save_dir, "DynEformer_google_best_c{}.pt".format(args.pool_clusters)),
                )
                pickle.dump(
                    [xscaler, yscaler],
                    open(os.path.join(args.save_dir, "dyne_google_data_scaler.pkl"), "wb"),
                )

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

    print("best epoch: {}, best mse: {}, best mae: {}, best smape(%): {}".format(best_epoch, best_mse, best_mae, best_smape))

    model.load_state_dict(best_state)
    test_loss, test_mse, test_mae, test_smape = evaluate(test_loader)
    print("test loss: {}, test mse: {}, test mae: {}, test smape(%): {}".format(
        test_loss, test_mse, test_mae, test_smape
    ))

    report_test_metrics('DynEformer', args.seed, args.train_data, args.test_data, test_mse, test_mae, test_smape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DynEformer on Google Cluster Trace arrays.")
    parser.add_argument("--train_data", type=str, default="../../../data/tr_google_v1.npy")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="../../../data/te_google_v1.npy")
    parser.add_argument("--pool_clusters", type=int, default=100)
    parser.add_argument("--pool_step", type=int, default=12)
    parser.add_argument("--save_dir", type=str, default="saved_model")

    parser.add_argument("--num_epoches", "-e", type=int, default=300)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--step", type=int, default=12)

    parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
    parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)
    parser.add_argument("--nhead", "-nh", type=int, default=8)
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=512)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)
    parser.add_argument("--enc_seq_len", "-not", type=int, default=48)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-activation", type=str, default="relu")
    parser.add_argument("-if_padding", type=bool, default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
