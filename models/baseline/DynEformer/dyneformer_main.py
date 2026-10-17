import numpy as np
import copy
import os
from protocol import split_workload, report_test_metrics, mean_by_batch
import pandas as pd
import random
from torch import nn
import argparse
import pickle
from dyneformer_tune import DynEformer
from sklearn import preprocessing
from tqdm import tqdm
from torch.optim import Adam
import torch
import time
from torch.utils.data import DataLoader, Dataset

# from models.baseline.data_process_utils import *
from models.baseline.GlobalPooling.vade_pooling.GlobalPool import GlobalPool
from models.baseline.pickle_compat import register_globalpool_aliases

register_globalpool_aliases()


def get_mape(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs((yTrue - yPred) / yTrue) * 100)


def get_mse(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)
    return np.mean((yTrue - yPred) ** 2)


def get_mae(yTrue, yPred, scaler):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs(yTrue - yPred))




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
class PPIO_Dataset(Dataset):
    def __init__(self, X, enc_len=48, label_len=12, pred_len=24, num_static=12, step=12):
        self.pred_len = pred_len
        num_ts, num_periods, num_features = X.shape
        sq_len = enc_len + pred_len
        X_train_all = []
        X.astype(float)

        for i in range(num_ts):
            for j in range(sq_len, num_periods + 1, step):
                X_train_all.append(X[i, j - sq_len:j, :])

        self.X = np.stack(X_train_all).reshape(-1, sq_len, num_features)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.X[index, -self.pred_len:, 0]


def train(X, y, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs('saved_model', exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_size = 4
    cn = 551
    print('cn: {}'.format(cn))
    with open(r'../GlobalPooling/pools/global_pool_c{}_s48_s12.pkl'.format(cn), 'rb') as f:
        globalPool_pool = pickle.load(f)
    sFeature = torch.tensor(np.stack(globalPool_pool.seasonal_pool).T).float().to(device)

    globalPool = sFeature
    P = globalPool.shape[-1]

    model = DynEformer(args.d_model, args.d_model, args.dim_feedforward, input_size, args.enc_seq_len, args.dec_seq_len,
                        args.out_seq_len, n_encoder_layers=args.n_encoder_layers, n_decoder_layers=args.n_decoder_layers,
                        n_heads=args.nhead, dropout=args.dropout, activation=args.activation, time_feature_len=P,
                        if_padding=args.if_padding)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    test_data = None
    if args.test_data:
        with open(args.test_data, 'rb') as stream:
            test_data = pickle.load(stream) if args.test_data.endswith('.pkl') else np.load(stream, allow_pickle=True)
    Xtr, Xva, Xte = split_workload(X, args.data, test_data)
    ytr = Xtr[:, :, 0]
    xscaler = preprocessing.MinMaxScaler()
    yscaler = preprocessing.MinMaxScaler()
    num_ts, num_periods, num_features = Xtr.shape
    Xtr = xscaler.fit_transform(Xtr.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    num_ts, num_periods, num_features = Xva.shape
    Xva = xscaler.transform(Xva.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    num_ts, num_periods, num_features = Xte.shape
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, num_periods, num_features)
    yscaler.fit(ytr.reshape(-1, 1))
    Xtr_loader = DataLoader(PPIO_Dataset(Xtr, step=args.step), batch_size=args.batch_size)
    Xva_loader = DataLoader(PPIO_Dataset(Xva, step=args.test_step), batch_size=args.batch_size)
    Xte_loader = DataLoader(PPIO_Dataset(Xte, step=args.test_step), batch_size=args.batch_size)

    pickle.dump([xscaler, yscaler], open('saved_model/dyne_data_scaler_tmp.pkl', 'wb'))

    losses = []
    mse = nn.MSELoss().to(device)

    def evaluate(loader):
        epo_test_losses = []
        epo_mse = []
        epo_mae = []
        epo_smape = []
        model.eval()
        batch_sizes = []
        with torch.no_grad():
            for x, y in loader:
                x_load = x.float().to(device)[:, :, :4]
                x_static = x.float().to(device)[:, 0, 4:].squeeze(1)
                y = y.float().to(device)

                yPred = model(x_load, globalPool, x_static)

                loss = mse(yPred, y)

                batch_sizes.append(len(y))
                epo_test_losses.append(loss.item())
                epo_mse.append(get_mse(yPred.cpu(), y.cpu(), yscaler))
                epo_mae.append(get_mae(yPred.cpu(), y.cpu(), yscaler))
                epo_smape.append(get_smape(yPred.cpu(), y.cpu(), yscaler))

        epoch_loss = float(mean_by_batch(epo_test_losses, batch_sizes))
        epoch_mse = float(mean_by_batch(epo_mse, batch_sizes))
        epoch_mae = float(mean_by_batch(epo_mae, batch_sizes))
        epoch_smape = float(mean_by_batch(epo_smape, batch_sizes))
        return epoch_loss, epoch_mse, epoch_mae, epoch_smape



    min_loss = float("inf")
    best_state = None
    best_epoch = None
    # training
    for epoch in tqdm(range(args.num_epoches)):
        model.train()
        train_epoch_loss = []
        for Xtrain, ytrain in Xtr_loader:
            Xtrain_tensor = Xtrain.float().to(device)[:, :, :4]
            Train_static_context = Xtrain.float().to(device)[:, 0, 4:].squeeze(1)
            ytrain_tensor = ytrain.float().to(device)

            ypred = model(Xtrain_tensor, globalPool, Train_static_context)
            loss = mse(ypred, ytrain_tensor)
            train_epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(np.average(train_epoch_loss))

        val_loss, val_mse, val_mae, val_smape = evaluate(Xva_loader)
        if val_mse < min_loss:
            min_loss = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            if args.save_model:
                torch.save(model, 'saved_model/DynEformer_best_n{}_drop01.pt'.format(cn))
        print(f"epoch {epoch}, train loss: {losses[-1]}, val loss: {val_loss}, mse: {val_mse}, mae: {val_mae}, smape(%): {val_smape}")
    model.load_state_dict(best_state)
    test_metrics = evaluate(Xte_loader)
    print('best epoch: {}, validation mse: {}'.format(best_epoch, min_loss))
    print('test loss: {}, test mse: {}, test mae: {}, test smape(%): {}'.format(*test_metrics))
    report_test_metrics('DynEformer', args.seed, args.data, args.test_data, *test_metrics[1:])
    return losses, *test_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../../../data/ECW_08.npy")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--test_step", type=int, default=12)
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--minmax_scaler", "-mm", action="store_true", default=True)

    parser.add_argument("--num_epoches", "-e", type=int, default=300)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--n_encoder_layers", "-nel", type=int, default=2)
    parser.add_argument("--n_decoder_layers", "-ndl", type=int, default=1)
    parser.add_argument("--d_model", "-dm", type=int, default=256)
    parser.add_argument("--nhead", "-nh", type=int, default=8)
    parser.add_argument("--dim_feedforward", "-hs", type=int, default=512)
    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24*2)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-activation", type=str, default='relu')
    parser.add_argument("-if_padding", type=bool, default=True)

    parser.add_argument("--run_test", "-rt", action="store_true", default=True)
    parser.add_argument("--save_model", "-sm", type=bool, default=True)
    parser.add_argument("--load_model", "-lm", type=bool, default=False)
    parser.add_argument("--show_plot", "-sp", type=bool, default=False)

    args = parser.parse_args()

    if args.run_test:
        X_all = np.load(open(args.data, 'rb'), allow_pickle=True)
        y_all = X_all[:, :, 0]
        losses, test_losses, mse_l, mae_l, smape_l = train(X_all, y_all, args)
