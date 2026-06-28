import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import argparse
import math
from dyneformer_tune import DynEformer
from sklearn import preprocessing
import torch
import time
import pickle

from models.baseline.GlobalPooling.vade_pooling.GlobalPool import GlobalPool
from models.baseline.pickle_compat import register_globalpool_aliases

register_globalpool_aliases()
from models.baseline.global_utils import draw_data_plots



def smape_percent(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))

def batch_generator_padding(X, Y, num_obs_to_train, seq_len, step):
    '''
        Args:
        X (array like): shape (num_samples, num_features, num_periods)
        y (array like): shape (num_samples, num_periods)
        num_obs_to_train (int):
        seq_len (int): sequence/encoder/decoder length
        batch_size (int)
        '''
    num_ts, num_periods, n_feats = X.shape
    X_all = []
    Y_all = []

    for i in range(num_ts):
        for j in range(num_obs_to_train, num_periods - seq_len, step):
            X_all.append(X[i, j-num_obs_to_train:j+seq_len, :])
            Y_all.append(Y[i, j:j+seq_len])

    X_all = np.asarray(X_all).reshape(-1, num_obs_to_train+seq_len, n_feats)
    Y_all = np.asarray(Y_all).reshape(-1, seq_len)
    return X_all, Y_all


def reference(X, y, args):
    model = torch.load('saved_model/DynEformer_best_n551_drop01.pt')
    device = torch.device('cuda:0')

    num_ts, num_periods, num_features = X.shape
    input_size = 4  # encoder输入维度 只包括bw和日期mark
    with open(r'../GlobalPooling/pools/global_pool_c{}_s48_s12.pkl'.format(551), 'rb') as f:
        timeFeature_pool = pickle.load(f)
    sFeature = torch.tensor(np.stack(timeFeature_pool.seasonal_pool).T).float().to(device)
    timeFeature = sFeature
    model = model.to(device)
    model.eval()

    Xte = X
    yte = y

    xscaler, yscaler = pickle.load(open('saved_model/dyne_data_scaler_tmp.pkl', 'rb'))
    Xte = xscaler.transform(Xte.reshape(-1, num_features)).reshape(num_ts, -1, num_features)

    pre_seq_len = args.out_seq_len
    num_obs_to_train = args.enc_seq_len
    X_test_all, Y_test_all = batch_generator_padding(Xte, yte, num_obs_to_train, pre_seq_len, args.step)

    with torch.no_grad():
        Xtest, ytest = X_test_all, Y_test_all
        y_pred_batches = []

        for start in range(0, Xtest.shape[0], args.batch_size):
            end = start + args.batch_size
            Xtest_batch = torch.from_numpy(Xtest[start:end].astype(float)).float().to(device)
            Xtest_tensor = Xtest_batch[:, :, :4]
            Test_static_context = Xtest_batch[:, 0, 4:].squeeze(1)

            yPred_test = model(Xtest_tensor, timeFeature, Test_static_context)
            y_pred_batches.append(yPred_test.cpu().numpy())

            del Xtest_batch, Xtest_tensor, Test_static_context, yPred_test

        yPred_test = np.concatenate(y_pred_batches, axis=0)

        if yscaler is not None:
            yPred_test = yscaler.inverse_transform(yPred_test.reshape(-1, 1))

        test_mse = ((ytest.reshape(-1) - yPred_test.reshape(-1)) ** 2).mean()
        test_mae = np.abs(ytest.reshape(-1) - yPred_test.reshape(-1)).mean()
        test_smape = smape_percent(ytest.reshape(-1), yPred_test.reshape(-1))

        print('The Mean Squared Error of forecasts is {} (raw)'.format(test_mse))
        print('The Mean Absolute Error of forecasts is {} (raw)'.format(test_mae))
        print('The SMAPE of forecasts is {}% (raw)'.format(test_smape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-stp", type=int, default=2)

    parser.add_argument("--dec_seq_len", "-dl", type=int, default=12)  # decoder用到的输入长度
    parser.add_argument("--out_seq_len", "-ol", type=int, default=24)  # 预测长度
    parser.add_argument("--enc_seq_len", "-not", type=int, default=24 * 2)  # 输入训练长度
    parser.add_argument("--batch_size", "-b", type=int, default=64)  # inference batch size

    parser.add_argument("--run_test", "-rt", action="store_true", default=True)

    args = parser.parse_args()

    if args.run_test:
        X_all = pickle.load(open(r"../../../data/ECW_newmac.pkl", 'rb'))
        y_all = X_all[:, :, 0]

        reference(X_all, y_all, args)
