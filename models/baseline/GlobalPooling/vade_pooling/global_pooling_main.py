import argparse
from dataloader import *
from model_vade import VaDE
import numpy as np
from protocol import pool_training_partition
from build_global_pool import build_gp



if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=40000)
    parse.add_argument('--dataX_dir', type=str, default='../../../../data/ECW_08.npy')
    parse.add_argument('--nClusters', type=int, default=500)

    parse.add_argument('--series_len', type=int, default=48)
    parse.add_argument('--step', type=int, default=12)

    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)
    parse.add_argument('--gmm_reg_covar', type=float, default=1e-4)
    parse.add_argument('--gmm_n_init', type=int, default=3)
    parse.add_argument('--seed', type=int, default=2)

    args=parse.parse_args()
    np.random.seed(args.seed)

    X = np.load(open(args.dataX_dir, 'rb'), allow_pickle=True)
    X = np.asarray(X)

    X_train = pool_training_partition(X, args.dataX_dir)[:, :, 0]

    vade = VaDE(args, X_train)
    if args.cuda:
        vade = vade.cuda()

    X_train_all_np, pre_res = vade.pre_train(pre_epoch=500)
    build_gp(X_train_all_np, pre_res, args)







