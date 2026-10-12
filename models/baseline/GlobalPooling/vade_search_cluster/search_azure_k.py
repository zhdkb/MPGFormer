import argparse
import os
import random
from pathlib import Path

import numpy as np
from protocol import pool_training_partition
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from series_decomp import s_decomp


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def parse_candidates(value):
    if ":" in value:
        start, stop, step = [int(x) for x in value.split(":")]
        return list(range(start, stop + 1, step))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_windows(data, series_len, step, target_col):
    raw = np.load(open(data, "rb"), allow_pickle=True).astype(float)
    if raw.ndim == 3:
        raw = pool_training_partition(raw, data)
        workload = raw[:, :, target_col]
    elif raw.ndim == 2:
        workload = raw
    else:
        raise ValueError("Expected a 2D or 3D numpy array, got shape {}".format(raw.shape))

    seasonal = []
    for s in tqdm(workload, desc="STL seasonal"):
        seasonal.append(s_decomp(s, type="seasonal"))
    seasonal = np.asarray(seasonal)

    scaler = MinMaxScaler()
    seasonal = scaler.fit_transform(seasonal.T).T

    windows = []
    for i in range(seasonal.shape[0]):
        for j in range(series_len, seasonal.shape[1], step):
            windows.append(seasonal[i, j - series_len:j])

    windows = np.asarray(windows).reshape(-1, series_len).astype(np.float32)
    return windows


def block(in_c, out_c):
    return [nn.Linear(in_c, out_c), nn.ReLU(True)]


class Encoder(nn.Module):
    def __init__(self, input_dim=48, inter_dims=(500, 500, 2000), hid_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            *block(input_dim, inter_dims[0]),
            *block(inter_dims[0], inter_dims[1]),
            *block(inter_dims[1], inter_dims[2]),
        )
        self.mu_l = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        return self.mu_l(self.encoder(x))


class Decoder(nn.Module):
    def __init__(self, input_dim=48, inter_dims=(500, 500, 2000), hid_dim=10):
        super().__init__()
        self.decoder = nn.Sequential(
            *block(hid_dim, inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-3]),
            nn.Linear(inter_dims[-3], input_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(z)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, hid_dim=hid_dim)
        self.decoder = Decoder(input_dim=input_dim, hid_dim=hid_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    windows = make_windows(
        args.dataX_dir,
        args.series_len,
        args.step,
        args.target_col,
    )
    print("Pool search windows:", windows.shape)

    x = torch.from_numpy(windows).float().to(device)
    model = AutoEncoder(args.series_len, args.hid_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    epoch_bar = tqdm(range(args.pre_epoch), desc="AE pretrain")
    for _ in epoch_bar:
        recon, _ = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_bar.set_postfix(loss="{:.6f}".format(loss.item()))

    model.eval()
    with torch.no_grad():
        _, z = model(x)
    z = z.detach().cpu().numpy()

    candidates = [k for k in parse_candidates(args.candidates) if 1 < k < len(z)]
    rows = []
    best = None
    for k in tqdm(candidates, desc="GMM BIC search"):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            random_state=args.seed,
        )
        gmm.fit(z)
        bic = float(gmm.bic(z))
        aic = float(gmm.aic(z))
        rows.append((k, bic, aic))
        if best is None or bic < best[1]:
            best = (k, bic, aic)
        print("K={}, BIC={:.3f}, AIC={:.3f}".format(k, bic, aic))

    print("BEST_K_BY_BIC={} BIC={:.3f} AIC={:.3f}".format(best[0], best[1], best[2]))

    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(
            args.output_dir,
            "{}_k_search_s{}_s{}.csv".format(Path(args.dataX_dir).stem, args.series_len, args.step),
        )
        with open(out, "w", encoding="utf-8") as f:
            f.write("k,bic,aic\n")
            for k, bic, aic in rows:
                f.write("{},{},{}\n".format(k, bic, aic))
        print("Saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the global-pool K with AE latent GMM BIC.")
    parser.add_argument("--dataX_dir", type=str, default="../../../../data/tr_vm.npy")
    parser.add_argument("--series_len", type=int, default=48)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--target_col", type=int, default=0)
    parser.add_argument("--hid_dim", type=int, default=10)
    parser.add_argument("--pre_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--candidates", type=str, default="20,30,40,50,60,80,100,120,150,200,250,300,400,500")
    parser.add_argument("--n_init", type=int, default=3)
    parser.add_argument("--reg_covar", type=float, default=1e-6)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--save_csv", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="../pools")
    main(parser.parse_args())
