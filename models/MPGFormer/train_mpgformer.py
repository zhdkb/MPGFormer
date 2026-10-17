import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from mpgformer import MPGFormer
from data import MPGFormerWindowDataset
from losses import MPGFormerLoss
from protocol import split_workload, report_test_metrics, mean_by_batch


def use_dyn_azure_scale(*paths):
    """Match DynEformer convention: Azure/VM workload targets are divided by 10."""

    names = " ".join(Path(path).name.lower() for path in paths if path)
    return "azure" in names or "_vm" in names or names.startswith("tr_vm") or "tr_vm" in names or "te_vm" in names


def load_array(path):
    """Load a 3-D workload tensor from pickle or NumPy format."""

    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = np.load(path, allow_pickle=True)

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Expected data with shape [num_series, num_periods, num_features], got {data.shape}")

    return data.astype(np.float32)


def scale_dynamic(train, test, feature_count, target_col=0):
    """Fit scalers on training data and apply the feature scaler to test data."""

    scaler = preprocessing.MinMaxScaler()
    target_scaler = preprocessing.MinMaxScaler()
    target_scaler.fit(train[:, :, target_col].reshape(-1, 1))

    n, t, f = train.shape
    train_dyn = scaler.fit_transform(train[:, :, :feature_count].reshape(-1, feature_count)).reshape(n, t, feature_count)

    n2, t2, _ = test.shape
    test_dyn = scaler.transform(test[:, :, :feature_count].reshape(-1, feature_count)).reshape(n2, t2, feature_count)

    train_out = train.copy()
    test_out = test.copy()
    train_out[:, :, :feature_count] = train_dyn
    test_out[:, :, :feature_count] = test_dyn

    return train_out, test_out, scaler, target_scaler


def inverse_target(values, target_scaler):
    """Undo target scaling before reporting MSE/MAE."""

    arr = values.detach().cpu().numpy()
    inv = target_scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)
    return inv


def get_mse(y_true, y_pred, target_scaler):
    """Compute MSE on the original workload scale."""

    y_true = inverse_target(y_true, target_scaler)
    y_pred = inverse_target(y_pred, target_scaler)
    return np.mean((y_true - y_pred) ** 2)


def get_mae(y_true, y_pred, target_scaler):
    """Compute MAE on the original workload scale."""

    y_true = inverse_target(y_true, target_scaler)
    y_pred = inverse_target(y_pred, target_scaler)
    return np.mean(np.abs(y_true - y_pred))




def get_smape(y_true, y_pred, target_scaler):
    """Compute SMAPE (%) on the original workload scale."""

    y_true = inverse_target(y_true, target_scaler)
    y_pred = inverse_target(y_pred, target_scaler)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))


def evaluate(model, loader, criterion, device, static_dim, target_scaler):
    """Average loss and inverse-scaled metrics by sample count."""

    model.eval()
    losses, mses, maes, smapes = [], [], [], []

    batch_sizes = []
    with torch.no_grad():
        for x, static, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            static = static.to(device) if static_dim > 0 else None
            out = model(x, static_context=static)
            loss, _parts = criterion(out, y)

            batch_sizes.append(len(y))
            losses.append(loss.item())
            mses.append(get_mse(y, out.point, target_scaler))
            maes.append(get_mae(y, out.point, target_scaler))
            smapes.append(get_smape(y, out.point, target_scaler))

    return float(mean_by_batch(losses, batch_sizes)), float(mean_by_batch(mses, batch_sizes)), float(mean_by_batch(maes, batch_sizes)), float(mean_by_batch(smapes, batch_sizes))


def main():
    """Train, select by validation MSE, and test the selected checkpoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../../data/ECW_08.npy")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--enc_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--test_step", type=int, default=12)
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--static_start", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--num_prototypes", type=int, default=48)
    parser.add_argument("--graph_topk", type=int, default=8)
    parser.add_argument("--disable", nargs="*", default=[], choices=(
        "revin", "temperature", "short_encoder", "prototype", "graph", "router",
        "workload_graph", "context_graph", "prototype_entropy", "drift_bias"))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--point_weight", type=float, default=0.9)
    parser.add_argument("--mse_weight", type=float, default=0.6)
    parser.add_argument("--tail_mse_weight", type=float, default=0.5)
    parser.add_argument("--target_tail_mse_weight", type=float, default=0.2)
    parser.add_argument("--quantile_weight", type=float, default=0.3)
    parser.add_argument("--peak_weight", type=float, default=0.2)
    parser.add_argument("--contrastive_weight", type=float, default=0.04)
    parser.add_argument("--peak_ratio", type=float, default=0.85)
    parser.add_argument("--tail_mse_ratio", type=float, default=0.9)
    parser.add_argument("--target_tail_mse_ratio", type=float, default=0.9)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_array(args.data)
    min_window = args.enc_len + args.pred_len

    train_raw, val_raw, test_raw = split_workload(
        data, args.data,
        load_array(args.test_data) if args.test_data else None,
        load_array(args.val_data) if args.val_data else None,
    )
    if min(train_raw.shape[1], val_raw.shape[1], test_raw.shape[1]) < min_window:
        raise ValueError(f"Each split needs at least {min_window} time points")

    if use_dyn_azure_scale(args.data, args.test_data):
        train_raw = train_raw.copy()
        val_raw = val_raw.copy()
        test_raw = test_raw.copy()
        train_raw[:, :, 0] = train_raw[:, :, 0] / 10.0
        val_raw[:, :, 0] = val_raw[:, :, 0] / 10.0
        test_raw[:, :, 0] = test_raw[:, :, 0] / 10.0

    print(f"train data: {train_raw.shape}")
    print(f"validation data: {val_raw.shape}")
    print(f"test data: {test_raw.shape}")

    train_raw, test_raw, feature_scaler, target_scaler = scale_dynamic(train_raw, test_raw, args.input_size)
    # Use the training scaler for validation as well.
    val_raw[:, :, :args.input_size] = feature_scaler.transform(
        val_raw[:, :, :args.input_size].reshape(-1, args.input_size)
    ).reshape(val_raw.shape[0], val_raw.shape[1], args.input_size)
    static_dim = max(0, data.shape[-1] - args.static_start)

    # Dataset objects mask future target values while keeping known future covariates.
    train_ds = MPGFormerWindowDataset(
        train_raw,
        enc_len=args.enc_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        input_size=args.input_size,
        static_start=args.static_start,
        stride=args.label_len,
    )
    val_ds = MPGFormerWindowDataset(
        val_raw,
        enc_len=args.enc_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        input_size=args.input_size,
        static_start=args.static_start,
        stride=args.test_step,
    )
    test_ds = MPGFormerWindowDataset(
        test_raw,
        enc_len=args.enc_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        input_size=args.input_size,
        static_start=args.static_start,
        stride=args.test_step,
    )
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"No windows were generated: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model hyperparameters mirror the other baselines where possible.
    model = MPGFormer(
        input_size=args.input_size,
        enc_seq_len=args.enc_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        dropout=args.dropout,
        static_dim=static_dim,
        num_prototypes=args.num_prototypes,
        graph_topk=args.graph_topk,
        disabled_components=args.disable,
    ).to(device)
    criterion = MPGFormerLoss(
        point_weight=args.point_weight,
        mse_weight=args.mse_weight,
        tail_mse_weight=args.tail_mse_weight,
        target_tail_mse_weight=args.target_tail_mse_weight,
        quantile_weight=args.quantile_weight,
        peak_weight=args.peak_weight,
        contrastive_weight=args.contrastive_weight,
        peak_ratio=args.peak_ratio,
        tail_mse_ratio=args.tail_mse_ratio,
        target_tail_mse_ratio=args.target_tail_mse_ratio,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    best_loss = float("nan")
    best_mse = float("nan")
    best_mae = float("nan")
    best_smape = float("nan")
    best_epoch = -1
    progress = tqdm(range(args.epochs))

    for epoch in progress:
        model.train()
        train_losses = []

        for x, static, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            static = static.to(device) if static_dim > 0 else None

            out = model(x, static_context=static)
            loss, _ = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, val_mse, val_mae, val_smape = evaluate(
            model, val_loader, criterion, device, static_dim, target_scaler
        )

        # Validation MSE alone selects the checkpoint.
        select_score = val_mse
        if select_score < best_score:
            best_score = select_score
            best_loss = val_loss
            best_mse = val_mse
            best_mae = val_mae
            best_smape = val_smape
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / "MPGFormer_best.pt")

        # tqdm.write keeps the progress bar readable while preserving exact metric formatting.
        log_fn = getattr(progress, "write", print)
        log_fn(
            f"epoch {epoch:03d} train_loss={train_loss:.16f}, "
            f"val_loss={val_loss:.16f}, mse={val_mse:.16f}, mae={val_mae:.16f}, "
            f"smape(%)={val_smape:.16f}"
        )

    print(
        f"best_epoch={best_epoch:03d}, select_metric=mse, best_score={best_score:.16f}, "
        f"best_loss={best_loss:.16f}, best_mse={best_mse:.16f}, best_mae={best_mae:.16f}, "
        f"best_smape(%)={best_smape:.16f}"
    )
    model.load_state_dict(torch.load(save_dir / "MPGFormer_best.pt", map_location=device))
    test_loss, test_mse, test_mae, test_smape = evaluate(
        model, test_loader, criterion, device, static_dim, target_scaler
    )
    print(f"test_loss={test_loss:.16f}, test_mse={test_mse:.16f}, "
          f"test_mae={test_mae:.16f}, test_smape(%)={test_smape:.16f}")

    report_test_metrics('MPGFormer', args.seed, args.data, args.test_data, test_mse, test_mae, test_smape)


if __name__ == "__main__":
    main()
