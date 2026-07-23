import argparse
import os
import pickle
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
from data import MPGFormerWindowDataset, chronological_split, chronological_train_val_test_split
from losses import MPGFormerLoss


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


def scale_dynamic(train, val, test, feature_count, target_col=0):
    """Scale dynamic features using training statistics only."""

    scaler = preprocessing.MinMaxScaler()
    target_scaler = preprocessing.MinMaxScaler()
    target_scaler.fit(train[:, :, target_col].reshape(-1, 1))

    n, t, f = train.shape
    train_dyn = scaler.fit_transform(train[:, :, :feature_count].reshape(-1, feature_count)).reshape(n, t, feature_count)

    n2, t2, _ = val.shape
    val_dyn = scaler.transform(val[:, :, :feature_count].reshape(-1, feature_count)).reshape(n2, t2, feature_count)

    n3, t3, _ = test.shape
    test_dyn = scaler.transform(test[:, :, :feature_count].reshape(-1, feature_count)).reshape(n3, t3, feature_count)

    train_out = train.copy()
    val_out = val.copy()
    test_out = test.copy()
    train_out[:, :, :feature_count] = train_dyn
    val_out[:, :, :feature_count] = val_dyn
    test_out[:, :, :feature_count] = test_dyn

    return train_out, val_out, test_out, scaler, target_scaler


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
    """Evaluate validation loss and inverse-scaled metrics."""

    model.eval()
    losses, mses, maes, smapes = [], [], [], []

    with torch.no_grad():
        for x, static, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            static = static.to(device) if static_dim > 0 else None
            out = model(x, static_context=static)
            loss, _parts = criterion(out, y)

            losses.append(loss.item())
            mses.append(get_mse(y, out.point, target_scaler))
            maes.append(get_mae(y, out.point, target_scaler))
            smapes.append(get_smape(y, out.point, target_scaler))

    return float(np.mean(losses)), float(np.mean(mses)), float(np.mean(maes)), float(np.mean(smapes))


def select_checkpoint_score(metric_name, val_loss, val_mse, val_mae):
    """Return the scalar used for best-checkpoint selection."""

    if metric_name == "loss":
        return val_loss
    if metric_name == "mae":
        return val_mae
    return val_mse


def main():
    """Train MPGFormer on a workload dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../../data/ECW_08.npy")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--enc_len", type=int, default=48)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--static_start", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--num_prototypes", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--select_metric", type=str, default="mse", choices=["mse", "loss", "mae"])
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_array(args.data)
    min_window = args.enc_len + args.pred_len

    # MPGFormer windows require a full encoder history plus prediction horizon.
    if args.test_data:
        train_periods = data.shape[1]
        train_len = int(train_periods * (1.0 - args.val_ratio))
        val_len = train_periods - train_len
        if train_len < min_window or val_len < min_window:
            max_val_ratio = 1.0 - min_window / train_periods
            raise ValueError(
                "MPGFormer needs at least enc_len + pred_len time points in both training and validation splits. "
                f"Got train_file_periods={train_periods}, train_len={train_len}, val_len={val_len}, "
                f"required_window={min_window}. Use --val_ratio no larger than {max_val_ratio:.3f}, "
                "or reduce --enc_len/--pred_len."
            )
        train_raw, val_raw = chronological_split(data, train_ratio=1.0 - args.val_ratio)
        test_raw = load_array(args.test_data)
        if train_raw.shape[-1] != test_raw.shape[-1]:
            raise ValueError(
                f"Train/test feature counts differ: train={train_raw.shape[-1]}, test={test_raw.shape[-1]}"
            )
        if train_raw.shape[1] < min_window or test_raw.shape[1] < min_window:
            raise ValueError(
                "MPGFormer needs at least enc_len + pred_len time points in both files. "
                f"Got train_periods={train_raw.shape[1]}, test_periods={test_raw.shape[1]}, "
                f"required_window={min_window}."
            )
    else:
        periods = data.shape[1]
        train_len = int(periods * args.train_ratio)
        val_len = int(periods * (args.train_ratio + args.val_ratio)) - train_len
        test_len = periods - train_len - val_len
        if train_len < min_window or val_len < min_window or test_len < min_window:
            lo = min_window / periods
            hi = 1.0 - 2.0 * lo
            raise ValueError(
                "MPGFormer needs at least enc_len + pred_len time points in train, validation, and test splits. "
                f"Got periods={periods}, train_len={train_len}, val_len={val_len}, test_len={test_len}, "
                f"required_window={min_window}. Use --train_ratio between {lo:.3f} and {hi:.3f} "
                f"with --val_ratio at least {lo:.3f}, "
                "or reduce --enc_len/--pred_len, or pass --test_data for a separate evaluation file."
            )
        train_raw, val_raw, test_raw = chronological_train_val_test_split(
            data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    if use_dyn_azure_scale(args.data, args.test_data):
        train_raw = train_raw.copy()
        val_raw = val_raw.copy()
        test_raw = test_raw.copy()
        train_raw[:, :, 0] = train_raw[:, :, 0] / 10.0
        val_raw[:, :, 0] = val_raw[:, :, 0] / 10.0
        test_raw[:, :, 0] = test_raw[:, :, 0] / 10.0

    print(f"train data: {train_raw.shape}")
    print(f"val data: {val_raw.shape}")
    print(f"test data: {test_raw.shape}")

    train_raw, val_raw, test_raw, _, target_scaler = scale_dynamic(train_raw, val_raw, test_raw, args.input_size)
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
        stride=args.label_len,
    )
    test_ds = MPGFormerWindowDataset(
        test_raw,
        enc_len=args.enc_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        input_size=args.input_size,
        static_start=args.static_start,
        stride=args.label_len,
    )
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"No training/evaluation windows were generated. train_windows={len(train_ds)}, "
            f"val_windows={len(val_ds)}, "
            f"test_windows={len(test_ds)}. Check --enc_len, --pred_len, --train_ratio, and --stride."
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
    best_path = save_dir / "MPGFormer_best.pt"

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

        # Select checkpoints by the requested metric. MSE is the default because
        # MPGFormer's composite loss intentionally includes MAE, quantile, peak,
        # and contrastive terms that are not always aligned with raw MSE.
        select_score = select_checkpoint_score(args.select_metric, val_loss, val_mse, val_mae)
        if select_score < best_score:
            best_score = select_score
            best_loss = val_loss
            best_mse = val_mse
            best_mae = val_mae
            best_smape = val_smape
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        # tqdm.write keeps the progress bar readable while preserving exact metric formatting.
        log_fn = getattr(progress, "write", print)
        log_fn(
            f"epoch {epoch:03d} train_loss={train_loss:.16f}, "
            f"val_loss={val_loss:.16f}, mse={val_mse:.16f}, mae={val_mae:.16f}, "
            f"smape(%)={val_smape:.16f}"
        )

    print(
        f"best_epoch={best_epoch:03d}, select_metric={args.select_metric}, best_score={best_score:.16f}, "
        f"best_val_loss={best_loss:.16f}, best_val_mse={best_mse:.16f}, best_val_mae={best_mae:.16f}, "
        f"best_val_smape(%)={best_smape:.16f}"
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_mse, test_mae, test_smape = evaluate(
        model, test_loader, criterion, device, static_dim, target_scaler
    )
    print(
        f"test_loss={test_loss:.16f}, test_mse={test_mse:.16f}, "
        f"test_mae={test_mae:.16f}, test_smape(%)={test_smape:.16f}"
    )


if __name__ == "__main__":
    main()
