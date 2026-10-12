"""Dataset partitions and metric records for forecasting experiments."""

import json
from pathlib import Path

import numpy as np


def split_workload(train, train_path, test=None, validation=None, split_seed=0):
    train = np.asarray(train)
    if train.ndim != 3:
        raise ValueError("Expected a [series, time, features] training array")
    if validation is not None:
        if test is None:
            raise ValueError("An explicit validation file requires a separate test file")
        tr, va, te = train, np.asarray(validation), np.asarray(test)
    elif test is None:
        # ECW: chronological 64/16/20 of the complete source trace.
        n = train.shape[1]
        a, b = int(0.64 * n), int(0.80 * n)
        tr, va, te = train[:, :a], train[:, a:b], train[:, b:]
    elif "ecw" in Path(train_path).name.lower():
        # New Infras./New App: source trace 80/20, independent test trace.
        a = int(0.8 * train.shape[1])
        tr, va, te = train[:, :a], train[:, a:], np.asarray(test)
    else:
        # Azure/Google: reserve 20% of training series IDs, fixed across seeds.
        if train.shape[0] < 2:
            raise ValueError("Series holdout requires at least two training series")
        ids = np.random.default_rng(split_seed).permutation(train.shape[0])
        n_val = max(1, int(np.ceil(0.2 * len(ids))))
        tr, va, te = train[ids[n_val:]], train[ids[:n_val]], np.asarray(test)
    if tr.shape[2] != va.shape[2] or tr.shape[2] != te.shape[2]:
        raise ValueError("Train, validation and test feature counts differ")
    if min(len(tr), len(va), len(te)) == 0 or min(tr.shape[1], va.shape[1], te.shape[1]) == 0:
        raise ValueError("Train, validation and test splits must all be nonempty")
    return tr.copy(), va.copy(), te.copy()


def pool_training_partition(data, train_path, split_seed=0, source_train_ratio=None):
    """Return the training observations used to fit a DynEformer global pool."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("Expected a [series, time, features] array")
    name = Path(train_path).name.lower()
    if "ecw" in name:
        ratio = 0.64 if source_train_ratio is None else source_train_ratio
        if not 0 < ratio < 1:
            raise ValueError("Source training ratio must be between zero and one")
        return data[:, :int(ratio * data.shape[1])].copy()
    if "materna" in name:
        return data.copy()
    ids = np.random.default_rng(split_seed).permutation(data.shape[0])
    n_val = max(1, int(np.ceil(0.2 * len(ids))))
    if n_val == len(ids):
        raise ValueError("Series holdout requires at least two training series")
    return data[ids[n_val:]].copy()


def report_test_metrics(model, seed, train_path, test_path, mse, mae, smape):
    """Emit one machine-readable record after the held-out test evaluation."""
    if not np.isfinite([mse, mae, smape]).all():
        raise ValueError("Test metrics must be finite")
    dataset = Path(test_path).stem if test_path else "ECW"
    print("TEST_METRICS " + json.dumps({
        "model": model,
        "dataset": dataset,
        "seed": int(seed),
        "mse": float(mse),
        "mae": float(mae),
        "smape": float(smape),
    }, sort_keys=True))


def mean_by_batch(values, batch_sizes):
    """Average per-batch means over samples, including the final partial batch."""
    if not values or len(values) != len(batch_sizes):
        raise ValueError("Expected one value and a positive size for every batch")
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("Batch sizes must be positive")
    return float(np.average(values, weights=batch_sizes))
