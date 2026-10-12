import argparse
import os
from glob import glob

import numpy as np
import pandas as pd


TRACE_CONFIGS = {
    "mt1": {
        "trace_dir": "Materna-Trace-1",
        "train_name": "tr_materna_mt1.npy",
        "val_name": "va_materna_mt1.npy",
        "test_name": "te_materna_mt1.npy",
        "paper_count": 413,
    },
    "mt2": {
        "trace_dir": "Materna-Trace-2",
        "train_name": "tr_materna_mt2.npy",
        "val_name": "va_materna_mt2.npy",
        "test_name": "te_materna_mt2.npy",
        "paper_count": 402,
    },
    "mt3": {
        "trace_dir": "Materna-Trace-3",
        "train_name": "tr_materna_mt3.npy",
        "val_name": "va_materna_mt3.npy",
        "test_name": "te_materna_mt3.npy",
        "paper_count": 371,
    },
}

RAW_COLUMNS = [
    "Timestamp",
    "CPU usage [MHZ]",
    "CPU usage [%]",
    "Memory usage [KB]",
    "Memory usage [%]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def read_trace_csv(path):
    frame = pd.read_csv(path, sep=";", decimal=",", usecols=RAW_COLUMNS, encoding="utf-8-sig")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    frame = frame.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    frame = frame.drop_duplicates(subset=["Timestamp"], keep="first")
    for column in RAW_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def time_features(index):
    elapsed = index - index[0]
    day = np.asarray(elapsed.days, dtype=np.float32)
    hour = np.asarray(index.hour, dtype=np.float32)
    minute = np.asarray(index.minute, dtype=np.float32)
    return day, hour, minute


def build_vm_series(path, grid, max_missing_rate):
    frame = read_trace_csv(path)
    if frame.empty:
        return None, 1.0

    frame = frame.set_index("Timestamp").reindex(grid)
    missing_rate = float(frame["CPU usage [%]"].isna().mean())
    if missing_rate > max_missing_rate:
        return None, missing_rate

    # Fill from past observations only, including across chronological splits.
    frame = frame.ffill().fillna(0.0)
    if frame[RAW_COLUMNS[1:]].isna().any().any():
        return None, missing_rate

    day, hour, minute = time_features(grid)
    values = np.column_stack(
        [
            frame["CPU usage [%]"].to_numpy(dtype=np.float32),
            day,
            hour,
            minute,
            frame["CPU usage [MHZ]"].to_numpy(dtype=np.float32),
            frame["Memory usage [KB]"].to_numpy(dtype=np.float32) / 1024.0,
            frame["Memory usage [%]"].to_numpy(dtype=np.float32),
            frame["Disk read throughput [KB/s]"].to_numpy(dtype=np.float32),
            frame["Disk write throughput [KB/s]"].to_numpy(dtype=np.float32),
            frame["Network received throughput [KB/s]"].to_numpy(dtype=np.float32)
            + frame["Network transmitted throughput [KB/s]"].to_numpy(dtype=np.float32),
        ]
    )
    return values.astype(np.float32), missing_rate


def trace_grid(paths):
    starts = []
    ends = []
    for path in paths:
        frame = pd.read_csv(path, sep=";", usecols=["Timestamp"], encoding="utf-8-sig")
        timestamps = pd.to_datetime(frame["Timestamp"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        timestamps = timestamps.dropna()
        if timestamps.empty:
            continue
        starts.append(timestamps.min())
        ends.append(timestamps.max())
    if not starts:
        raise RuntimeError("No valid timestamps found.")
    return pd.date_range(min(starts), max(ends), freq="5min")


def split_by_time(data, train_ratio, val_ratio):
    periods = data.shape[1]
    train_end = int(periods * train_ratio)
    val_end = int(periods * (train_ratio + val_ratio))
    return data[:, :train_end], data[:, train_end:val_end], data[:, val_end:]


def prepare_trace(trace_key, args):
    config = TRACE_CONFIGS[trace_key]
    raw_dir = os.path.join(args.raw_root, config["trace_dir"])
    paths = sorted(glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError("No CSV files found under {}".format(raw_dir))

    grid = trace_grid(paths)
    kept = []
    kept_rates = []
    skipped = 0
    for path in paths:
        series, missing_rate = build_vm_series(path, grid, args.max_missing_rate)
        if series is None:
            skipped += 1
            continue
        kept.append(series)
        kept_rates.append(missing_rate)

    if args.match_fremer_counts:
        target_count = config["paper_count"]
        if len(kept) < target_count:
            raise RuntimeError(
                "{} only has {} valid VMs after filtering; cannot match paper count {}.".format(
                    trace_key, len(kept), target_count
                )
            )
        order = np.argsort(np.asarray(kept_rates))
        kept = [kept[i] for i in order[:target_count]]
        kept_rates = [kept_rates[i] for i in order[:target_count]]

    data = np.stack(kept).astype(np.float32)
    train, val, test = split_by_time(data, args.train_ratio, args.val_ratio)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, config["train_name"])
    val_path = os.path.join(args.output_dir, config["val_name"])
    test_path = os.path.join(args.output_dir, config["test_name"])
    np.save(train_path, train)
    np.save(val_path, val)
    np.save(test_path, test)

    print("{}:".format(trace_key))
    print("  raw dir:", raw_dir)
    print("  full grid:", len(grid), grid[0], grid[-1])
    print("  kept VMs:", data.shape[0], "skipped:", skipped)
    print("  saved train:", train_path, train.shape)
    print("  saved val:", val_path, val.shape)
    print("  saved test:", test_path, test.shape)
    print("  target min/max/mean:", float(data[:, :, 0].min()), float(data[:, :, 0].max()), float(data[:, :, 0].mean()))
    print("  max missing rate kept:", float(max(kept_rates)) if kept_rates else 0.0)


def main(args):
    trace_keys = list(TRACE_CONFIGS) if args.trace == "all" else [args.trace]
    for trace_key in trace_keys:
        prepare_trace(trace_key, args)
    print(
        "Feature layout: [cpu_percent_target, day, hour, minute, cpu_mhz, memory_mb, "
        "memory_percent, disk_read_kbps, disk_write_kbps, network_total_kbps]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the three GWA-T-13 Materna traces separately.")
    parser.add_argument("--raw_root", type=str, default="data/GWA-T-13_Materna-Workload-Traces")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--trace", choices=["all", "mt1", "mt2", "mt3"], default="all")
    parser.add_argument("--max_missing_rate", type=float, default=0.02)
    parser.add_argument("--match_fremer_counts", type=str2bool, default=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    main(parser.parse_args())
