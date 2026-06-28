import argparse
import gzip
import hashlib
import os
import random
import shutil
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd


GOOGLE_V1_URL = "https://commondatastorage.googleapis.com/clusterdata-misc/google-cluster-data-1.csv.gz"


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def download_file(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    print("Downloading:", url)
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp_path, out_path)


def stable_bucket(value, num_buckets):
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def timestamp_features(timestamp):
    day = timestamp // 86400
    hour = (timestamp % 86400) // 3600
    minute = (timestamp % 3600) // 60
    return day, hour, minute


def read_google_v1_as_buckets(path, num_buckets, bucket_key, bucket_agg):
    names = ["timestamp", "job_id", "task_id", "job_type", "cpu", "memory"]
    bucket_stats = defaultdict(lambda: [0.0, 0])
    memory_stats = defaultdict(lambda: [0.0, 0])
    type_stats = defaultdict(lambda: [0.0, 0])

    key_col = "job_id" if bucket_key == "job" else "task_id"
    with gzip.open(path, "rt") as f:
        chunks = pd.read_csv(
            f,
            sep=r"\s+",
            header=None,
            names=names,
            chunksize=1_000_000,
        )
        for chunk in chunks:
            chunk = chunk[["timestamp", key_col, "job_type", "cpu", "memory"]].dropna()
            for column in ["timestamp", key_col, "job_type", "cpu", "memory"]:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            chunk = chunk.dropna()
            chunk["bucket"] = chunk[key_col].map(lambda value: stable_bucket(value, num_buckets))
            grouped = chunk.groupby(["bucket", "timestamp"]).agg(
                cpu_sum=("cpu", "sum"),
                cpu_count=("cpu", "count"),
                memory_sum=("memory", "sum"),
                job_type_sum=("job_type", "sum"),
            )
            for (bucket, timestamp), row in grouped.iterrows():
                key = (int(bucket), int(timestamp))
                count = int(row["cpu_count"])
                bucket_stats[key][0] += float(row["cpu_sum"])
                bucket_stats[key][1] += count
                memory_stats[key][0] += float(row["memory_sum"])
                memory_stats[key][1] += count
                type_stats[key][0] += float(row["job_type_sum"])
                type_stats[key][1] += count

    timestamps = sorted({timestamp for _, timestamp in bucket_stats.keys()})
    by_bucket = {}
    for bucket in range(num_buckets):
        rows = []
        for timestamp in timestamps:
            key = (bucket, timestamp)
            cpu_total, cpu_count = bucket_stats.get(key, (np.nan, 0))
            mem_total, mem_count = memory_stats.get(key, (np.nan, 0))
            type_total, type_count = type_stats.get(key, (np.nan, 0))
            if not cpu_count:
                rows.append((timestamp, np.nan, np.nan, np.nan))
                continue

            if bucket_agg == "mean":
                cpu = cpu_total / cpu_count
                memory = mem_total / max(mem_count, 1)
            else:
                cpu = cpu_total
                memory = mem_total
            job_type = type_total / max(type_count, 1)
            rows.append((timestamp, cpu, memory, job_type))

        frame = pd.DataFrame(rows, columns=["timestamp", "cpu", "memory", "job_type"])
        if frame["cpu"].isna().all():
            continue
        frame[["cpu", "memory", "job_type"]] = frame[["cpu", "memory", "job_type"]].interpolate(
            limit_direction="both"
        )
        frame[["cpu", "memory", "job_type"]] = frame[["cpu", "memory", "job_type"]].ffill().bfill()
        by_bucket[bucket] = frame

    return by_bucket


def build_series(bucket_frames, min_periods):
    series = []
    for _, frame in bucket_frames.items():
        if len(frame) < min_periods:
            continue
        values = []
        for timestamp, cpu, memory, job_type in frame.itertuples(index=False, name=None):
            day, hour, minute = timestamp_features(int(timestamp))
            values.append([float(cpu), day, hour, minute, float(memory), float(job_type)])
        series.append(np.asarray(values, dtype=np.float32))
    return series


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.download:
        download_file(GOOGLE_V1_URL, args.raw_path)
    if not os.path.exists(args.raw_path):
        raise FileNotFoundError(
            "{} not found. Download it from {} or rerun with --download true.".format(args.raw_path, GOOGLE_V1_URL)
        )

    bucket_frames = read_google_v1_as_buckets(
        args.raw_path,
        num_buckets=args.num_buckets,
        bucket_key=args.bucket_key,
        bucket_agg=args.bucket_agg,
    )
    min_periods = args.enc_len + args.pred_len + 1
    series = build_series(bucket_frames, min_periods=min_periods)
    if len(series) < args.min_series:
        raise RuntimeError(
            "Only built {} series; need at least {}. Increase --num_buckets or reduce --min_series.".format(
                len(series), args.min_series
            )
        )

    random.shuffle(series)
    if args.max_series:
        series = series[: args.max_series]

    split = int(len(series) * args.train_ratio)
    train = np.stack(series[:split]).astype(np.float32)
    test = np.stack(series[split:]).astype(np.float32)

    train_path = os.path.join(args.output_dir, args.train_name)
    test_path = os.path.join(args.output_dir, args.test_name)
    np.save(train_path, train)
    np.save(test_path, test)

    print("Saved train:", train_path, train.shape)
    print("Saved test:", test_path, test.shape)
    print("Feature layout: [cpu_target, day, hour, minute, memory, avg_job_type]")
    print("Train target min/max/mean:", train[:, :, 0].min(), train[:, :, 0].max(), train[:, :, 0].mean())
    print("Test target min/max/mean:", test[:, :, 0].min(), test[:, :, 0].max(), test[:, :, 0].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Google Cluster Trace v1 arrays for DynEformer/Informer.")
    parser.add_argument("--download", type=str2bool, default=True)
    parser.add_argument("--raw_path", type=str, default="external/google_cluster_data_raw/google-cluster-data-1.csv.gz")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--train_name", type=str, default="tr_google_v1.npy")
    parser.add_argument("--test_name", type=str, default="te_google_v1.npy")
    parser.add_argument("--num_buckets", type=int, default=1000)
    parser.add_argument("--bucket_key", choices=["job", "task"], default="job")
    parser.add_argument("--bucket_agg", choices=["sum", "mean"], default="sum")
    parser.add_argument("--max_series", type=int, default=800)
    parser.add_argument("--min_series", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--enc_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2)
    main(parser.parse_args())
