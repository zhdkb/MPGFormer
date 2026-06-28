import argparse
import gzip
import hashlib
import os
import random
import shutil
import urllib.request
from collections import Counter
from collections import defaultdict

import numpy as np
import pandas as pd


V1_LINKS = "external/AzurePublicDataset/AzurePublicDatasetV1Links.txt"
V2_LINKS = "external/AzurePublicDataset/AzurePublicDatasetLinksV2.txt"


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def get_cpu_links(version):
    link_file = V1_LINKS if version == "v1" else V2_LINKS
    with open(link_file, "r", encoding="utf-8-sig") as f:
        links = [line.strip() for line in f if line.strip()]
    return [link for link in links if "/vm_cpu_readings/" in link]


def download_file(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    tmp_path = out_path + ".tmp"
    print("Downloading:", url)
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp_path, out_path)


def timestamp_features(timestamp):
    day = timestamp // 86400
    hour = (timestamp % 86400) // 3600
    minute = (timestamp % 3600) // 60
    return day, hour, minute


def stable_bucket(value, num_buckets):
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


def read_cpu_files_as_buckets(paths, num_buckets, bucket_agg):
    names = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
    bucket_stats = defaultdict(lambda: [0.0, 0])

    for index, path in enumerate(paths, start=1):
        print("Parsing bucket file {}/{}: {}".format(index, len(paths), path))
        with gzip.open(path, "rt") as f:
            chunks = pd.read_csv(f, header=None, names=names, chunksize=1_000_000)
            for chunk in chunks:
                chunk = chunk[["timestamp", "vm_id", "avg_cpu"]].dropna()
                chunk["bucket"] = chunk["vm_id"].map(lambda vm_id: stable_bucket(vm_id, num_buckets))
                grouped = chunk.groupby(["bucket", "timestamp"])["avg_cpu"].agg(["sum", "count"])
                for (bucket, timestamp), row in grouped.iterrows():
                    key = (int(bucket), int(timestamp))
                    bucket_stats[key][0] += float(row["sum"])
                    bucket_stats[key][1] += int(row["count"])

    by_bucket = defaultdict(list)
    timestamps = sorted({timestamp for _, timestamp in bucket_stats.keys()})
    for bucket in range(num_buckets):
        values = {}
        counts = {}
        for timestamp in timestamps:
            total, count = bucket_stats.get((bucket, timestamp), (np.nan, 0))
            if count:
                values[timestamp] = total
                counts[timestamp] = count
        if not values:
            continue

        if bucket_agg == "sum":
            series = pd.Series(values, index=timestamps, dtype=float)
        else:
            series = pd.Series(
                {timestamp: values[timestamp] / counts[timestamp] for timestamp in values},
                index=timestamps,
                dtype=float,
            )
        series = series.interpolate(limit_direction="both").ffill().bfill()

        for timestamp, avg_cpu in series.items():
            day, hour, minute = timestamp_features(int(timestamp))
            by_bucket[bucket].append((int(timestamp), float(avg_cpu) * 10.0, day, hour, minute))

    return by_bucket


def read_cpu_file(path, series, selected_vms=None):
    names = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
    with gzip.open(path, "rt") as f:
        chunks = pd.read_csv(f, header=None, names=names, chunksize=1_000_000)
        for chunk in chunks:
            chunk = chunk[["timestamp", "vm_id", "avg_cpu"]].dropna()
            if selected_vms is not None:
                chunk = chunk[chunk["vm_id"].isin(selected_vms)]
            for timestamp, vm_id, avg_cpu in chunk.itertuples(index=False, name=None):
                day, hour, minute = timestamp_features(int(timestamp))
                workload = float(avg_cpu) * 10.0
                series[vm_id].append((int(timestamp), workload, day, hour, minute))


def select_top_vms(paths, limit):
    names = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
    counts = Counter()
    for path in paths:
        print("Counting VM readings:", path)
        with gzip.open(path, "rt") as f:
            chunks = pd.read_csv(
                f,
                header=None,
                names=names,
                usecols=["vm_id"],
                chunksize=1_000_000,
            )
            for chunk in chunks:
                counts.update(chunk["vm_id"].dropna().tolist())
    return {vm_id for vm_id, _ in counts.most_common(limit)}


def build_segments(series, seq_len, stride, max_segments_per_vm):
    segments = []
    for vm_id, rows in series.items():
        rows = sorted(rows, key=lambda row: row[0])
        if len(rows) < seq_len:
            continue

        by_ts = {row[0]: row for row in rows}
        timestamps = sorted(by_ts)
        current = []
        previous = None
        for timestamp in timestamps:
            if previous is None or timestamp - previous == 300:
                current.append(by_ts[timestamp])
            else:
                segments.extend(slice_run(current, seq_len, stride, max_segments_per_vm))
                current = [by_ts[timestamp]]
            previous = timestamp
        segments.extend(slice_run(current, seq_len, stride, max_segments_per_vm))

    return segments


def slice_run(rows, seq_len, stride, max_segments_per_vm):
    if len(rows) < seq_len:
        return []
    picked = []
    for start in range(0, len(rows) - seq_len + 1, stride):
        window = rows[start:start + seq_len]
        arr = np.asarray([[w, d, h, m] for _, w, d, h, m in window], dtype=np.float32)
        picked.append(arr)
        if max_segments_per_vm and len(picked) >= max_segments_per_vm:
            break
    return picked


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    cpu_links = get_cpu_links(args.version)[:args.num_files]
    if not cpu_links:
        raise RuntimeError("No VM CPU links found for {}".format(args.version))

    series = defaultdict(list)
    local_paths = []
    for index, url in enumerate(cpu_links, start=1):
        name = os.path.basename(url)
        local_path = os.path.join(args.download_dir, name)
        if args.download:
            download_file(url, local_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                "{} not found. Rerun with --download true or download it manually.".format(local_path)
            )
        local_paths.append(local_path)

    if args.mode == "bucket":
        series = read_cpu_files_as_buckets(local_paths, args.num_buckets, args.bucket_agg)
    else:
        selected_vms = None
        if args.max_vms_to_track is not None:
            selected_vms = select_top_vms(local_paths, args.max_vms_to_track)
            print("Selected {} VMs with the most readings.".format(len(selected_vms)))

        for index, local_path in enumerate(local_paths, start=1):
            print("Parsing file {}/{}: {}".format(index, len(cpu_links), local_path))
            read_cpu_file(local_path, series, selected_vms=selected_vms)

    segments = build_segments(
        series,
        seq_len=args.seq_len,
        stride=args.segment_stride,
        max_segments_per_vm=args.max_segments_per_vm,
    )
    if len(segments) < args.min_segments:
        raise RuntimeError(
            "Only built {} segments; need at least {}. Increase --num_files or reduce --min_segments.".format(
                len(segments), args.min_segments
            )
        )

    random.shuffle(segments)
    if args.max_segments:
        segments = segments[:args.max_segments]

    split = int(len(segments) * args.train_ratio)
    train = np.stack(segments[:split]).astype(np.float32)
    test = np.stack(segments[split:]).astype(np.float32)

    train_path = os.path.join(args.output_dir, args.train_name)
    test_path = os.path.join(args.output_dir, args.test_name)
    np.save(train_path, train)
    np.save(test_path, test)

    print("Saved train:", train_path, train.shape)
    print("Saved test:", test_path, test.shape)
    print("Target train min/max/mean:", train[:, :, 0].min(), train[:, :, 0].max(), train[:, :, 0].mean())
    print("Target test min/max/mean:", test[:, :, 0].min(), test[:, :, 0].max(), test[:, :, 0].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DynEformer Azure VM arrays from AzurePublicDataset.")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--mode", choices=["bucket", "vm"], default="bucket")
    parser.add_argument("--num_buckets", type=int, default=1000)
    parser.add_argument("--bucket_agg", choices=["mean", "sum"], default="mean")
    parser.add_argument("--download", type=str2bool, default=True)
    parser.add_argument("--download_dir", type=str, default="external/azure_vm_cpu_readings")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--train_name", type=str, default="tr_vm_official.npy")
    parser.add_argument("--test_name", type=str, default="te_vm_official.npy")
    parser.add_argument("--num_files", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=144)
    parser.add_argument("--segment_stride", type=int, default=144)
    parser.add_argument("--max_segments_per_vm", type=int, default=1)
    parser.add_argument("--max_segments", type=int, default=800)
    parser.add_argument("--min_segments", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--max_vms_to_track", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2)
    main(parser.parse_args())
