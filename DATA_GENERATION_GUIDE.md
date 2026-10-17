# Dataset Preparation Guide

This guide explains how to regenerate the processed workload files expected by MPGFormer and the adapted baselines.

All models use the same tensor convention:

```text
[num_series, num_periods, num_features]
```

The first feature channel is the forecasting target. The remaining channels are dynamic covariates or optional static/context features, depending on the dataset.

Run all commands from the repository root unless otherwise specified:

```bash
cd /path/to/MPGFormer
conda activate MPGFormer
mkdir -p data external
```

## Expected Processed Files

| Dataset | Required processed files | Source/preparation |
| --- | --- | --- |
| ECW | `data/ECW_08.npy`, `data/ECW_newmac.pkl`, `data/ECW_newapp.pkl` | Preprocessed files from `hsy23/ECWDataset` |
| Microsoft Azure | `data/tr_vm_official.npy`, `data/te_vm_official.npy` | `scripts/prepare_azure_vm_dataset.py` |
| Google v1 | `data/tr_google_v1.npy`, `data/te_google_v1.npy` | `scripts/prepare_google_cluster_dataset.py` |
| Materna MT-1 | `data/tr_materna_mt1.npy`, `data/va_materna_mt1.npy`, `data/te_materna_mt1.npy` | `scripts/prepare_materna_workload_dataset.py` |
| Materna MT-2 | `data/tr_materna_mt2.npy`, `data/va_materna_mt2.npy`, `data/te_materna_mt2.npy` | `scripts/prepare_materna_workload_dataset.py` |
| Materna MT-3 | `data/tr_materna_mt3.npy`, `data/va_materna_mt3.npy`, `data/te_materna_mt3.npy` | `scripts/prepare_materna_workload_dataset.py` |

## ECW Workload Files

The ECW files can be downloaded from the public ECW dataset repository:

```bash
git clone https://github.com/hsy23/ECWDataset.git external/ECWDataset
```

Copy the required processed files into `data/`:

```bash
cp external/ECWDataset/ECW_08.npy data/
cp external/ECWDataset/ECW_newmac.pkl data/
cp external/ECWDataset/ECW_newapp.pkl data/
```

Expected shapes:

```text
ECW_08.npy       -> (797, 720, 16)
ECW_newmac.pkl   -> (651, 144, 16)
ECW_newapp.pkl   -> (11, 720, 16)
```

## Microsoft Azure VM CPU

The Azure VM CPU traces are from the Microsoft Azure Public Dataset:

```bash
git clone https://github.com/Azure/AzurePublicDataset.git external/AzurePublicDataset
```

The preparation script reads the official link file from `external/AzurePublicDataset/` and downloads VM CPU `.csv.gz` files to `external/azure_vm_cpu_readings/`.

The script splits by VM or bucket identity before saving the fixed train/test arrays, so multiple segments from one source stay in one partition. Missing measurements are filled from preceding observations only.

Automatic download and preprocessing:

```bash
python scripts/prepare_azure_vm_dataset.py \
  --version v1 \
  --download true \
  --num_files 1 \
  --max_segments 800 \
  --train_ratio 0.8 \
  --seed 2
```

Offline preprocessing, if the `.csv.gz` files have already been placed in `external/azure_vm_cpu_readings/`:

```bash
python scripts/prepare_azure_vm_dataset.py \
  --version v1 \
  --download false \
  --num_files 1 \
  --max_segments 800 \
  --train_ratio 0.8 \
  --seed 2
```

Expected outputs:

```text
data/tr_vm_official.npy
data/te_vm_official.npy
```

The generated Azure tensors use the feature layout `[workload, day, hour, minute]`.

## Google Cluster Trace v1

The Google Cluster Trace v1 preparation script uses the URL assigned to `GOOGLE_V1_URL` in `scripts/prepare_google_cluster_dataset.py`.

Automatic download and preprocessing:

```bash
python scripts/prepare_google_cluster_dataset.py \
  --download true \
  --raw_path external/google_cluster_data_raw/google-cluster-data-1.csv.gz \
  --max_series 800 \
  --train_ratio 0.8 \
  --seed 2
```

Offline preprocessing, if the raw file has already been downloaded:

```bash
python scripts/prepare_google_cluster_dataset.py \
  --download false \
  --raw_path external/google_cluster_data_raw/google-cluster-data-1.csv.gz \
  --max_series 800 \
  --train_ratio 0.8 \
  --seed 2
```

Expected outputs:

```text
data/tr_google_v1.npy
data/te_google_v1.npy
```

The generated Google tensors use the feature layout `[cpu_target, day, hour, minute, memory, avg_job_type]`.

## GWA-T-13 Materna

The Materna traces are from the Grid Workloads Archive GWA-T-13 collection:

```text
https://gwa.ewi.tudelft.nl/datasets/gwa-t-13-materna/
```

Download and extract the raw trace directories into:

```text
data/GWA-T-13_Materna-Workload-Traces/
```

Then generate MT-1, MT-2, and MT-3 separately:

```bash
python scripts/prepare_materna_workload_dataset.py \
  --raw_root data/GWA-T-13_Materna-Workload-Traces \
  --trace all \
  --match_fremer_counts true
```

Expected outputs:

```text
data/tr_materna_mt1.npy
data/va_materna_mt1.npy
data/te_materna_mt1.npy
data/tr_materna_mt2.npy
data/va_materna_mt2.npy
data/te_materna_mt2.npy
data/tr_materna_mt3.npy
data/va_materna_mt3.npy
data/te_materna_mt3.npy
```

The Materna preprocessing follows the Fremer-style setup: traces with more than 2 percent missing values are filtered, each trace is kept as a separate dataset, and the time axis is split into train/validation/test partitions.

## Verify Shapes

After preparing the files, verify their shapes:

```bash
python - <<'PY'
import os
import pickle
import numpy as np

for name in sorted(os.listdir("data")):
    path = os.path.join("data", name)
    if name.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        print(name, arr.shape)
    elif name.endswith(".pkl"):
        with open(path, "rb") as f:
            arr = pickle.load(f)
        if hasattr(arr, "shape"):
            print(name, arr.shape)
PY
```

Every printed dataset used for training should be three-dimensional and follow `[num_series, num_periods, num_features]`.
