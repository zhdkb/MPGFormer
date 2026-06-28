# Dataset Preparation Guide

This document explains how to regenerate the processed workload files expected by the training scripts. All models in this repository use the same tensor convention:

```text
[num_series, num_periods, num_features]
```

The first feature channel is the target workload. The remaining channels are used as dynamic covariates or static/context features, depending on the dataset.

Unless otherwise specified, run the commands from the repository root:

```bash
cd /path/to/MPGFormer
conda activate MPGFormer
mkdir -p data external
```

Replace `/path/to/MPGFormer` with your local clone path.

## Processed Files

| Output file | Source | Preparation script |
| --- | --- | --- |
| `ECW_08.npy`, `ECW_switch.pkl`, `ECW_newmac.pkl`, `ECW_newapp.pkl`, `Task_ID_Dict.pkl` | ECW workload data used by DynEformer / One for All | Preprocessed files from the original ECW package |
| `tr_vm_official.npy`, `te_vm_official.npy` | Microsoft Azure Public Dataset VM CPU traces | `scripts/prepare_azure_vm_dataset.py` |
| `tr_google_v1.npy`, `te_google_v1.npy` | Google Cluster Trace v1 | `scripts/prepare_google_cluster_dataset.py` |
| `tr_materna_mt1.npy`, `va_materna_mt1.npy`, `te_materna_mt1.npy` | GWA-T-13 Materna trace 1 | `scripts/prepare_materna_workload_dataset.py` |
| `tr_materna_mt2.npy`, `va_materna_mt2.npy`, `te_materna_mt2.npy` | GWA-T-13 Materna trace 2 | `scripts/prepare_materna_workload_dataset.py` |
| `tr_materna_mt3.npy`, `va_materna_mt3.npy`, `te_materna_mt3.npy` | GWA-T-13 Materna trace 3 | `scripts/prepare_materna_workload_dataset.py` |

## ECW Workload Files

The ECW files are:

```text
data/ECW_08.npy
data/ECW_switch.pkl
data/ECW_newmac.pkl
data/ECW_newapp.pkl
data/Task_ID_Dict.pkl
```

These files correspond to the main ECW setting, App Switch, New Infras., New App, and the task-ID dictionary used by the ECW preprocessing pipeline. The raw ECW business logs and machine metadata are not redistributed in this repository. To reproduce ECW experiments, copy the preprocessed files from the original ECW/DynEformer data package:

```bash
mkdir -p data
cp /path/to/original_ecw/ECW_08.npy data/
cp /path/to/original_ecw/ECW_switch.pkl data/
cp /path/to/original_ecw/ECW_newmac.pkl data/
cp /path/to/original_ecw/ECW_newapp.pkl data/
cp /path/to/original_ecw/Task_ID_Dict.pkl data/
```

Expected shapes:

```text
ECW_08.npy       -> (797, 720, 16)
ECW_switch.pkl   -> (3, 120, 16)
ECW_newmac.pkl   -> (651, 144, 16)
ECW_newapp.pkl   -> (11, 720, 16)
```

## Microsoft Azure VM CPU

The Azure VM CPU traces come from the Microsoft Azure Public Dataset:

```bash
git clone https://github.com/Azure/AzurePublicDataset.git external/AzurePublicDataset
```

Generate the processed arrays:

```bash
python scripts/prepare_azure_vm_dataset.py \
  --version v1 \
  --num_files 1 \
  --max_vms_to_track 2000 \
  --max_segments 800 \
  --train_ratio 0.8
```

If the raw `.csv.gz` files have already been downloaded, place them under:

```text
external/azure_vm_cpu_readings/
```

Then run:

```bash
python scripts/prepare_azure_vm_dataset.py \
  --version v1 \
  --download false \
  --num_files 1 \
  --max_vms_to_track 2000 \
  --max_segments 800 \
  --train_ratio 0.8
```

Expected outputs:

```text
data/tr_vm_official.npy
data/te_vm_official.npy
```

## Google Cluster Trace v1

The Google Cluster Trace v1 data can be obtained from the official Google cluster-data repository:

```bash
git clone https://github.com/google/cluster-data.git external/google_cluster_data
```

Place the raw trace files under:

```text
external/google_cluster_data_raw/
```

Then generate the processed arrays:

```bash
python scripts/prepare_google_cluster_dataset.py \
  --raw_root external/google_cluster_data_raw \
  --max_series 1000 \
  --max_segments 1200 \
  --train_ratio 0.8
```

Expected outputs:

```text
data/tr_google_v1.npy
data/te_google_v1.npy
```

## GWA-T-13 Materna

The Materna traces are from the Grid Workloads Archive GWA-T-13 collection:

```text
https://atlarge-research.com/gwa-t-13/
```

Download and extract the raw traces into:

```text
data/GWA-T-13_Materna-Workload-Traces/
```

Generate MT-1, MT-2, and MT-3 separately:

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

## Shape Check

After preparing the files, verify their shapes:

```bash
python - <<'PY'
import os
import pickle
import numpy as np

for name in sorted(os.listdir("data")):
    if name.endswith(".npy"):
        arr = np.load(os.path.join("data", name), allow_pickle=True)
        print(name, arr.shape)
    elif name.endswith(".pkl"):
        with open(os.path.join("data", name), "rb") as f:
            arr = pickle.load(f)
        if hasattr(arr, "shape"):
            print(name, arr.shape)
PY
```

The training scripts expect the processed files to remain under `data/` with the filenames listed above.
