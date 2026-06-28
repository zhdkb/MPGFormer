# data 文件夹生成说明

这个文档说明如何重新生成 `data/` 文件夹下用于训练和测试的 workload 数据文件：需要从哪里下载原始数据、放到哪里，以及运行哪个脚本把原始数据处理成项目统一使用的 `.npy/.pkl` 格式。

项目里各模型默认读取三维数据：

```text
[num_series, num_periods, num_features]
```

其中第 `0` 维特征永远是预测目标，也就是 workload target；后面的维度是时间特征、动态协变量或静态/上下文特征。

下面所有命令默认从项目根目录执行：

```bash
cd /path/to/MPGFormer_GitHub
conda activate DynEformer
mkdir -p data external
```

如果是在本地机器执行，把第一行路径换成你的本地项目路径即可。

## 数据文件总览

| data 输出文件 | 原始数据来源 | 生成方式 |
| --- | --- | --- |
| `ECW_08.npy`, `ECW_switch.pkl`, `ECW_newmac.pkl`, `ECW_newapp.pkl`, `Task_ID_Dict.pkl` | DynEformer/One for All 论文里的 ECW 原始业务 workload | 项目里只保留预处理后的产物；公开原始日志和完整生成脚本不在本仓库中 |
| `tr_vm_official.npy`, `te_vm_official.npy` | Microsoft Azure Public Dataset 的 VM CPU traces | `scripts/prepare_azure_vm_dataset.py` |
| `tr_google_v1.npy`, `te_google_v1.npy` | Google Cluster Trace v1 | `scripts/prepare_google_cluster_dataset.py` |
| `tr_materna_mt1.npy`, `va_materna_mt1.npy`, `te_materna_mt1.npy` | GWA-T-13 Materna Trace 1 | `scripts/prepare_materna_workload_dataset.py` |
| `tr_materna_mt2.npy`, `va_materna_mt2.npy`, `te_materna_mt2.npy` | GWA-T-13 Materna Trace 2 | `scripts/prepare_materna_workload_dataset.py` |
| `tr_materna_mt3.npy`, `va_materna_mt3.npy`, `te_materna_mt3.npy` | GWA-T-13 Materna Trace 3 | `scripts/prepare_materna_workload_dataset.py` |
| `tr_vm.npy`, `te_vm.npy` | 旧版 Azure 预处理文件 | 旧代码兼容文件；现在建议使用 `tr_vm_official.npy` 和 `te_vm_official.npy` |

## 1. ECW / DynEformer 原论文数据

ECW 相关文件包括：

```text
data/ECW_08.npy
data/ECW_switch.pkl
data/ECW_newmac.pkl
data/ECW_newapp.pkl
data/Task_ID_Dict.pkl
```

它们分别对应：

- `ECW_08.npy`：论文主实验 ECW 训练/测试数据。
- `ECW_switch.pkl`：App Switch 场景。
- `ECW_newmac.pkl`：New Infras. 场景。
- `ECW_newapp.pkl`：New App 场景。
- `Task_ID_Dict.pkl`：原始 ECW 特征处理里用到的任务 ID 字典。

注意：这个仓库里有部分 ECW 处理工具，例如 `models/data_process_utils.py`，但是它依赖原始业务日志和机器属性文件，例如 `raw_data/merged_0801_0830_bd_t_feats_hour.pkl`、`raw_data/20220801-20220830_mac_attr.pkl` 以及若干内部 `ams_*.pkl` 文件。这些原始文件没有放在当前仓库里，也没有公开下载地址。

因此，ECW 这几个文件不能像 Azure/Google/Materna 一样通过公开脚本完整重建。要复现实验，需要从原 DynEformer/ECW 数据包中复制预处理产物：

```bash
mkdir -p data
cp /path/to/original_ecw/ECW_08.npy data/
cp /path/to/original_ecw/ECW_switch.pkl data/
cp /path/to/original_ecw/ECW_newmac.pkl data/
cp /path/to/original_ecw/ECW_newapp.pkl data/
cp /path/to/original_ecw/Task_ID_Dict.pkl data/
```

当前项目中的期望形状是：

```text
ECW_08.npy       -> (797, 720, 16)
ECW_switch.pkl   -> (3, 120, 16)
ECW_newmac.pkl   -> (651, 144, 16)
ECW_newapp.pkl   -> (11, 720, 16)
```

ECW 的特征约定大致是：

```text
[target_workload, hour, day, week, static/context features...]
```

## 2. Azure VM CPU 数据集

原始数据来源：Microsoft Azure Public Dataset。

先拉取官方仓库，它里面包含 Azure 数据文件的下载链接列表：

```bash
git clone https://github.com/Azure/AzurePublicDataset.git external/AzurePublicDataset
```

如果目录已经存在，用下面命令更新：

```bash
git -C external/AzurePublicDataset pull
```

处理脚本会读取：

```text
external/AzurePublicDataset/AzurePublicDatasetV1Links.txt
external/AzurePublicDataset/AzurePublicDatasetLinksV2.txt
```

然后从中筛选 `vm_cpu_readings` 的 `.csv.gz` 文件，下载到 `external/azure_vm_cpu_readings/`，再处理成 `data/` 下的 `.npy`。

生成当前项目推荐使用的 Azure official 数据：

```bash
python scripts/prepare_azure_vm_dataset.py \
  --version v1 \
  --mode bucket \
  --download true \
  --download_dir external/azure_vm_cpu_readings \
  --output_dir data \
  --train_name tr_vm_official.npy \
  --test_name te_vm_official.npy \
  --num_files 1 \
  --num_buckets 1000 \
  --bucket_agg mean \
  --seq_len 120 \
  --segment_stride 120 \
  --max_segments 800 \
  --train_ratio 0.8
```

如果远端服务器不能联网，先手动把 Azure CPU 原始文件放到：

```text
external/azure_vm_cpu_readings/
```

然后用 `--download false` 处理：

```bash
python scripts/prepare_azure_vm_dataset.py \
  --version v1 \
  --mode bucket \
  --download false \
  --download_dir external/azure_vm_cpu_readings \
  --output_dir data \
  --train_name tr_vm_official.npy \
  --test_name te_vm_official.npy \
  --num_files 1 \
  --num_buckets 1000 \
  --bucket_agg mean \
  --seq_len 120 \
  --segment_stride 120 \
  --max_segments 800 \
  --train_ratio 0.8
```

生成结果：

```text
data/tr_vm_official.npy -> (640, 120, 4)
data/te_vm_official.npy -> (160, 120, 4)
```

Azure 特征格式：

```text
[avg_cpu_target_times_10, day, hour, minute]
```

说明：`data/tr_vm.npy` 和 `data/te_vm.npy` 是旧版 Azure 兼容文件，旧脚本中可能还会引用它们；新实验和文档里统一建议使用 `tr_vm_official.npy`、`te_vm_official.npy`。

## 3. Google Cluster Trace v1

原始数据来源：Google Cluster Trace v1。

项目脚本默认使用这个原始文件：

```text
https://commondatastorage.googleapis.com/clusterdata-misc/google-cluster-data-1.csv.gz
```

也可以额外拉取 Google 官方说明仓库：

```bash
git clone https://github.com/google/cluster-data.git external/google_cluster_data
```

生成 `data/tr_google_v1.npy` 和 `data/te_google_v1.npy`：

```bash
python scripts/prepare_google_cluster_dataset.py \
  --download true \
  --raw_path external/google_cluster_data_raw/google-cluster-data-1.csv.gz \
  --output_dir data \
  --train_name tr_google_v1.npy \
  --test_name te_google_v1.npy \
  --num_buckets 1000 \
  --bucket_key job \
  --bucket_agg sum \
  --max_series 800 \
  --train_ratio 0.8
```

如果原始文件已经下载好了，就运行：

```bash
python scripts/prepare_google_cluster_dataset.py \
  --download false \
  --raw_path external/google_cluster_data_raw/google-cluster-data-1.csv.gz \
  --output_dir data \
  --train_name tr_google_v1.npy \
  --test_name te_google_v1.npy \
  --num_buckets 1000 \
  --bucket_key job \
  --bucket_agg sum \
  --max_series 800 \
  --train_ratio 0.8
```

生成结果：

```text
data/tr_google_v1.npy -> (640, 76, 6)
data/te_google_v1.npy -> (160, 76, 6)
```

Google v1 特征格式：

```text
[cpu_target, day, hour, minute, memory, avg_job_type]
```

## 4. GWA-T-13 Materna Workload Traces

原始数据来源：Grid Workloads Archive 的 GWA-T-13 Materna traces。

下载页面：

```text
https://atlarge-research.com/gwa-t-13/
```

直接下载链接：

```text
https://atlarge-research.com/gwa-traces/gwa_t_13_materna.zip
```

下载并解压：

```bash
mkdir -p /tmp/materna
curl -L https://atlarge-research.com/gwa-traces/gwa_t_13_materna.zip \
  -o /tmp/materna/gwa_t_13_materna.zip
unzip /tmp/materna/gwa_t_13_materna.zip -d data/
```

解压后应当得到这个目录结构：

```text
data/GWA-T-13_Materna-Workload-Traces/
  Materna-Trace-1/
  Materna-Trace-2/
  Materna-Trace-3/
```

然后运行处理脚本，把 MT-1、MT-2、MT-3 分别生成，不要聚合在一起：

```bash
python scripts/prepare_materna_workload_dataset.py \
  --raw_root data/GWA-T-13_Materna-Workload-Traces \
  --output_dir data \
  --trace all \
  --max_missing_rate 0.02 \
  --match_fremer_counts true \
  --train_ratio 0.7 \
  --val_ratio 0.1
```

生成结果：

```text
data/tr_materna_mt1.npy -> (413, 5846, 10)
data/va_materna_mt1.npy -> (413, 835, 10)
data/te_materna_mt1.npy -> (413, 1671, 10)

data/tr_materna_mt2.npy -> (402, 6249, 10)
data/va_materna_mt2.npy -> (402, 893, 10)
data/te_materna_mt2.npy -> (402, 1786, 10)

data/tr_materna_mt3.npy -> (371, 7257, 10)
data/va_materna_mt3.npy -> (371, 1037, 10)
data/te_materna_mt3.npy -> (371, 2074, 10)
```

Materna 特征格式：

```text
[
  cpu_percent_target,
  day,
  hour,
  minute,
  cpu_mhz,
  memory_mb,
  memory_percent,
  disk_read_kbps,
  disk_write_kbps,
  network_total_kbps
]
```

这个处理方式和 Fremer 论文里的 Materna 设置保持一致：过滤缺失率超过 `2%` 的 VM，保留论文对应的 VM 数量，然后按照时间维度做 `7:1:2` 的 train/validation/test 划分。

## 5. 生成后检查

处理完成后，可以用下面命令检查 `data/` 下各文件形状：

```bash
python - <<'PY'
import os
import pickle
import numpy as np

for name in sorted(os.listdir("data")):
    if not name.endswith((".npy", ".pkl")):
        continue
    path = os.path.join("data", name)
    if name.endswith(".pkl"):
        with open(path, "rb") as f:
            value = pickle.load(f)
        arr = np.asarray(value)
    else:
        arr = np.load(path, allow_pickle=True)
    print(f"{name:24s} shape={arr.shape} dtype={arr.dtype}")
PY
```

核心文件应包含：

```text
ECW_08.npy
ECW_switch.pkl
ECW_newmac.pkl
ECW_newapp.pkl
Task_ID_Dict.pkl
tr_vm_official.npy
te_vm_official.npy
tr_google_v1.npy
te_google_v1.npy
tr_materna_mt1.npy
va_materna_mt1.npy
te_materna_mt1.npy
tr_materna_mt2.npy
va_materna_mt2.npy
te_materna_mt2.npy
tr_materna_mt3.npy
va_materna_mt3.npy
te_materna_mt3.npy
```
