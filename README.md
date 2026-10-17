# MPGFormer

Implementation of **MPGFormer: Drift-Aware Multi-scale Prototype-Graph Transformer for Cloud Workload Forecasting**.

MPGFormer formulates cloud workload forecasting as a drift-aware evidence selection problem. Given a historical workload window, the model extracts temporal evidence, multi-scale time-frequency prototype evidence, and dynamic graph evidence, then adaptively routes these evidence sources for multi-step workload prediction.

## Highlights

- **Non-stationary temporal evidence modeling** with attention scaled by pre-RevIN target-history mean, standard deviation, least-squares slope, and mean absolute first difference.
- **Multi-scale time-frequency prototype retrieval** for reusable workload shape patterns.
- **Batch-wise dynamic graph evidence** for cross-workload co-fluctuation modeling.
- **Drift-aware evidence routing** that combines temporal, prototype, and graph contexts.
- **Point and quantile forecasts** with point-error evaluation using MSE, MAE, and SMAPE.

## Repository Structure

```text
.
models/                       Model implementations and training entries
scripts/                      Dataset preparation and result summary
data/                         Processed dataset location
DATA_GENERATION_GUIDE.md      Dataset preparation instructions
REPRODUCE_ALL_MODELS_COMMANDS.md  Training commands
requirements.txt              Python dependencies
```

Large raw datasets and checkpoints are not included. Prepare processed tensors with [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md), and keep them under [data/](data/).

## Installation

```bash
conda create -n MPGFormer python=3.9 -y
conda activate MPGFormer
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build following the official PyTorch instructions if your environment requires a specific CUDA version.

## Data Format

All processed workload tensors should be stored under [data/](data/) with shape:

```text
[num_series, num_periods, num_features]
```

The first feature is treated as the target workload by default. Remaining features are used as dynamic covariates or static/context features depending on the dataset.

Typical filenames used by the scripts:

```text
data/ECW_08.npy
data/ECW_newmac.pkl
data/ECW_newapp.pkl
data/tr_vm_official.npy
data/te_vm_official.npy
data/tr_google_v1.npy
data/te_google_v1.npy
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

See [data/README.md](data/README.md) for the expected file list and [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md) for dataset sources and preparation commands.

## Train MPGFormer

Run from [models/MPGFormer](models/MPGFormer):

```bash
cd models/MPGFormer
PYTHONPATH=../.. python train_mpgformer.py \
  --data ../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24
```

For datasets with separate train/test files:

```bash
cd models/MPGFormer
PYTHONPATH=../.. python train_mpgformer.py \
  --data ../../data/tr_google_v1.npy \
  --test_data ../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --input_size 6 \
  --static_start 6
```

Commands for MPGFormer and the included baselines are summarized in [REPRODUCE_ALL_MODELS_COMMANDS.md](REPRODUCE_ALL_MODELS_COMMANDS.md).

## Baselines

The [models/baseline](models/baseline) directory contains local code and workload training entries for all 16 baselines in the paper's main and appendix comparisons: Autoformer, Informer, Fremer, iTransformer, DynEformer, PDF, TFEGRU, SOFTS, MICN, PatchTST, Non-stationary Transformer, Pathformer, TimeMixer, MSGNet, CrossGNN, and TimesNet. Each model has a command in [REPRODUCE_ALL_MODELS_COMMANDS.md](REPRODUCE_ALL_MODELS_COMMANDS.md). The imported architecture components are attributed in their model directories.

## Evaluation protocol

Training data alone fits the scalers. Validation MSE selects the checkpoint; the held-out test set is evaluated only once after training. Use the same 11 seed IDs, 0 through 10, for every model and report the mean and sample SD of final test metrics. The [batch runner](scripts/run_baseline_11_seeds.py) runs MPGFormer and the baselines that do not require a prebuilt Global Pool; [REPRODUCE_ALL_MODELS_COMMANDS.md](REPRODUCE_ALL_MODELS_COMMANDS.md) gives commands for every model and the eight dataset presets.

Future decoder inputs contain calendar features only; future measured resource covariates are hidden. Evaluation includes the final complete forecasting window and weights the final partial batch by its sample count. Previously generated datasets and metrics should be regenerated with the current preparation and evaluation code.

Azure target values are divided by 10 before fitting the training scaler. Inverse scaling removes the fitted scaler but retains this fixed unit conversion, so reported Azure MAE and MSE use target/10 units.

## License

The repository's own code is released under the [MIT License](LICENSE). Imported architecture files retain their source notices and licenses where supplied; see the source notes in their model directories. Public datasets have separate terms.
