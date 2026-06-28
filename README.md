# MPG-Former

Official implementation of **MPG-Former: Drift-Aware Multi-scale Prototype-Graph Transformer for Cloud Workload Forecasting**.

MPG-Former formulates cloud workload forecasting as a drift-aware evidence selection problem. Given a historical workload window, the model extracts temporal evidence, multi-scale time-frequency prototype evidence, and dynamic graph evidence, then adaptively routes these evidence sources for multi-step workload prediction.

## Highlights

- **Non-stationary temporal evidence modeling** with window-statistics-modulated attention.
- **Multi-scale time-frequency prototype retrieval** for reusable workload shape patterns.
- **Batch-wise dynamic graph evidence** for cross-workload co-fluctuation modeling.
- **Drift-aware evidence routing** that combines temporal, prototype, and graph contexts.
- **Point and interval forecasting** with MSE, MAE, and SMAPE evaluation.

## Repository Structure

```text
.
├── models/
│   ├── MPGFormer/          # Proposed model and training entry
│   └── baseline/           # Adapted baseline implementations
├── scripts/                # Dataset preparation and figure scripts
├── data/                   # Put processed datasets here
├── results/                # CSV files used for reported analysis
├── figures/                # Paper figures and generated analysis figures
├── docs/                   # Paper draft and architecture notes
├── DATA_GENERATION_GUIDE.md
├── REPRODUCE_SIX_MODELS_COMMANDS.md
└── requirements.txt
```

Large raw datasets and checkpoints are not included. Please follow `DATA_GENERATION_GUIDE.md` and the dataset links below to prepare data.

## Installation

```bash
conda create -n mpgformer python=3.9 -y
conda activate mpgformer
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build following the official PyTorch instructions if your environment requires a specific CUDA version.

## Data Format

All processed workload tensors should be stored under `data/` with shape:

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
data/te_materna_mt1.npy
data/tr_materna_mt2.npy
data/te_materna_mt2.npy
data/tr_materna_mt3.npy
data/te_materna_mt3.npy
```

See `data/README.md` and `DATA_GENERATION_GUIDE.md` for dataset sources and preparation commands.

## Train MPG-Former

Run from `models/MPGFormer`:

```bash
cd models/MPGFormer
python train_mpgformer.py \
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
python train_mpgformer.py \
  --data ../../data/tr_google_v1.npy \
  --test_data ../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24
```

More commands for MPG-Former and baselines are summarized in `REPRODUCE_SIX_MODELS_COMMANDS.md`.

## Baselines

The `models/baseline` directory contains adapted implementations for the comparison models used in the paper, including Autoformer, Informer, Fremer, iTransformer, DynEformer, PDF, TFEGRU, SOFTS, and MICN. These implementations use the same processed data format and evaluation scale whenever possible.

## Results and Figures

The `results/` directory contains CSV files for sensitivity, ablation, and interpretability analyses. The `figures/` directory contains generated PDF figures that can be directly used in LaTeX.

```bash
python scripts/draw_mpgformer_sensitivity.py
python scripts/draw_mpgformer_interpretability.py
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{mpgformer2026,
  title   = {MPG-Former: Drift-Aware Multi-scale Prototype-Graph Transformer for Cloud Workload Forecasting},
  author  = {Anonymous Authors},
  journal = {Under Review},
  year    = {2026}
}
```

## License

This project is released under the MIT License. Please check the licenses of the original baseline repositories and public datasets before redistribution.
