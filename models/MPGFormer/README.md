# MPGFormer

MPGFormer is implemented in code as `MPGFormer`. It is a workload forecasting model for dynamic cloud systems and combines:

- multi-scale time-frequency prototype routing,
- non-stationary temporal attention,
- short-term CNN/GRU encoding,
- dynamic graph contrastive learning,
- point and quantile forecasting.

An online conformal calibration utility is available separately from the default training entry.

## Input

The main model expects:

```python
x: [batch, enc_len + pred_len, input_size]
static_context: [batch, static_dim] or None
```

The first feature is treated as the target workload by default. Future target values inside `x` are ignored and replaced with prototype-based synchronous padding. Future known covariates may be placed in the remaining feature columns.

## Output

`MPGFormer.forward(..., return_dict=True)` returns `MPGFormerOutput`:

- `point`: point forecast `[batch, pred_len]`
- `quantiles`: ordered quantile forecast `[batch, pred_len, 3]`
- `lower`, `upper`: default interval bounds from quantiles
- `router_weights`: weights over temporal, graph, and prototype branches
- `drift_score`: estimated drift intensity
- `contrastive_loss`: graph-view contrastive regularizer
- `prototype_weights`: multi-scale prototype attention

## Training

From `models/MPGFormer`:

```bash
PYTHONPATH=../.. python train_mpgformer.py \
  --data ../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24
```

For the paper's sensitivity analysis, vary `--e_layers`, `--n_heads`,
`--num_prototypes`, or `--graph_topk` while keeping the dataset preset and
validation-selected checkpoint protocol fixed.
