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

For component ablations, pass `--disable` followed by one or more component
names. The names are `revin`, `temperature`, `short_encoder`, `prototype`,
`graph`, `router`, `workload_graph`, `context_graph`, `prototype_entropy`, and
`drift_bias`. For example, `--disable workload_graph` removes only the workload
graph view, while `--disable workload_graph context_graph` removes both views.
`temperature` sets the attention temperature to one, matching the paper's
standard-attention ablation; `router` averages the active experts equally.
Loss ablations set `--quantile_weight 0`, `--peak_weight 0`, or both
`--tail_mse_weight 0 --target_tail_mse_weight 0`. Set
`--contrastive_weight 0` to remove graph-view alignment while retaining both
graph views.

From the repository root, use the matched-seed runner for each paper variant:

```bash
python scripts/run_baseline_11_seeds.py --dataset GoogleV1 \
  --models MPGFormer --mpgformer_variant no_both_views
```

`--mpgformer_variant` lists all broad, fine, and loss variants. By default the
runner uses seeds 0 through 10 and writes each variant into a separate log
directory under the dataset. The processed dataset files must be in `data/`.
