# Reproduce All Models

This file keeps the reproduction commands compact. Use it in two steps:

1. Select one dataset preset.
2. Run the command block for MPGFormer or a baseline.

All commands assume the repository is cloned as `/path/to/MPGFormer`.

```bash
export REPO=/path/to/MPGFormer
cd "$REPO"
conda activate MPGFormer
export SEED=0  # repeat each training command with SEED=0,1,...,10
```

## 1. Dataset Presets

Choose one preset before running a model command.

### ECW

```bash
export TRAIN="$REPO/data/ECW_08.npy"
unset TEST VAL
export INPUT_SIZE=4
export STATIC_START=4
export EPOCHS=300
export BATCH=64
export TEST_STEP=12
```

### New Infras.

```bash
export TRAIN="$REPO/data/ECW_08.npy"
export TEST="$REPO/data/ECW_newmac.pkl"
unset VAL
export INPUT_SIZE=4
export STATIC_START=4
export EPOCHS=300
export BATCH=32
export TEST_STEP=2
```

### New App

```bash
export TRAIN="$REPO/data/ECW_08.npy"
export TEST="$REPO/data/ECW_newapp.pkl"
unset VAL
export INPUT_SIZE=4
export STATIC_START=4
export EPOCHS=300
export BATCH=64
export TEST_STEP=2
```

### Microsoft Azure

```bash
export TRAIN="$REPO/data/tr_vm_official.npy"
export TEST="$REPO/data/te_vm_official.npy"
unset VAL
export INPUT_SIZE=4
export STATIC_START=4
export EPOCHS=300
export BATCH=64
export TEST_STEP=12
```

### Google v1

```bash
export TRAIN="$REPO/data/tr_google_v1.npy"
export TEST="$REPO/data/te_google_v1.npy"
unset VAL
export INPUT_SIZE=6
export STATIC_START=6
export EPOCHS=300
export BATCH=64
export TEST_STEP=12
```

### Materna MT-1

```bash
export TRAIN="$REPO/data/tr_materna_mt1.npy"
export TEST="$REPO/data/te_materna_mt1.npy"
export VAL="$REPO/data/va_materna_mt1.npy"
export INPUT_SIZE=10
export STATIC_START=10
export EPOCHS=30
export BATCH=64
export TEST_STEP=12
```

### Materna MT-2

```bash
export TRAIN="$REPO/data/tr_materna_mt2.npy"
export TEST="$REPO/data/te_materna_mt2.npy"
export VAL="$REPO/data/va_materna_mt2.npy"
export INPUT_SIZE=10
export STATIC_START=10
export EPOCHS=30
export BATCH=64
export TEST_STEP=12
```

### Materna MT-3

```bash
export TRAIN="$REPO/data/tr_materna_mt3.npy"
export TEST="$REPO/data/te_materna_mt3.npy"
export VAL="$REPO/data/va_materna_mt3.npy"
export INPUT_SIZE=10
export STATIC_START=10
export EPOCHS=30
export BATCH=64
export TEST_STEP=12
```

Create the optional test-data argument used by most scripts:

```bash
test_arg=()
if [ -n "${TEST:-}" ]; then
  test_arg=(--test_data "$TEST")
fi
val_arg=()
if [ -n "${VAL:-}" ]; then
  val_arg=(--val_data "$VAL")
fi
```

## 2. MPGFormer

```bash
cd "$REPO/models/MPGFormer"
PYTHONPATH="$REPO" python train_mpgformer.py \
  --data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --static_start "$STATIC_START" \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --seed "$SEED"
```

## 3. Baselines Except DynEformer

Run one of the following blocks after selecting a dataset preset.

### Autoformer

```bash
cd "$REPO/models/baseline/auto_informer"
PYTHONPATH="$REPO" python auto_info_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --train_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### Informer

```bash
cd "$REPO/models/baseline/Informer"
PYTHONPATH="$REPO" python informer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --train_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### Fremer

```bash
cd "$REPO/models/baseline/Fremer"
PYTHONPATH="$REPO" python fremer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### iTransformer

```bash
cd "$REPO/models/baseline/iTransformer"
PYTHONPATH="$REPO" python itransformer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --train_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### PDF

```bash
cd "$REPO/models/baseline/PDF"
PYTHONPATH="$REPO" python pdf_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### TFEGRU

```bash
cd "$REPO/models/baseline/TFEGRU"
PYTHONPATH="$REPO" python tfegru_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### SOFTS

```bash
cd "$REPO/models/baseline/SOFTS"
PYTHONPATH="$REPO" python softs_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### MICN

```bash
cd "$REPO/models/baseline/MICN"
PYTHONPATH="$REPO" python micn_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

## 4. DynEformer

DynEformer requires a dataset-specific Global Pool before training. First search the number of clusters `K`, then build the pool, and finally run the corresponding DynEformer script.

For ECW, New Infras., and New App, the original paper path uses `K=551`.

### ECW, New Infras., and New App

```bash
cd "$REPO/models/baseline/GlobalPooling/vade_pooling"
PYTHONPATH="$REPO" python global_pooling_main.py \
  --dataX_dir "$REPO/data/ECW_08.npy" \
  --nClusters 551 \
  --series_len 48 \
  --step 12

cd "$REPO/models/baseline/DynEformer"
PYTHONPATH="$REPO" python dyneformer_main.py --num_epoches 300 --batch_size 256 --seed "$SEED"
PYTHONPATH="$REPO" python dynet_newmac.py --batch_size 32
PYTHONPATH="$REPO" python dynet_newadded.py
```

### Azure, Google v1, and Materna

Search `K` on the selected training file:

```bash
cd "$REPO/models/baseline/GlobalPooling/vade_search_cluster"
PYTHONPATH="$REPO" python search_azure_k.py \
  --dataX_dir "$TRAIN" \
  --candidates 10,20,30,40,50,60,80,100,120,150,200,250,300,400,500 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the pool with the printed `BEST_K_BY_BIC`:

```bash
export BEST_K=<BEST_K_BY_BIC>

cd "$REPO/models/baseline/GlobalPooling/vade_pooling"
PYTHONPATH="$REPO" python global_pooling_main.py \
  --dataX_dir "$TRAIN" \
  --nClusters "$BEST_K" \
  --series_len 48 \
  --step 12 \
  --gmm_reg_covar 1e-4 \
  --gmm_n_init 3
```

Run DynEformer:

```bash
cd "$REPO/models/baseline/DynEformer"

# Use dyneformer_azure_main.py for Microsoft Azure.
PYTHONPATH="$REPO" python dyneformer_azure_main.py \
  --train_data "$TRAIN" \
  --test_data "$TEST" \
  "${val_arg[@]}" \
  --pool_clusters "$BEST_K" \
  --num_epoches "$EPOCHS" \
  --batch_size "$BATCH" \
  --seed "$SEED"

# Use dyneformer_google_main.py for Google v1 and Materna.
PYTHONPATH="$REPO" python dyneformer_google_main.py \
  --train_data "$TRAIN" \
  --test_data "$TEST" \
  "${val_arg[@]}" \
  --pool_clusters "$BEST_K" \
  --num_epoches "$EPOCHS" \
  --batch_size "$BATCH" \
  --seed "$SEED"
```

Run only the command that matches the selected dataset.

## 5. Notes

- Prepare datasets first with [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md).
- All scripts reserve validation data before fitting scalers. Validation MSE selects the best checkpoint; the test set is evaluated once afterward.
- ECW uses chronological 64/16/20 train/validation/test; New Infras. and New App use an 80/20 source-trace train/validation split plus a separate test trace. Azure and Google reserve 20% of training series IDs (fixed split seed 0). Materna uses its separate 70/10/20 train/validation/test files.
- For the paper protocol, run every model and dataset with matched `--seed` values 0 through 10, capture each final test metric, and report mean ± sample SD. Compute paired differences from matching seed IDs. Do not select epochs or hyperparameters with test metrics.
- Each run emits a `TEST_METRICS` JSON line. Save logs for all 11 seeds and summarize them with `python "$REPO/scripts/summarize_seed_runs.py" run.log --baseline <preselected_model_name>`. The script requires exactly seeds 0 through 10, reports sample SD, and computes paired 95% t intervals. Choose the comparator before inspecting test results.
- All scripts report MSE, MAE, and SMAPE on the inverse-scaled workload target.
- Azure target scaling follows the same convention across MPGFormer and baselines.
- Use smaller `EPOCHS` for quick debugging, but keep the listed values for paper reproduction.
