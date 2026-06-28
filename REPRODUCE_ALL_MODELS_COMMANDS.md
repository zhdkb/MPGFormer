# Reproduce All Models

This file keeps the reproduction commands compact. Use it in two steps:

1. Select one dataset preset.
2. Run the command block for MPG-Former or a baseline.

All commands assume the repository is cloned as `/path/to/MPGFormer`.

```bash
export REPO=/path/to/MPGFormer
cd "$REPO"
conda activate MPGFormer
```

## 1. Dataset Presets

Choose one preset before running a model command.

### ECW

```bash
export TRAIN="$REPO/data/ECW_08.npy"
unset TEST
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
```

## 2. MPG-Former

```bash
cd "$REPO/models/MPGFormer"
PYTHONPATH="$REPO" python train_mpgformer.py \
  --data "$TRAIN" \
  "${test_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --static_start "$STATIC_START" \
  --enc_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --select_metric mse
```

## 3. Baselines Except DynEformer

Run one of the following blocks after selecting a dataset preset.

### Autoformer

```bash
cd "$REPO/models/baseline/auto_informer"
PYTHONPATH="$REPO" python auto_info_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --train_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --test_step "$TEST_STEP"
```

### Informer

```bash
cd "$REPO/models/baseline/Informer"
PYTHONPATH="$REPO" python informer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --train_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --test_step "$TEST_STEP"
```

### Fremer

```bash
cd "$REPO/models/baseline/Fremer"
PYTHONPATH="$REPO" python fremer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP"
```

### iTransformer

```bash
cd "$REPO/models/baseline/iTransformer"
PYTHONPATH="$REPO" python itransformer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --train_epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP"
```

### PDF

```bash
cd "$REPO/models/baseline/PDF"
PYTHONPATH="$REPO" python pdf_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP"
```

### TFEGRU

```bash
cd "$REPO/models/baseline/TFEGRU"
PYTHONPATH="$REPO" python tfegru_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP"
```

### SOFTS

```bash
cd "$REPO/models/baseline/SOFTS"
PYTHONPATH="$REPO" python softs_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP"
```

### MICN

```bash
cd "$REPO/models/baseline/MICN"
PYTHONPATH="$REPO" python micn_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --test_step "$TEST_STEP"
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
PYTHONPATH="$REPO" python dyneformer_main.py --num_epoches 300 --batch_size 256
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
  --pool_clusters "$BEST_K" \
  --num_epoches "$EPOCHS" \
  --batch_size "$BATCH"

# Use dyneformer_google_main.py for Google v1 and Materna.
PYTHONPATH="$REPO" python dyneformer_google_main.py \
  --train_data "$TRAIN" \
  --test_data "$TEST" \
  --pool_clusters "$BEST_K" \
  --num_epoches "$EPOCHS" \
  --batch_size "$BATCH"
```

Run only the command that matches the selected dataset.

## 5. Notes

- Prepare datasets first with [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md).
- All scripts report MSE, MAE, and SMAPE on the inverse-scaled workload target.
- Azure target scaling follows the same convention across MPG-Former and baselines.
- Use smaller `EPOCHS` for quick debugging, but keep the listed values for paper reproduction.
