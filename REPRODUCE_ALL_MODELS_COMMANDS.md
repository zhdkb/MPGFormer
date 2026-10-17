# Reproduce All 17 Models

This file covers MPGFormer and every baseline in the paper's Table 10. Use the commands in two steps:

1. Select one dataset preset.
2. Run the command block for MPGFormer or a baseline.

All commands assume the repository is cloned as `/path/to/MPGFormer`.

```bash
export REPO=/path/to/MPGFormer
cd "$REPO"
conda activate MPGFormer
pip install -r requirements.txt
export SEED=0  # repeat each training command with matched seed IDs 0,...,10
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
  --test_step "$TEST_STEP" \
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

### PatchTST

```bash
cd "$REPO/models/baseline/PatchTST"
PYTHONPATH="$REPO" python patchtst_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### Non-stationary Transformer

```bash
cd "$REPO/models/baseline/NS_Transformer"
PYTHONPATH="$REPO" python ns_transformer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### Pathformer

```bash
cd "$REPO/models/baseline/Pathformer"
PYTHONPATH="$REPO" python pathformer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### TimeMixer

```bash
cd "$REPO/models/baseline/TimeMixer"
PYTHONPATH="$REPO" python timemixer_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### MSGNet

```bash
cd "$REPO/models/baseline/MSGNet"
PYTHONPATH="$REPO" python msgnet_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### CrossGNN

```bash
cd "$REPO/models/baseline/CrossGNN"
PYTHONPATH="$REPO" python crossgnn_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### TimesNet

```bash
cd "$REPO/models/baseline/TimesNet"
PYTHONPATH="$REPO" python timesnet_workload_main.py \
  --train_data "$TRAIN" \
  "${test_arg[@]}" \
  "${val_arg[@]}" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --input_size "$INPUT_SIZE" \
  --seq_len 48 \
  --label_len 12 \
  --pred_len 24 \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

## 4. DynEformer

DynEformer requires a Global Pool before training. The ECW commands use a fixed cluster count; the Azure, Google, and Materna commands search for `K` on training data before building a pool.

For ECW, New Infras., and New App, use `K=551`. Build a pool for each scenario's source training interval before training that scenario.

### ECW

```bash
cd "$REPO/models/baseline/GlobalPooling/vade_pooling"
PYTHONPATH="$REPO" python global_pooling_main.py \
  --dataX_dir "$REPO/data/ECW_08.npy" \
  --nClusters 551 \
  --series_len 48 \
  --step 12 \
  --source_train_ratio 0.64

cd "$REPO/models/baseline/DynEformer"
PYTHONPATH="$REPO" python dyneformer_main.py \
  --data "$TRAIN" \
  --num_epoches "$EPOCHS" \
  --batch_size "$BATCH" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
```

### New Infras. and New App

Build the pool from the first 80% of the ECW source, then train a separate model for each held-out scenario. Run this block once with the New Infras. preset and once with the New App preset.

```bash
cd "$REPO/models/baseline/GlobalPooling/vade_pooling"
PYTHONPATH="$REPO" python global_pooling_main.py \
  --dataX_dir "$REPO/data/ECW_08.npy" \
  --nClusters 551 \
  --series_len 48 \
  --step 12 \
  --source_train_ratio 0.80

cd "$REPO/models/baseline/DynEformer"
PYTHONPATH="$REPO" python dyneformer_main.py \
  --data "$TRAIN" \
  --test_data "$TEST" \
  --num_epoches "$EPOCHS" \
  --batch_size "$BATCH" \
  --test_step "$TEST_STEP" \
  --seed "$SEED"
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
export BEST_K=REPLACE_WITH_PRINTED_INTEGER

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

Run the 11 matched seeds for MPGFormer and all baselines except DynEformer with:

```bash
cd "$REPO"
python scripts/run_baseline_11_seeds.py --dataset MT2
python scripts/summarize_seed_runs.py logs/seed_runs/MT2/runs.jsonl
```

The batch runner accepts `--models` for selected models. It uses the dataset presets above and seed IDs 0 through 10 by default. DynEformer uses the Global Pool preparation commands in Section 4 before its per-seed runs.

The decoder label length is 12 for models with a decoder. Encoder-only baselines use the same 48-point history and 24-point horizon.

Autoformer and Informer use target history with calendar time marks. The other baseline adapters use the first `INPUT_SIZE` history channels; DynEformer also reads ECW static context. These are the inputs used by the training entries in this repository.

- Prepare datasets first with [DATA_GENERATION_GUIDE.md](DATA_GENERATION_GUIDE.md).
- Training entries fit scalers on their training partitions. Validation MSE selects the checkpoint, followed by one test evaluation. DynEformer's global pool is fitted on the training partition.
- ECW uses chronological 64/16/20 train/validation/test; New Infras. and New App use an 80/20 source-trace train/validation split plus a separate test trace. Azure and Google reserve 20% of training series IDs (fixed split seed 0). Materna uses its separate 70/10/20 train/validation/test files.
- Run each model and dataset with matched `--seed` values 0 through 10. Report the mean and sample SD of the final test metrics.
- Each model training or evaluation command emits a `TEST_METRICS` JSON line with the model's name. Combine the logs for models being compared, then run `python "$REPO/scripts/summarize_seed_runs.py" all_runs.log --model MPGFormer --baseline Informer` (substitute the prespecified baseline name). The script checks the 11 seed IDs and reports paired 95% t intervals, raw p-values, and Holm-adjusted p-values across the comparisons printed together.
- Select the comparison baseline before inspecting held-out test results. Paired intervals across training seeds describe run-to-run variation on the fixed test set; they do not measure uncertainty over new traces. If testing several datasets or model pairs, account for multiple comparisons when interpreting p-values.
- All scripts report MSE, MAE, and SMAPE on the inverse-scaled workload target.
- Only calendar fields (columns 1 through 3) are treated as known future covariates. Measured resource fields are hidden at forecast time.
- Regenerate processed data and all paper metrics after changes to data preparation or evaluation code. The repository does not contain the raw eleven-seed results required to verify the paper's tables or paired intervals.
- Azure targets are divided by 10 before fitting the training scaler for MPGFormer and every baseline. Inverse scaling retains that fixed unit conversion; Azure MAE and MSE are therefore in target/10 units.
- Use smaller `EPOCHS` for quick debugging, but keep the listed values for paper reproduction.
