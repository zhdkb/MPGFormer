# Reproduce Models

This file summarizes commands for `models/MPGFormer` and the baseline model folders under `models/baseline/`:

- MPG-Former (`models/MPGFormer`)
- DynEformer (`models/baseline/DynEformer`)
- Autoformer (`models/baseline/auto_informer`)
- Informer (`models/baseline/Informer`)
- Fremer (`models/baseline/Fremer`)
- iTransformer (`models/baseline/iTransformer`)
- PDF (`models/baseline/PDF`)
- TFEGRU (`models/baseline/TFEGRU`)
- SOFTS (`models/baseline/SOFTS`)
- MICN (`models/baseline/MICN`)

Run from your machine and activate the environment first. Replace `/path/to/MPGFormer_GitHub` with the path where you cloned this repository:

```bash
cd /path/to/MPGFormer_GitHub
conda activate DynEformer
```

The reproduction scenarios use these files:

- ECW: `data/ECW_08.npy`
- App Switch: `data/ECW_switch.pkl`
- New Infras.: `data/ECW_newmac.pkl`
- New App: `data/ECW_newapp.pkl`
- Azure: `data/tr_vm_official.npy` and `data/te_vm_official.npy`
- Google v1: `data/tr_google_v1.npy` and `data/te_google_v1.npy`
- Materna MT-1: `data/tr_materna_mt1.npy`, `data/va_materna_mt1.npy`, and `data/te_materna_mt1.npy`
- Materna MT-2: `data/tr_materna_mt2.npy`, `data/va_materna_mt2.npy`, and `data/te_materna_mt2.npy`
- Materna MT-3: `data/tr_materna_mt3.npy`, `data/va_materna_mt3.npy`, and `data/te_materna_mt3.npy`

For Azure, the scripts below follow the same target scale convention as DynEformer: the target column is divided by `10` before scaling and raw MSE/MAE are reported after inverse scaling.

Materna follows the Fremer setup: MT-1, MT-2, and MT-3 are prepared as separate datasets. VMs with more than `2%` missing points are filtered, and each trace is split by time span into train/validation/test with a `7:1:2` ratio. Regenerate the arrays with:

```bash
cd /path/to/MPGFormer_GitHub

python scripts/prepare_materna_workload_dataset.py \
  --raw_root data/GWA-T-13_Materna-Workload-Traces \
  --trace all \
  --match_fremer_counts true
```

## 1. MPG-Former

```bash
cd /path/to/MPGFormer_GitHub/models/MPGFormer
```

ECW:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --static_start 4 \
  --select_metric mse
```
App Switch:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/ECW_08.npy \
  --test_data ../../data/ECW_switch.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --static_start 4 \
  --select_metric mse
```
New Infras.:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/ECW_08.npy \
  --test_data ../../data/ECW_newmac.pkl \
  --epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --static_start 4 \
  --select_metric mse
```
New App:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/ECW_08.npy \
  --test_data ../../data/ECW_newapp.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --static_start 4 \
  --select_metric mse
```
Azure:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/tr_vm_official.npy \
  --test_data ../../data/te_vm_official.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --static_start 4 \
  --select_metric mse
```
Google v1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/tr_google_v1.npy \
  --test_data ../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --static_start 6 \
  --select_metric mse
```
Materna MT-1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/tr_materna_mt1.npy \
  --test_data ../../data/te_materna_mt1.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --static_start 10 \
  --select_metric mse
```
Materna MT-2:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/tr_materna_mt2.npy \
  --test_data ../../data/te_materna_mt2.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --static_start 10 \
  --select_metric mse
```
Materna MT-3:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python train_mpgformer.py \
  --data ../../data/tr_materna_mt3.npy \
  --test_data ../../data/te_materna_mt3.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --static_start 10 \
  --select_metric mse
```

## 2. DynEformer

DynEformer must build a dataset-specific Global Pool before training/testing. Following the paper's experimental protocol, first search the cluster number `K` on the training split by AE latent GMM BIC, then rebuild the Global Pool with the selected `K`, and finally run DynEformer with the same `K`.

The current K-search helper is named `search_azure_k.py` for historical reasons, but it can search any `.npy` workload array used below. Each search command prints:

```text
BEST_K_BY_BIC=<K>
```

Use that printed value as `<BEST_K_...>` in the corresponding pool-build and training commands. For the ECW-family paper scripts, the original code loads `global_pool_c551_s48_s12.pkl` and `DynEformer_best_n551_drop01.pt`, so the paper reproduction path keeps the searched ECW pool at `K=551`.

Search the ECW Global Pool size:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_search_cluster

PYTHONPATH=/path/to/MPGFormer_GitHub \
python search_azure_k.py \
  --dataX_dir ../../../../data/ECW_08.npy \
  --candidates 100,150,200,250,300,350,400,450,500,551,600,650 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the ECW Global Pool with the searched paper value `K=551`:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_pooling

PYTHONPATH=/path/to/MPGFormer_GitHub \
python global_pooling_main.py \
  --dataX_dir ../../../../data/ECW_08.npy \
  --nClusters 551 \
  --series_len 48 \
  --step 12
```

ECW:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/DynEformer

PYTHONPATH=/path/to/MPGFormer_GitHub \
python dyneformer_main.py \
  --num_epoches 300 \
  --batch_size 256
```

App Switch:

Before this test, make sure the ECW K-search and ECW Global Pool build above have already been run. This script evaluates the ECW-trained DynEformer under an application-switch scenario and loads the same ECW `K=551` pool.

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python dynet_app_switch.py
```

New Infras.:

Before this test, make sure the ECW K-search and ECW Global Pool build above have already been run. This script evaluates the ECW-trained DynEformer under a new-infrastructure scenario and loads the same ECW `K=551` pool.

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python dynet_newmac.py \
  --batch_size 32
```

New App:

Before this test, make sure the ECW K-search and ECW Global Pool build above have already been run. This script evaluates the ECW-trained DynEformer under a new-application scenario and loads the same ECW `K=551` pool.

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python dynet_newadded.py
```

Search the Azure Global Pool size:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_search_cluster

PYTHONPATH=/path/to/MPGFormer_GitHub \
python search_azure_k.py \
  --dataX_dir ../../../../data/tr_vm_official.npy \
  --candidates 20,30,40,50,60,80,100,120,150,200,250,300,400,500 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the Azure Global Pool with `<BEST_K_AZURE>`:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_pooling

PYTHONPATH=/path/to/MPGFormer_GitHub \
python global_pooling_main.py \
  --dataX_dir ../../../../data/tr_vm_official.npy \
  --nClusters <BEST_K_AZURE> \
  --series_len 48 \
  --step 12 \
  --gmm_reg_covar 1e-4 \
  --gmm_n_init 3
```
Azure:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/DynEformer

PYTHONPATH=/path/to/MPGFormer_GitHub \
python dyneformer_azure_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --pool_clusters <BEST_K_AZURE> \
  --num_epoches 300 \
  --batch_size 64
```
Search the Google v1 Global Pool size:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_search_cluster

PYTHONPATH=/path/to/MPGFormer_GitHub \
python search_azure_k.py \
  --dataX_dir ../../../../data/tr_google_v1.npy \
  --candidates 10,20,30,40,50,60,80,100,120,150,200,250,300 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the Google v1 Global Pool with `<BEST_K_GOOGLE_V1>`:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_pooling

PYTHONPATH=/path/to/MPGFormer_GitHub \
python global_pooling_main.py \
  --dataX_dir ../../../../data/tr_google_v1.npy \
  --nClusters <BEST_K_GOOGLE_V1> \
  --series_len 48 \
  --step 12 \
  --gmm_reg_covar 1e-4 \
  --gmm_n_init 3
```
Google v1:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/DynEformer

PYTHONPATH=/path/to/MPGFormer_GitHub \
python dyneformer_google_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --pool_clusters <BEST_K_GOOGLE_V1> \
  --num_epoches 300 \
  --batch_size 64
```
Search the Materna MT-1 Global Pool size:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_search_cluster

PYTHONPATH=/path/to/MPGFormer_GitHub \
python search_azure_k.py \
  --dataX_dir ../../../../data/tr_materna_mt1.npy \
  --candidates 10,20,30,40,50,60,80,100,120,150,200,250,300 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the Materna MT-1 Global Pool with `<BEST_K_MATERNA_MT1>`:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_pooling

PYTHONPATH=/path/to/MPGFormer_GitHub \
python global_pooling_main.py \
  --dataX_dir ../../../../data/tr_materna_mt1.npy \
  --nClusters <BEST_K_MATERNA_MT1> \
  --series_len 48 \
  --step 12 \
  --gmm_reg_covar 1e-4 \
  --gmm_n_init 3
```
Materna MT-1:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/DynEformer

PYTHONPATH=/path/to/MPGFormer_GitHub \
python dyneformer_google_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --pool_clusters <BEST_K_MATERNA_MT1> \
  --num_epoches 30 \
  --batch_size 32
```
Search the Materna MT-2 Global Pool size:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_search_cluster

PYTHONPATH=/path/to/MPGFormer_GitHub \
python search_azure_k.py \
  --dataX_dir ../../../../data/tr_materna_mt2.npy \
  --candidates 10,20,30,40,50,60,80,100,120,150,200,250,300 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the Materna MT-2 Global Pool with `<BEST_K_MATERNA_MT2>`:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_pooling

PYTHONPATH=/path/to/MPGFormer_GitHub \
python global_pooling_main.py \
  --dataX_dir ../../../../data/tr_materna_mt2.npy \
  --nClusters <BEST_K_MATERNA_MT2> \
  --series_len 48 \
  --step 12 \
  --gmm_reg_covar 1e-4 \
  --gmm_n_init 3
```
Materna MT-2:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/DynEformer

PYTHONPATH=/path/to/MPGFormer_GitHub \
python dyneformer_google_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --pool_clusters <BEST_K_MATERNA_MT2> \
  --num_epoches 30 \
  --batch_size 32
```
Search the Materna MT-3 Global Pool size:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_search_cluster

PYTHONPATH=/path/to/MPGFormer_GitHub \
python search_azure_k.py \
  --dataX_dir ../../../../data/tr_materna_mt3.npy \
  --candidates 10,20,30,40,50,60,80,100,120,150,200,250,300 \
  --series_len 48 \
  --step 12 \
  --pre_epoch 500 \
  --reg_covar 1e-4 \
  --n_init 3 \
  --output_dir ../pools
```

Build the Materna MT-3 Global Pool with `<BEST_K_MATERNA_MT3>`:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/GlobalPooling/vade_pooling

PYTHONPATH=/path/to/MPGFormer_GitHub \
python global_pooling_main.py \
  --dataX_dir ../../../../data/tr_materna_mt3.npy \
  --nClusters <BEST_K_MATERNA_MT3> \
  --series_len 48 \
  --step 12 \
  --gmm_reg_covar 1e-4 \
  --gmm_n_init 3
```
Materna MT-3:

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/DynEformer

PYTHONPATH=/path/to/MPGFormer_GitHub \
python dyneformer_google_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --pool_clusters <BEST_K_MATERNA_MT3> \
  --num_epoches 30 \
  --batch_size 32
```

## 3. Autoformer

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/auto_informer
```

ECW:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 12
```
App Switch:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 2
```
New Infras.:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --train_epochs 300 \
  --batch_size 32 \
  --test_step 2
```
New App:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 2
```
Azure:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 12
```
Google v1:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 12
```
Materna MT-1:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --test_step 12
```
Materna MT-2:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --test_step 12
```
Materna MT-3:

```bash
python auto_info_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --test_step 12
```

## 4. Informer

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/Informer
```

ECW:

```bash
python informer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 12
```
App Switch:

```bash
python informer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 2
```
New Infras.:

```bash
python informer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --train_epochs 300 \
  --batch_size 32 \
  --test_step 2
```
New App:

```bash
python informer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 2
```
Azure:

```bash
python informer_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 12
```
Google v1:

```bash
python informer_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --test_step 12
```
Materna MT-1:

```bash
python informer_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --test_step 12
```
Materna MT-2:

```bash
python informer_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --test_step 12
```
Materna MT-3:

```bash
python informer_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --test_step 12
```



## 5. Fremer

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/Fremer
```

ECW:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
App Switch:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
New Infras.:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --test_step 2
```
New App:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
Azure:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
Google v1:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --test_step 12
```
Materna MT-1:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-2:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-3:

```bash
python fremer_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```

## 6. iTransformer

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/iTransformer
```

ECW:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
App Switch:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --train_epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
New Infras.:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --train_epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --test_step 2
```
New App:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --train_epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
Azure:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
Google v1:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --train_epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --test_step 12
```
Materna MT-1:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-2:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-3:

```bash
python itransformer_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --train_epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```

## 7. PDF

PDF follows the paper's periodicity decoupling idea: each input window is reshaped by several candidate periods, global periodic tokens are modeled with a Transformer branch, local within-period variations are modeled with a convolution branch, and all period branches are combined for the final forecast. The default periods below are adapted to this project's `48 -> 24` workload setting: `24 12 6`.

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/PDF
```

ECW:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
App Switch:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
New Infras.:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --test_step 2
```
New App:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
Azure:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
Google v1:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --test_step 12
```
Materna MT-1:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-2:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-3:

```bash
python pdf_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```

## 8. TFEGRU

TFEGRU follows the paper's time-frequency enhanced GRU design: a frequency-domain enhancement block first strengthens periodic components, then channel-independent GRU modeling and multi-head attention produce the final forecast. The adapted script reports raw-scale MSE, MAE, and SMAPE with the same target-column convention as the other workload models.

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/TFEGRU
```

ECW:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
App Switch:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
New Infras.:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --test_step 2
```
New App:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
Azure:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
Google v1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --test_step 12
```
Materna MT-1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-2:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-3:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python tfegru_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```


## 9. SOFTS

SOFTS follows the paper's Series-Core Fusion idea: each variable is embedded as a token, and STAR repeatedly aggregates series information into a compact core and redistributes it back to each series. The adapted script uses the same workload windows, target column, Azure `/10` target scale convention, and raw-scale MSE/MAE/SMAPE reporting as the other project models.

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/SOFTS
```

ECW:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
App Switch:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
New Infras.:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --test_step 2
```
New App:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
Azure:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
Google v1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --test_step 12
```
Materna MT-1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-2:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-3:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python softs_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```

## 10. MICN

MICN follows the paper's multi-scale local and global context modeling design: it first decomposes the sequence into seasonal and trend parts, then uses multi-scale downsampling/upsampling convolution and isometric convolution to capture local details and global correlations. The adapted script uses generated hourly time features for MICN's temporal embedding and reports raw-scale MSE, MAE, and SMAPE under the same target-column convention as the other workload models.

```bash
cd /path/to/MPGFormer_GitHub/models/baseline/MICN
```

ECW:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
App Switch:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_switch.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
New Infras.:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newmac.pkl \
  --epochs 300 \
  --batch_size 32 \
  --input_size 4 \
  --test_step 2
```
New App:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/ECW_08.npy \
  --test_data ../../../data/ECW_newapp.pkl \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 2
```
Azure:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/tr_vm_official.npy \
  --test_data ../../../data/te_vm_official.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 4 \
  --test_step 12
```
Google v1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/tr_google_v1.npy \
  --test_data ../../../data/te_google_v1.npy \
  --epochs 300 \
  --batch_size 64 \
  --input_size 6 \
  --test_step 12
```
Materna MT-1:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/tr_materna_mt1.npy \
  --test_data ../../../data/te_materna_mt1.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-2:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/tr_materna_mt2.npy \
  --test_data ../../../data/te_materna_mt2.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```
Materna MT-3:

```bash
PYTHONPATH=/path/to/MPGFormer_GitHub \
python micn_workload_main.py \
  --train_data ../../../data/tr_materna_mt3.npy \
  --test_data ../../../data/te_materna_mt3.npy \
  --epochs 30 \
  --batch_size 64 \
  --input_size 10 \
  --test_step 12
```

## Quick Smoke Tests

To check that an entry can run before a full reproduction, reduce epochs:

```bash
python auto_info_workload_main.py --train_epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python train_mpgformer.py --epochs 2 --batch_size 32 --test_data ../../data/ECW_switch.pkl
python informer_workload_main.py --train_epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python fremer_workload_main.py --epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python itransformer_workload_main.py --train_epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python pdf_workload_main.py --epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python tfegru_workload_main.py --epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python softs_workload_main.py --epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
python micn_workload_main.py --epochs 2 --batch_size 32 --test_data ../../../data/ECW_switch.pkl
```

If CUDA OOM occurs, lower `--batch_size` to `32` or `16`.
