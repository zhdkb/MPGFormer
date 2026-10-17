# Baselines

Each baseline has its model code and a workload training entry in this directory. All entries use the dataset presets and evaluation protocol in [REPRODUCE_ALL_MODELS_COMMANDS.md](../../REPRODUCE_ALL_MODELS_COMMANDS.md).

| Model | Training entry |
| --- | --- |
| Autoformer | `auto_informer/auto_info_workload_main.py` |
| Informer | `Informer/informer_workload_main.py` |
| Fremer | `Fremer/fremer_workload_main.py` |
| iTransformer | `iTransformer/itransformer_workload_main.py` |
| DynEformer | `DynEformer/dyneformer_main.py` and dataset-specific entries |
| PDF | `PDF/pdf_workload_main.py` |
| TFEGRU | `TFEGRU/tfegru_workload_main.py` |
| SOFTS | `SOFTS/softs_workload_main.py` |
| MICN | `MICN/micn_workload_main.py` |
| PatchTST | `PatchTST/patchtst_workload_main.py` |
| Non-stationary Transformer | `NS_Transformer/ns_transformer_workload_main.py` |
| Pathformer | `Pathformer/pathformer_workload_main.py` |
| TimeMixer | `TimeMixer/timemixer_workload_main.py` |
| MSGNet | `MSGNet/msgnet_workload_main.py` |
| CrossGNN | `CrossGNN/crossgnn_workload_main.py` |
| TimesNet | `TimesNet/timesnet_workload_main.py` |

The model directories contain the architecture components needed for forecasting, their local training entry, and source attribution where imported. [workload_runner.py](workload_runner.py) provides shared data preparation and validation-selected evaluation for the entries that use it. Each entry is an independent command and runs in its own process because several architecture modules use the same Python package names.
