# Baseline implementations

The paper's Table 10 lists MPGFormer plus 16 baselines. This directory contains adapted workload entries for DynEformer, Autoformer, Informer, Fremer, iTransformer, PDF, TFEGRU, SOFTS, and MICN. Seven additional architectures are pinned to their authors' upstream repositories as Git submodules. The top-level MIT license applies to this repository's own code; each submodule retains its own terms.

| Baseline | Upstream source | Pinned commit | Upstream top-level license |
| --- | --- | --- | --- |
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | `204c21e` | Apache-2.0 |
| Non-stationary Transformer | [thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers) | `c4ec406` | MIT |
| Pathformer | [decisionintelligence/pathformer](https://github.com/decisionintelligence/pathformer) | `ea85d82` | Not declared |
| TimeMixer | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) | `e246105` | Apache-2.0 |
| MSGNet | [YoZhibo/MSGNet](https://github.com/YoZhibo/MSGNet) | `953b833` | Not declared |
| CrossGNN | [hqh0728/CrossGNN](https://github.com/hqh0728/CrossGNN) | `0407abd` | Not declared |
| TimesNet | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | `4e938a1` | MIT |

Use `git submodule update --init --recursive` after cloning. [official_runner.py](official_runner.py) maps each upstream model to the workload data, validation checkpoint selection, and inverse-scale test metrics. It loads one upstream project per process because multiple projects use the same Python package names. For CPU-only runs, it substitutes the device allocation string in memory for Pathformer and CrossGNN; the pinned source remains unmodified. The model-specific architecture settings and common defaults are documented in [REPRODUCE_ALL_MODELS_COMMANDS.md](../../REPRODUCE_ALL_MODELS_COMMANDS.md).
