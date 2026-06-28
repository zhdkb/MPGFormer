# Data

This repository does not include large workload datasets or raw traces.

Please download the original datasets from their official sources and run the preparation scripts under `scripts/` to generate tensors in the following format:

```text
[num_series, num_periods, num_features]
```

The first feature is treated as the target workload.

## Dataset Sources

- ECW workload data: see the original ECW/DynEformer release and `DATA_GENERATION_GUIDE.md`.
- Microsoft Azure VM traces: https://github.com/Azure/AzurePublicDataset
- Google cluster traces: https://github.com/google/cluster-data
- Materna workload traces: https://atlarge-research.com/gwa-t-13/

## Expected Processed Files

```text
ECW_08.npy
ECW_newmac.pkl
ECW_newapp.pkl
tr_vm_official.npy
te_vm_official.npy
tr_google_v1.npy
te_google_v1.npy
tr_materna_mt1.npy
te_materna_mt1.npy
tr_materna_mt2.npy
te_materna_mt2.npy
tr_materna_mt3.npy
te_materna_mt3.npy
```

Put the generated files in this directory before running training scripts.
