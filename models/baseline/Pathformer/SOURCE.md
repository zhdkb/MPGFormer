# Pathformer source

Architecture components were adapted from [https://github.com/decisionintelligence/pathformer](https://github.com/decisionintelligence/pathformer) at commit `ea85d82932215e171357da47b3bc82d502344758`.

The referenced upstream revision has no top-level license file.

Retained components:

- `layers/AMS.py`
- `layers/Embedding.py`
- `layers/Layer.py`
- `layers/RevIN.py`
- `models/PathFormer.py`
- `utils/Other.py`

The model device allocation was adapted to support both CPU and CUDA.
