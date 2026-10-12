"""Compatibility entry for the supported global-pool cluster search."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("search_azure_k.py")), run_name="__main__")
