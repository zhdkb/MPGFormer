"""Compatibility entry for the Google Informer experiment."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--train_data" not in sys.argv:
        sys.argv += ["--train_data", "../../../data/tr_google_v1.npy"]
    if "--test_data" not in sys.argv:
        sys.argv += ["--test_data", "../../../data/te_google_v1.npy"]
    runpy.run_path(str(Path(__file__).with_name("informer_workload_main.py")), run_name="__main__")
