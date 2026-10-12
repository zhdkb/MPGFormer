"""Compatibility entry for the ECW Informer experiment.

The workload entry implements the shared train/validation/test protocol.
"""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.argv = ["--train_data" if arg == "--data_path" else arg for arg in sys.argv]
    runpy.run_path(str(Path(__file__).with_name("informer_workload_main.py")), run_name="__main__")
