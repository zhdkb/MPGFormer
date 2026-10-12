"""Compatibility entry for Autoformer on ECW.

The workload entry implements the shared train/validation/test protocol.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("auto_info_workload_main.py")), run_name="__main__")
