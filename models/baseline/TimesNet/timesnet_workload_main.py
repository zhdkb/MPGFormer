"""Train and evaluate TimesNet on the workload benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workload_runner import parse_args, run  # noqa: E402


if __name__ == "__main__":
    run(parse_args("TimesNet"))
