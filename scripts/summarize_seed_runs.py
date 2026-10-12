"""Summarize the TEST_METRICS lines emitted by training entries.

Example: python scripts/summarize_seed_runs.py run.log --baseline informer_workload_main
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def read_records(paths):
    records = {}
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if "TEST_METRICS " not in line:
                    continue
                record = json.loads(line.split("TEST_METRICS ", 1)[1])
                key = (record["dataset"], record["model"], record["seed"])
                if key in records:
                    raise ValueError("Duplicate test result for {}".format(key))
                records[key] = record
    return records


def summarize(records, baseline=None, metric="mae"):
    groups = defaultdict(dict)
    for (dataset, model, seed), record in records.items():
        groups[(dataset, model)][seed] = float(record[metric])
    for (dataset, model), by_seed in sorted(groups.items()):
        seeds = set(by_seed)
        if seeds != set(range(11)):
            raise ValueError("{} {} needs exactly matched seeds 0..10; found {}".format(
                dataset, model, sorted(seeds)))
        values = np.array([by_seed[i] for i in range(11)])
        print("{} {} {} mean={:.8f} sample_sd={:.8f}".format(
            dataset, model, metric, values.mean(), values.std(ddof=1)))

    if baseline:
        for dataset, model in sorted(groups):
            if model == baseline:
                continue
            peer = groups.get((dataset, baseline))
            if peer is None:
                raise ValueError("No {} result for {}".format(baseline, dataset))
            differences = np.array([peer[i] - groups[(dataset, model)][i] for i in range(11)])
            mean = differences.mean()
            se = differences.std(ddof=1) / np.sqrt(11)
            half_width = stats.t.ppf(0.975, df=10) * se
            pvalue = stats.ttest_1samp(differences, 0).pvalue
            print("{} {} vs {} paired_{} difference={:.8f} CI95=[{:.8f}, {:.8f}] p={:.8g}".format(
                dataset, model, baseline, metric, mean, mean - half_width, mean + half_width, pvalue))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Training logs containing TEST_METRICS lines")
    parser.add_argument("--baseline", help="Model name for paired differences")
    parser.add_argument("--metric", choices=("mse", "mae", "smape"), default="mae")
    args = parser.parse_args()
    summarize(read_records(args.logs), args.baseline, args.metric)
