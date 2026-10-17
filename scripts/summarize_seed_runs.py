"""Summarize TEST_METRICS log lines or plain JSONL records.

Example: python scripts/summarize_seed_runs.py run.log --baseline Informer
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def read_records(paths):
    """Read unique finite test records from logs or JSONL files."""
    records = {}
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if "TEST_METRICS " in line:
                    record = json.loads(line.split("TEST_METRICS ", 1)[1])
                elif line.lstrip().startswith("{"):
                    record = json.loads(line)
                    if not {"dataset", "model", "seed", "mse", "mae", "smape"} <= record.keys():
                        continue
                else:
                    continue
                if not all(np.isfinite(float(record[name])) for name in ("mse", "mae", "smape")):
                    raise ValueError("Non-finite test metric in {}".format(path))
                key = (record["dataset"], record["model"], record["seed"])
                if key in records:
                    raise ValueError("Duplicate test result for {}".format(key))
                records[key] = record
    if not records:
        raise ValueError("No TEST_METRICS records found")
    return records


def holm_adjust(pvalues):
    """Control family-wise error across the paired comparisons printed together."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    ranked = np.minimum(1.0, pvalues[order] * (len(pvalues) - np.arange(len(pvalues))))
    adjusted[order] = np.maximum.accumulate(ranked)
    return adjusted


def summarize(records, baseline=None, metric="mae", model=None):
    """Report 11-seed mean/SD and optional baseline-minus-model paired tests."""
    groups = defaultdict(dict)
    for (dataset, model_name, seed), record in records.items():
        groups[(dataset, model_name)][seed] = float(record[metric])
    for (dataset, model_name), by_seed in sorted(groups.items()):
        seeds = set(by_seed)
        if seeds != set(range(11)):
            raise ValueError("{} {} needs exactly matched seeds 0..10; found {}".format(
                dataset, model_name, sorted(seeds)))
        values = np.array([by_seed[i] for i in range(11)])
        print("{} {} {} mean={:.8f} sample_sd={:.8f}".format(
            dataset, model_name, metric, values.mean(), values.std(ddof=1)))

    if baseline:
        comparisons = []
        for dataset, model_name in sorted(groups):
            if model_name == baseline or (model and model_name != model):
                continue
            peer = groups.get((dataset, baseline))
            if peer is None:
                raise ValueError("No {} result for {}".format(baseline, dataset))
            differences = np.array([peer[i] - groups[(dataset, model_name)][i] for i in range(11)])
            mean = differences.mean()
            se = differences.std(ddof=1) / np.sqrt(11)
            half_width = stats.t.ppf(0.975, df=10) * se
            if np.isclose(se, 0.0, atol=1e-15):
                pvalue = 1.0 if mean == 0 else 0.0
            else:
                pvalue = stats.ttest_1samp(differences, 0).pvalue
            comparisons.append((dataset, model_name, mean, mean - half_width, mean + half_width, pvalue))
        if model and not comparisons:
            raise ValueError("No paired results found for {}".format(model))
        corrected = holm_adjust([row[-1] for row in comparisons])
        for (dataset, model_name, mean, low, high, pvalue), adjusted in zip(comparisons, corrected):
            print("{} {} vs {} paired_{} difference={:.8f} CI95=[{:.8f}, {:.8f}] p={:.8g} p_holm={:.8g}".format(
                dataset, model_name, baseline, metric, mean, low, high, pvalue, adjusted))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Training logs containing TEST_METRICS lines")
    parser.add_argument("--baseline", help="Model name for paired differences")
    parser.add_argument("--model", help="Model name to include in paired comparisons")
    parser.add_argument("--metric", choices=("mse", "mae", "smape"), default="mae")
    args = parser.parse_args()
    summarize(read_records(args.logs), args.baseline, args.metric, args.model)
