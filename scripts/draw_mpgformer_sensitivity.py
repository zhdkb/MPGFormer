#!/usr/bin/env python3
"""Draw MPG-Former hyperparameter sensitivity curves from experiment CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.ticker import FormatStrFormatter


DATASETS = [
    ("google_v1", "Google", "s", "-", "black", True),
    ("mt1", "MT1", "s", "--", "black", False),
    ("mt2", "MT2", "^", "-.", "black", False),
    ("mt3", "MT3", "*", ":", "black", True),
]

PARAMETERS = [
    ("e_layers", "Number of Temporal Layers"),
    ("n_heads", "Number of Attention Heads"),
    ("num_prototypes", "Number of Prototypes"),
    ("graph_topk", "Number of Graph Neighbors"),
]

METRICS = [
    ("mse", "MSE"),
    ("mae", "MAE"),
]


def read_results(csv_path: Path) -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    results: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            parameter = row["parameter"]
            value = int(float(row["value"]))
            results[(dataset, parameter)][value] = {
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "smape": float(row["smape_percent"]) if row.get("smape_percent") else float("nan"),
            }
    return results


def rounded_limits(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    margin = (high - low) * 0.055 if high > low else max(abs(high) * 0.04, 0.01)
    low -= margin
    high += margin
    return low, high


def tick_format(low: float, high: float) -> str:
    return "%.3f" if high - low < 0.08 else "%.2f"


def draw(csv_path: Path, output_prefix: Path) -> None:
    results = read_results(csv_path)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 5.6))
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    metric_limits = {}
    for metric, _ in METRICS:
        metric_values = [
            metric_map[metric]
            for series in results.values()
            for metric_map in series.values()
        ]
        metric_limits[metric] = rounded_limits(metric_values)

    panel_specs = []
    for param, x_label in PARAMETERS:
        for metric, y_label in METRICS:
            panel_specs.append((param, x_label, metric, y_label))

    for panel_idx, (param, x_label, metric, y_label) in enumerate(panel_specs):
        row = panel_idx // 4
        col = panel_idx % 4
        ax = axes[row, col]
        all_y = []
        x_values = sorted(
            {
                value
                for dataset, _, _, _, _, _ in DATASETS
                for value in results.get((dataset, param), {})
            }
        )
        for dataset, label, marker, linestyle, color, filled in DATASETS:
            series = results.get((dataset, param), {})
            xs = [x for x in x_values if x in series]
            ys = [series[x][metric] for x in xs]
            all_y.extend(ys)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                linestyle=linestyle,
                color=color,
                linewidth=1.55,
                marker=marker,
                markersize=7.1 if marker != "*" else 10.2,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                clip_on=False,
            )[0]
        ax.set_xlabel(f"{panel_labels[panel_idx]} {x_label}", labelpad=2)
        ax.set_ylabel(y_label, labelpad=0)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])
        ax.set_xlim(x_values[0], x_values[-1])
        y_min, y_max = metric_limits[metric]
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(y_min, y_max, 5))
        ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format(y_min, y_max)))
        ax.tick_params(direction="in", length=5.0, width=1.0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    legend_ax = fig.add_axes([0.020, 0.075, 0.105, 0.86])
    legend_ax.axis("off")
    legend_y = [0.92, 0.65, 0.38, 0.11]
    for y, (_, label, marker, linestyle, color, filled) in zip(legend_y, DATASETS):
        legend_ax.plot(
            [0.05, 0.12, 0.19],
            [y, y, y],
            transform=legend_ax.transAxes,
            linestyle=linestyle,
            color=color,
            linewidth=1.55,
            marker=marker,
            markevery=[1],
            markersize=7.6 if marker != "*" else 10.6,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
        )
        legend_ax.text(0.25, y, label, transform=legend_ax.transAxes, va="center", ha="left", fontsize=16)

    fig.subplots_adjust(left=0.225, right=0.995, top=0.965, bottom=0.17, wspace=0.34, hspace=0.42)
    for ax in axes.flat:
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.090, pos.y0, pos.width, pos.height])
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    bbox = Bbox.from_extents(
        tight_bbox.x0 - 0.03,
        tight_bbox.y0 - 0.28,
        tight_bbox.x1 + 0.28,
        tight_bbox.y1 + 0.28,
    )
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches=bbox, pad_inches=0)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches=bbox, pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="result/mpgformer_sensitivity_results.csv",
        type=Path,
        help="Path to sensitivity result CSV.",
    )
    parser.add_argument(
        "--output",
        default="pics/mpgformer_hyperparameter_sensitivity",
        type=Path,
        help="Output path prefix without extension.",
    )
    args = parser.parse_args()
    draw(args.csv, args.output)


if __name__ == "__main__":
    main()
