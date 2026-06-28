#!/usr/bin/env python3
"""Draw the motivation figure for MPG-Former introduction.

This script synthesizes representative cloud workload traces for illustrating
the problem motivation: one workload can move across stable, bursty, and
drifted states, while different workloads may also co-fluctuate.
"""

from __future__ import annotations

from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = ROOT / "models" / "MPGFormer" / "pic"


def gaussian_pulse(x: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)


def add_impulses(y: np.ndarray, events: list[tuple[int, float, int]]) -> np.ndarray:
    """Add workload-like sudden rises with short exponential tails."""
    out = y.copy()
    n = len(out)
    for center, amp, tail in events:
        for offset in range(tail):
            pos = center + offset
            if 0 <= pos < n:
                out[pos] += amp * np.exp(-offset / max(tail / 2.5, 1.0))
        # A small pre-rise makes spikes less synthetic.
        if 0 <= center - 1 < n:
            out[center - 1] += 0.35 * amp
    return out


def synthesize_single_workload(seed: int = 7) -> tuple[np.ndarray, tuple[int, int]]:
    """Create a single workload with stable, bursty, and drifted segments."""
    rng = np.random.default_rng(seed)
    n = 260
    x = np.arange(n)
    stable_end, burst_end = 86, 158

    y = np.zeros(n, dtype=float)

    # Stable periodic state: repeated low-amplitude demand with small random spikes.
    t = x[:stable_end]
    stable = 16.0 + 5.2 * np.sin(2 * np.pi * t / 14.0 + 0.5)
    stable += 2.0 * np.sin(2 * np.pi * t / 5.0 + 1.4)
    stable += rng.normal(0, 2.0, len(t))
    stable = add_impulses(stable, [(18, 7.0, 3), (42, 6.0, 3), (68, 7.5, 3)])
    y[:stable_end] = stable

    # Bursty high-load state: irregular high peaks and volatile local changes.
    t = x[stable_end:burst_end] - stable_end
    burst = 18.0 + 4.0 * np.sin(2 * np.pi * t / 17.0 + 0.4)
    burst += rng.normal(0, 4.5, len(t))
    burst = add_impulses(burst, [(8, 28.0, 6), (21, 18.0, 4), (35, 33.0, 6), (53, 22.0, 5), (65, 30.0, 5)])
    y[stable_end:burst_end] = burst

    # Drifted / atypical state: mean and variance move to a new regime.
    t = x[burst_end:] - burst_end
    drift_trend = 17.0 + 0.13 * t + 0.030 * (t**1.28)
    drift_cycle = (3.0 + 0.025 * t) * np.sin(2 * np.pi * t / 18.0 + 0.6)
    drift_noise = rng.normal(0, 2.8 + 0.018 * t, len(t))
    drift = drift_trend + drift_cycle + drift_noise
    drift = add_impulses(drift, [(22, 16.0, 7), (48, 11.0, 5), (73, 17.0, 6), (93, 12.0, 5)])
    y[burst_end:] = drift

    return np.clip(y, 0, None), (stable_end, burst_end)


def synthesize_cofluctuation(seed: int = 11) -> tuple[np.ndarray, float]:
    """Create three different workloads sharing a co-fluctuation driver."""
    rng = np.random.default_rng(seed)
    n = 120
    x = np.arange(n)

    shared = (
        13.0
        + 2.0 * np.sin(2 * np.pi * x / 28.0 + 0.2)
        + 1.1 * np.sin(2 * np.pi * x / 9.0 + 1.1)
    )
    shared += 4.6 / (1 + np.exp(-(x - 42) / 3.2))
    shared += 5.8 / (1 + np.exp(-(x - 78) / 2.7))
    shared = add_impulses(shared, [(30, 5.5, 5), (55, 7.5, 6), (84, 8.0, 6), (101, 5.0, 5)])

    y1 = 0.86 * shared + 2.2 + rng.normal(0, 0.85, n)
    y2 = 1.03 * shared + 4.6 + rng.normal(0, 0.90, n)
    y3 = 1.15 * shared + 3.5 + rng.normal(0, 0.95, n)
    y = np.vstack([y1, y2, y3])
    corr = np.corrcoef(y)[np.triu_indices(3, 1)].mean()
    return np.clip(y, 0, None), float(corr)


def main() -> None:
    y_top, (stable_end, burst_end) = synthesize_single_workload()
    x_top = np.arange(len(y_top))

    y_co, avg_corr = synthesize_cofluctuation()
    x_co = np.arange(y_co.shape[1])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.9,
            "axes.labelsize": 16.56,
            "xtick.labelsize": 13.68,
            "ytick.labelsize": 13.68,
            "legend.fontsize": 13.68,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=False,
        gridspec_kw={"height_ratios": [1.08, 1.0], "hspace": 0.36},
    )

    def draw_interval(
        ax,
        left,
        right,
        y,
        label,
        color="#111111",
        text_offset=1.2,
        text_va="bottom",
    ) -> None:
        ax.annotate(
            "",
            xy=(right, y),
            xytext=(left, y),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.25, shrinkA=0, shrinkB=0),
        )
        ax.text(
            (left + right) / 2,
            y + text_offset,
            label,
            ha="center",
            va=text_va,
            color=color,
            fontsize=13.824,
            fontweight="bold",
        )

    def add_axis_arrows(ax) -> None:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", direction="out", length=4.2, width=0.9)
        ax.annotate(
            "",
            xy=(1.012, 0),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.05, mutation_scale=11),
            clip_on=False,
        )
        ax.annotate(
            "",
            xy=(0, 1.055),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.05, mutation_scale=11),
            clip_on=False,
        )

    # Panel (a): state changes in one workload.
    ax = axes[0]
    y_arrow = y_top.max() + 7.0
    ax.plot(x_top, y_top, color="#111111", lw=1.05)
    for boundary in [stable_end, burst_end, len(y_top) - 1]:
        ax.axvline(boundary, color="#6b6b6b", lw=0.9, alpha=0.75)
    draw_interval(ax, 1, stable_end - 1, y_arrow, "Stable periodic", "#2b7a8b")
    draw_interval(ax, stable_end + 1, burst_end - 1, y_arrow, "Bursty high-load", "#c56b6b")
    draw_interval(ax, burst_end + 1, len(y_top) - 2, y_arrow, "Drifted / atypical", "#655a9a")
    ax.set_xlim(0, len(y_top) - 1)
    ax.set_ylim(max(0, y_top.min() - 4.0), y_arrow + 6.0)
    ax.set_xlabel("Time (h)", labelpad=0.0)
    ax.set_ylabel("Workload", labelpad=1.5)
    ax.grid(True, axis="y", color="#d7d7d7", lw=0.55, ls=":")
    add_axis_arrows(ax)

    # Panel (b): co-fluctuation among different workloads.
    ax = axes[1]
    colors = ["#1f5a93", "#b97919", "#2f7f4f"]
    labels = ["Workload A", "Workload B", "Workload C"]
    for yy, color, label in zip(y_co, colors, labels):
        ax.plot(x_co, yy, color=color, lw=1.45, label=label)
    y_arrow = y_co.max() + 0.8
    for boundary in [len(x_co) - 1]:
        ax.axvline(boundary, color="#6b6b6b", lw=0.9, alpha=0.75)
    draw_interval(
        ax,
        1,
        len(x_co) - 2,
        y_arrow,
        "Co-fluctuation window",
        "#2f7f4f",
        text_offset=0.28,
    )
    ax.set_xlim(0, len(x_co) - 1)
    ax.set_ylim(max(0, y_co.min() - 3.0), y_arrow + 5.0)
    ax.set_xlabel("Time (h)", labelpad=0.0)
    ax.set_ylabel("Workload", labelpad=1.5)
    ax.grid(True, axis="y", color="#d7d7d7", lw=0.55, ls=":")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.88),
        ncol=3,
        frameon=False,
        handlelength=1.45,
        columnspacing=0.75,
    )
    add_axis_arrows(ax)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.16, hspace=2.2)
    captions = [
        (axes[0], "(a) State transitions in one workload", 0.082, 16.992),
        (axes[1], "(b) Cross-workload co-fluctuation", 0.082, 16.992),
    ]
    for caption_ax, caption_text, offset, font_size in captions:
        pos = caption_ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y0 - offset,
            caption_text,
            ha="center",
            va="top",
            fontsize=font_size,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in ["intro_challenge_windows", "intro"]:
        for suffix in ["pdf", "png"]:
            fig.savefig(
                OUT_DIR / f"{stem}.{suffix}",
                bbox_inches="tight",
                pad_inches=0.025,
                dpi=300,
            )
    plt.close(fig)

    meta = OUT_DIR / "intro_challenge_windows.txt"
    meta.write_text(
        "\n".join(
            [
                "source=synthetic",
                "top_series=single workload with stable, bursty, and drifted states",
                f"top_length={len(y_top)}",
                f"stable_span=[0, {stable_end})",
                f"burst_span=[{stable_end}, {burst_end})",
                f"drift_span=[{burst_end}, {len(y_top)})",
                "bottom_series=three synthetic co-fluctuating workloads",
                f"bottom_length={y_co.shape[1]}",
                f"bottom_average_pairwise_correlation={avg_corr:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
