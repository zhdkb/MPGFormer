#!/usr/bin/env python3
"""Draw the MPGFormer introduction motivation figure.

The figure is intentionally a motivation/problem illustration rather than a
model architecture diagram. It shows that the same long-running workload can
move through different operational states, and the most reliable forecasting
evidence changes with those states.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "pics"


COLORS = {
    "ink": "#24313f",
    "muted": "#667085",
    "grid": "#d7dde5",
    "panel": "#f8fafc",
    "stable": "#3a6ea5",
    "burst": "#b75d69",
    "co": "#4e8d7c",
    "drift": "#c28b35",
    "gray": "#a4acb8",
}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4b2")
    ax.spines["bottom"].set_color("#9aa4b2")
    ax.tick_params(axis="both", colors=COLORS["muted"], labelsize=7, length=2)


def panel_label(ax, label: str, title: str):
    ax.text(
        0.0,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        va="bottom",
    )
    ax.text(
        0.145,
        1.06,
        title,
        transform=ax.transAxes,
        fontsize=9.6,
        color=COLORS["ink"],
        va="bottom",
    )


def add_window(ax, x0, x1, label, color):
    y0, y1 = ax.get_ylim()
    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=1.0,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
        zorder=0,
    )
    ax.add_patch(rect)
    ax.text(
        (x0 + x1) / 2,
        y1 - 0.06 * (y1 - y0),
        label,
        ha="center",
        va="top",
        fontsize=7.5,
        color=color,
        fontweight="bold",
    )


def rounded_box(
    ax,
    xy,
    width,
    height,
    text,
    edge,
    face="#ffffff",
    fontsize=8.2,
    weight="normal",
    align="center",
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=weight,
        linespacing=1.25,
    )
    return box


def arrow(ax, start, end, color="#9aa4b2", lw=1.0, mutation_scale=10):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arr)
    return arr


def mini_trace(ax, kind: str, color: str):
    t = np.linspace(0, 1, 80)
    rng = np.random.default_rng(2)
    if kind == "stable":
        y = 0.50 + 0.16 * np.sin(2 * np.pi * 3.0 * t) + 0.015 * rng.normal(size=t.size)
        ax.plot(t, y, color=color, lw=1.8)
    elif kind == "burst":
        y = 0.34 + 0.04 * np.sin(2 * np.pi * 2.0 * t)
        y += 0.46 * np.exp(-((t - 0.55) / 0.055) ** 2)
        ax.plot(t, y, color=color, lw=1.8)
    elif kind == "co":
        base = 0.38 + 0.18 * np.sin(2 * np.pi * 1.35 * t + 0.3)
        common = 0.20 * np.exp(-((t - 0.68) / 0.18) ** 2)
        for shift, alpha in [(-0.06, 0.55), (0.04, 0.45)]:
            ax.plot(t, base + common + shift, color=COLORS["gray"], lw=1.0, alpha=alpha)
        ax.plot(t, base + common, color=color, lw=1.8)
    elif kind == "drift":
        y = 0.32 + 0.06 * np.sin(2 * np.pi * 2.0 * t)
        y += 0.34 / (1 + np.exp(-(t - 0.52) * 18))
        ax.plot(t, y, color=color, lw=1.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#c7ced8")
    ax.set_facecolor("#ffffff")


def main():
    np.random.seed(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(13.2, 5.15), constrained_layout=False)
    gs = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.45, 1.28, 1.42, 1.34],
        wspace=0.34,
        left=0.045,
        right=0.985,
        top=0.76,
        bottom=0.16,
    )

    fig.suptitle(
        "Workload Forecasting as State-Contingent Evidence Selection",
        fontsize=11.6,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "The entity is not new, but its forecasting state changes over time.",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["muted"],
    )

    # (a) Long workload trace with changing windows.
    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "(a)", "Same workload, changing states")
    x = np.arange(240)
    rng = np.random.default_rng(3)
    y = 0.58 + 0.14 * np.sin(2 * np.pi * x / 38) + 0.035 * rng.normal(size=x.size)
    y += 0.48 * np.exp(-((x - 96) / 7) ** 2)
    y += 0.18 * np.exp(-((x - 150) / 18) ** 2)
    y += 0.34 / (1 + np.exp(-(x - 188) / 8))
    y += 0.05 * np.sin(2 * np.pi * (x - 185) / 18) * (x > 180)
    ax1.plot(x, y, color=COLORS["ink"], lw=1.55)
    ax1.set_xlim(0, 239)
    ax1.set_ylim(y.min() - 0.12, y.max() + 0.20)
    add_window(ax1, 24, 70, "periodic", COLORS["stable"])
    add_window(ax1, 78, 116, "burst", COLORS["burst"])
    add_window(ax1, 128, 172, "co-fluctuation", COLORS["co"])
    add_window(ax1, 184, 230, "drift", COLORS["drift"])
    ax1.set_xlabel("Time", fontsize=8, color=COLORS["muted"], labelpad=3)
    ax1.set_ylabel("Workload", fontsize=8, color=COLORS["muted"], labelpad=3)
    ax1.set_xticks([0, 60, 120, 180, 240])
    ax1.set_yticks([])
    ax1.grid(axis="y", color=COLORS["grid"], lw=0.6, alpha=0.7)
    style_axis(ax1)

    # (b) State examples.
    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "(b)", "Operational states")
    ax2.axis("off")
    states = [
        ("Stable periodic", "stable", COLORS["stable"]),
        ("Bursty high-load", "burst", COLORS["burst"]),
        ("Co-fluctuation", "co", COLORS["co"]),
        ("Drifted / atypical", "drift", COLORS["drift"]),
    ]
    positions = [(0.02, 0.58), (0.54, 0.58), (0.02, 0.12), (0.54, 0.12)]
    for (label, kind, color), (x0, y0) in zip(states, positions):
        ax2.text(
            x0,
            y0 + 0.34,
            label,
            transform=ax2.transAxes,
            fontsize=8.1,
            fontweight="bold",
            color=color,
            va="bottom",
        )
        inset = ax2.inset_axes([x0, y0, 0.42, 0.28])
        mini_trace(inset, kind, color)
    ax2.text(
        0.02,
        0.02,
        "Different states appear during ordinary operation.",
        transform=ax2.transAxes,
        fontsize=7.6,
        color=COLORS["muted"],
    )

    # (c) Evidence changes by state.
    ax3 = fig.add_subplot(gs[0, 2])
    panel_label(ax3, "(c)", "Reliable evidence is state-dependent")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    rows = [
        ("Periodic", "Temporal rhythm\nand multi-scale memory", COLORS["stable"]),
        ("Burst", "Recent local change\nand short-term shape", COLORS["burst"]),
        ("Co-fluct.", "Similar workloads\nand shared events", COLORS["co"]),
        ("Drift", "Prototype uncertainty\nand cautious extrapolation", COLORS["drift"]),
    ]
    y_positions = [0.78, 0.56, 0.34, 0.12]
    for (state, evidence, color), yy in zip(rows, y_positions):
        rounded_box(
            ax3,
            (0.02, yy),
            0.26,
            0.13,
            state,
            edge=color,
            face="#ffffff",
            fontsize=8.2,
            weight="bold",
        )
        arrow(ax3, (0.30, yy + 0.065), (0.39, yy + 0.065), color=color, lw=1.0)
        rounded_box(
            ax3,
            (0.42, yy),
            0.53,
            0.13,
            evidence,
            edge="#c8d0da",
            face="#fbfcfe",
            fontsize=8.0,
        )
    ax3.text(
        0.49,
        0.025,
        "The dominant cue is not fixed.",
        ha="center",
        fontsize=7.6,
        color=COLORS["muted"],
    )

    # (d) Problem formulation as evidence selection.
    ax4 = fig.add_subplot(gs[0, 3])
    panel_label(ax4, "(d)", "Problem formulation")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")

    rounded_box(
        ax4,
        (0.05, 0.63),
        0.30,
        0.18,
        "Current\nwindow",
        edge="#b7c1cc",
        face="#ffffff",
        fontsize=8.1,
        weight="bold",
    )
    rounded_box(
        ax4,
        (0.43, 0.60),
        0.30,
        0.24,
        "Evidence\nselection",
        edge=COLORS["ink"],
        face="#f8fafc",
        fontsize=8.2,
        weight="bold",
    )
    rounded_box(
        ax4,
        (0.80, 0.63),
        0.16,
        0.18,
        "Forecast",
        edge="#b7c1cc",
        face="#ffffff",
        fontsize=8.1,
        weight="bold",
    )
    arrow(ax4, (0.36, 0.72), (0.42, 0.72), color="#8e99a8", lw=1.0)
    arrow(ax4, (0.74, 0.72), (0.79, 0.72), color="#8e99a8", lw=1.0)

    chips = [
        ("Temporal\nmemory", COLORS["stable"], (0.08, 0.35)),
        ("Local\nchange", COLORS["burst"], (0.33, 0.35)),
        ("Cross-series\ncue", COLORS["co"], (0.58, 0.35)),
        ("Drift\nuncertainty", COLORS["drift"], (0.33, 0.12)),
    ]
    for label, color, pos in chips:
        rounded_box(
            ax4,
            pos,
            0.20,
            0.13,
            label,
            edge=color,
            face="#ffffff",
            fontsize=7.7,
        )

    ax4.text(
        0.50,
        0.025,
        "Prediction = extrapolation + evidence selection.",
        ha="center",
        fontsize=7.6,
        color=COLORS["muted"],
    )

    for ext in ("svg", "pdf", "png"):
        out = OUT_DIR / f"mpgformer_intro_motivation.{ext}"
        if ext == "png":
            fig.savefig(out, dpi=320, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main()
