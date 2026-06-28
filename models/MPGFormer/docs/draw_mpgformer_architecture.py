import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle


OUT_DIR = Path(__file__).resolve().parent


COLORS = {
    "blue": "#6f95d7",
    "blue_fill": "#eef5ff",
    "green": "#6aa35f",
    "green_fill": "#f1faee",
    "pink": "#d46b8c",
    "pink_fill": "#fff1f5",
    "yellow": "#e5b95c",
    "yellow_fill": "#fff9df",
    "purple": "#9a75c9",
    "purple_fill": "#f4efff",
    "orange": "#e68a2e",
    "grey": "#5a5a5a",
    "light_grey": "#f7f7f7",
    "black": "#1a1a1a",
}


def add_round_box(ax, xy, w, h, text="", fc="white", ec="#333333", lw=1.0, fs=7.5, bold=False, z=2):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.15,rounding_size=0.35",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            weight="bold" if bold else "normal",
            color=COLORS["black"],
            zorder=z + 1,
        )
    return box


def add_group(ax, x, y, w, h, title, color, fill, label):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=0.45",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.add_patch(rect)
    ax.add_patch(Rectangle((x, y + h - 3.0), w, 3.0, facecolor=color, alpha=0.35, edgecolor=color, lw=0.8))
    ax.text(
        x + w / 2,
        y + h - 1.5,
        f"({label}) {title}",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=COLORS["black"],
    )


def arrow(ax, p1, p2, color="#333333", lw=1.2, style="-|>", rad=0.0, ls="-"):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
    )
    ax.add_patch(arr)


def badge(ax, x, y, n, color):
    c = Circle((x, y), 0.95, facecolor=color, edgecolor="white", lw=1.0, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, str(n), ha="center", va="center", fontsize=8.5, color="white", weight="bold", zorder=6)


def tiny_series(ax, x, y, w, h, color="#2b6cb0", seed=0, burst=False, periodic=False):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 80)
    if periodic:
        s = 0.55 + 0.22 * np.sin(2 * np.pi * 4.5 * t) + 0.04 * rng.normal(size=len(t))
    elif burst:
        s = 0.35 + 0.08 * rng.normal(size=len(t))
        s += 0.65 * np.exp(-((t - 0.55) ** 2) / 0.004)
        s += 0.35 * np.exp(-((t - 0.78) ** 2) / 0.003)
    else:
        s = 0.45 + 0.18 * np.sin(2 * np.pi * (2.2 * t + 0.1)) + 0.08 * rng.normal(size=len(t))
    s = (s - s.min()) / max(1e-6, (s.max() - s.min()))
    xs = x + w * t
    ys = y + h * (0.12 + 0.76 * s)
    ax.plot(xs, ys, color=color, lw=1.15, zorder=3)
    ax.plot([x, x + w], [y, y], color="#444444", lw=0.4)
    ax.plot([x, x], [y, y + h], color="#444444", lw=0.4)


def tiny_matrix(ax, x, y, rows=4, cols=6, cell=0.42, color="#92b7e8", edge="#355985"):
    for i in range(rows):
        for j in range(cols):
            alpha = 0.35 + 0.45 * ((i + 2 * j) % 5) / 5
            ax.add_patch(
                Rectangle(
                    (x + j * cell, y + i * cell),
                    cell,
                    cell,
                    facecolor=color,
                    edgecolor=edge,
                    lw=0.45,
                    alpha=alpha,
                    zorder=2,
                )
            )


def tiny_graph(ax, x, y, color="#4a90d9", edge="#4b4b4b"):
    pts = np.array([[0, 0], [1.2, 0.6], [2.0, -0.1], [0.6, -1.0], [1.7, -1.1]]) * 0.8
    pts[:, 0] += x
    pts[:, 1] += y
    for a, b in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (0, 3)]:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=edge, lw=0.65, zorder=2)
    for i, p in enumerate(pts):
        ax.add_patch(Circle(tuple(p), 0.18, facecolor=color if i == 1 else "white", edgecolor=edge, lw=0.8, zorder=3))


def draw_patch_stack(ax, x, y):
    lengths = [4.2, 3.4, 2.5, 1.6]
    labels = ["p=6", "p=12", "p=24", "p=48"]
    for idx, (ln, lab) in enumerate(zip(lengths, labels)):
        yy = y - idx * 1.05
        ax.text(x - 0.2, yy + 0.15, lab, fontsize=6.5, ha="right", va="center")
        cells = max(1, int(round(48 / int(lab.split("=")[1]))))
        cell_w = ln / cells
        for c in range(cells):
            ax.add_patch(Rectangle((x + c * cell_w, yy), cell_w - 0.03, 0.35, facecolor="#9bbce8", edgecolor="#37699f", lw=0.35))


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(22, 7.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    ax.text(
        50,
        48.6,
        "MPG-Former: Drift-Aware Multi-scale Prototype-Graph Transformer for Cloud Workload Forecasting",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )

    add_group(ax, 1.0, 3.0, 24.5, 44.0, "Input Preparation & History Representation", COLORS["blue"], COLORS["blue_fill"], "a")
    add_group(ax, 26.5, 3.0, 46.0, 44.0, "Multi-Evidence Extraction", COLORS["green"], COLORS["green_fill"], "b")
    add_group(ax, 73.5, 3.0, 25.5, 44.0, "Drift-aware Fusion, Decoder & Outputs", COLORS["pink"], COLORS["pink_fill"], "c")

    # Left group.
    add_round_box(ax, (2.0, 37.0), 22.5, 8.5, fc="white", ec=COLORS["blue"], lw=0.9)
    badge(ax, 3.3, 44.2, 1, COLORS["blue"])
    ax.text(4.5, 44.2, "Raw Workload Window", fontsize=9.2, weight="bold", va="center")
    tiny_series(ax, 3.0, 38.2, 8.7, 3.7, seed=1)
    ax.text(12.3, 41.5, r"$X \in \mathbb{R}^{B\times72\times C}$", fontsize=8.5)
    ax.add_patch(Rectangle((12.4, 38.3), 6.7, 0.8, facecolor="#77a9e8", edgecolor="#477dbd", lw=0.6))
    ax.add_patch(Rectangle((19.1, 38.3), 3.4, 0.8, facecolor="#f3a6a6", edgecolor="#d56b6b", lw=0.6))
    ax.text(15.7, 39.45, "48 history", fontsize=6.5, ha="center")
    ax.text(20.8, 39.45, "24 future", fontsize=6.5, ha="center", color="#9e2f2f")
    ax.text(15.9, 37.65, "future target masked", fontsize=6.2, color="#9e2f2f", ha="center")

    add_round_box(ax, (2.0, 26.0), 22.5, 9.0, fc="white", ec=COLORS["blue"], lw=0.9)
    badge(ax, 3.3, 33.6, 2, COLORS["blue"])
    ax.text(4.5, 33.6, "Target RevIN", fontsize=9.2, weight="bold", va="center")
    tiny_series(ax, 3.2, 27.6, 5.2, 2.4, seed=2)
    add_round_box(ax, (9.6, 28.2), 4.8, 2.2, r"$\mu_w,\sigma_w$", fc=COLORS["purple_fill"], ec=COLORS["purple"], fs=8.2)
    add_round_box(ax, (16.0, 28.2), 6.4, 2.2, "Normalize\nTarget Only", fc=COLORS["purple_fill"], ec=COLORS["purple"], fs=7.0)
    ax.text(9.5, 31.3, r"$target\_history \in \mathbb{R}^{B\times48}$", fontsize=7.8)
    arrow(ax, (8.6, 28.9), (9.5, 28.9), color=COLORS["blue"])
    arrow(ax, (14.5, 28.9), (15.9, 28.9), color=COLORS["blue"])

    add_round_box(ax, (2.0, 5.0), 22.5, 19.0, fc="white", ec=COLORS["blue"], lw=0.9)
    badge(ax, 3.3, 22.5, 3, COLORS["blue"])
    ax.text(4.5, 22.5, "History Representation Generation", fontsize=9.2, weight="bold", va="center")
    tiny_matrix(ax, 3.0, 17.2, rows=4, cols=5, cell=0.55)
    ax.text(2.9, 15.8, r"$\tilde{X}^{enc}$", fontsize=7.5)
    boxes = [
        ((7.4, 17.0), 3.8, "Linear\nProjection"),
        ((12.2, 17.0), 3.8, "+ PE"),
        ((16.9, 17.0), 4.2, "Gated\nStatic Context"),
    ]
    for (xy, w, txt) in boxes:
        add_round_box(ax, xy, w, 2.6, txt, fc=COLORS["yellow_fill"], ec=COLORS["yellow"], fs=6.8)
    add_round_box(ax, (9.0, 9.6), 4.6, 2.1, "Static context\n(optional)", fc="#ebf8e6", ec=COLORS["green"], fs=6.6)
    for i in range(4):
        ax.add_patch(Rectangle((14.5 + i * 0.65, 10.2), 0.45, 0.45, facecolor="#a5d896", edgecolor=COLORS["green"], lw=0.5))
    tiny_matrix(ax, 20.8, 16.8, rows=4, cols=4, cell=0.50, color="#c3a5e8", edge="#7254a1")
    ax.text(19.6, 14.9, r"$H_s \in \mathbb{R}^{B\times48\times128}$", fontsize=7.4)
    arrow(ax, (5.9, 18.1), (7.3, 18.1), color=COLORS["blue"])
    arrow(ax, (11.3, 18.1), (12.1, 18.1), color=COLORS["blue"])
    arrow(ax, (16.1, 18.1), (16.8, 18.1), color=COLORS["blue"])
    arrow(ax, (13.6, 10.7), (17.4, 16.9), color=COLORS["green"], rad=-0.2)

    # Middle group top branch.
    add_round_box(ax, (27.6, 34.0), 44.0, 10.5, fc="white", ec=COLORS["green"], lw=0.9)
    badge(ax, 28.8, 42.9, 4, COLORS["green"])
    ax.text(30.0, 42.9, "Temporal Evidence Branch", fontsize=9.2, weight="bold", va="center")
    tiny_matrix(ax, 28.4, 38.3, rows=4, cols=5, cell=0.42, color="#c3a5e8", edge="#7254a1")
    ax.text(27.9, 37.1, r"$H_s$", fontsize=7.2)
    add_round_box(ax, (33.2, 38.2), 7.4, 2.7, "Non-stationary\nSelf-Attention", fc="#fff7dc", ec=COLORS["yellow"], fs=6.8)
    ax.text(33.0, 36.7, r"$A=\mathrm{softmax}(\tau QK^T/\sqrt{d}+\delta)$", fontsize=6.4)
    add_round_box(ax, (42.3, 38.2), 7.0, 2.7, "Moving-average\nDecomposition", fc="#ecf8ed", ec=COLORS["green"], fs=6.8)
    tiny_series(ax, 43.0, 35.2, 2.7, 1.2, color="#2f7d32", seed=3)
    tiny_series(ax, 46.3, 35.2, 2.7, 1.2, color="#2b6cb0", seed=4, periodic=True)
    ax.text(42.8, 34.7, "trend", fontsize=5.8, color="#2f7d32")
    ax.text(46.0, 34.7, "seasonal", fontsize=5.8, color="#2b6cb0")
    add_round_box(ax, (51.0, 36.1), 8.2, 5.2, "Short-term\nTemporal Encoder\nConv3 + Conv5 + BiGRU", fc=COLORS["purple_fill"], ec=COLORS["purple"], fs=6.6)
    add_round_box(ax, (61.0, 37.0), 5.8, 3.1, r"$c_{time}$" + "\n" + r"$\mathbb{R}^{B\times128}$", fc="#e7f4ff", ec=COLORS["blue"], fs=7.3)
    ax.text(60.4, 34.9, r"$encoded \in \mathbb{R}^{B\times48\times128}$", fontsize=6.8)
    arrow(ax, (30.9, 39.1), (33.1, 39.1), color=COLORS["green"])
    arrow(ax, (40.7, 39.1), (42.2, 39.1), color=COLORS["green"])
    arrow(ax, (49.4, 39.1), (50.9, 39.1), color=COLORS["green"])
    arrow(ax, (59.3, 38.7), (60.9, 38.7), color=COLORS["green"])

    # Prototype branch.
    add_round_box(ax, (27.6, 20.2), 44.0, 12.2, fc="white", ec=COLORS["green"], lw=0.9)
    badge(ax, 28.8, 30.9, 5, COLORS["green"])
    ax.text(30.0, 30.9, "Multi-scale Prototype Branch", fontsize=9.2, weight="bold", va="center")
    tiny_series(ax, 28.4, 22.0, 5.3, 2.2, seed=5, periodic=True)
    draw_patch_stack(ax, 35.0, 27.7)
    add_round_box(ax, (42.0, 24.1), 5.5, 4.8, "Time tokens\n+\nFFT tokens", fc="#f2fff0", ec=COLORS["green"], fs=6.7)
    add_round_box(ax, (49.0, 25.0), 5.8, 3.2, "Scale\nTokens", fc=COLORS["yellow_fill"], ec=COLORS["yellow"], fs=7.0)
    tiny_matrix(ax, 56.2, 25.1, rows=4, cols=5, cell=0.42, color="#cce2ff", edge="#37699f")
    ax.text(55.5, 29.3, r"Prototype Memory $M$", fontsize=6.8)
    add_round_box(ax, (60.2, 24.6), 5.9, 3.9, "Prototype\nMatching\nSoftmax", fc="#eef7ff", ec=COLORS["blue"], fs=6.7)
    add_round_box(ax, (67.0, 25.0), 3.9, 3.0, r"$c_{proto}$" + "\n" + r"$B\times128$", fc="#e9f8e4", ec=COLORS["green"], fs=6.7)
    add_round_box(ax, (52.3, 20.8), 9.4, 1.9, r"$future\_padding \in \mathbb{R}^{B\times24}$", fc="#ffecec", ec="#d96b6b", fs=6.5)
    arrow(ax, (33.7, 23.2), (35.0, 26.5), color=COLORS["green"])
    arrow(ax, (39.4, 25.5), (41.9, 25.5), color=COLORS["green"])
    arrow(ax, (47.7, 26.4), (48.9, 26.4), color=COLORS["green"])
    arrow(ax, (54.9, 26.4), (56.0, 26.4), color=COLORS["green"])
    arrow(ax, (58.8, 26.4), (60.1, 26.4), color=COLORS["green"])
    arrow(ax, (66.2, 26.4), (66.9, 26.4), color=COLORS["green"])
    arrow(ax, (69.0, 25.0), (61.8, 21.8), color="#d96b6b", lw=1.0, ls="--", rad=0.1)

    # Graph branch.
    add_round_box(ax, (27.6, 5.0), 44.0, 12.2, fc="white", ec=COLORS["green"], lw=0.9)
    badge(ax, 28.8, 15.6, 6, COLORS["green"])
    ax.text(30.0, 15.6, "Dynamic Graph Branch", fontsize=9.2, weight="bold", va="center")
    tiny_graph(ax, 29.6, 10.5, color="#5f9de0")
    ax.text(28.4, 6.3, "target-history\nsimilarity top-k", fontsize=6.2)
    tiny_matrix(ax, 35.3, 9.4, rows=4, cols=4, cell=0.45, color="#a9c5e9", edge="#37699f")
    ax.text(34.5, 8.2, r"$workload\_adj$", fontsize=6.4)
    tiny_graph(ax, 43.1, 10.5, color="#83c46d")
    tiny_matrix(ax, 48.6, 9.4, rows=4, cols=4, cell=0.45, color="#bfe3b5", edge="#4a8d3f")
    ax.text(48.2, 8.2, r"$context\_adj$", fontsize=6.4)
    add_round_box(ax, (55.0, 9.1), 7.0, 3.5, "GCN /\nMessage Passing", fc="#f0f6ff", ec=COLORS["blue"], fs=6.8)
    add_round_box(ax, (64.0, 9.1), 5.9, 3.5, r"$c_{graph}$" + "\n" + r"$B\times128$", fc="#e9f8e4", ec=COLORS["green"], fs=6.8)
    ax.text(55.4, 6.2, "contrastive alignment", fontsize=6.2, color=COLORS["purple"])
    arrow(ax, (32.2, 10.0), (35.0, 10.0), color=COLORS["green"])
    arrow(ax, (38.1, 10.2), (42.0, 10.2), color=COLORS["green"])
    arrow(ax, (51.4, 10.2), (54.9, 10.2), color=COLORS["green"])
    arrow(ax, (62.1, 10.8), (63.9, 10.8), color=COLORS["green"])

    # Right group.
    add_round_box(ax, (74.4, 34.0), 23.5, 10.5, fc="white", ec=COLORS["pink"], lw=0.9)
    badge(ax, 75.6, 42.9, 7, COLORS["pink"])
    ax.text(76.9, 42.9, "Drift-aware Router", fontsize=9.2, weight="bold", va="center")
    add_round_box(ax, (75.4, 39.0), 5.0, 2.3, r"$c_{time}$", fc="#e7f4ff", ec=COLORS["blue"], fs=7.0)
    add_round_box(ax, (75.4, 36.4), 5.0, 2.3, r"$c_{graph}$", fc="#e9f8e4", ec=COLORS["green"], fs=7.0)
    add_round_box(ax, (75.4, 33.8), 5.0, 2.3, r"$c_{proto}$", fc="#fff4dd", ec=COLORS["orange"], fs=7.0)
    add_round_box(ax, (81.5, 36.2), 6.3, 4.7, "Drift features\nz-score | |Δ|\nstd | entropy", fc="#fff7fb", ec=COLORS["pink"], fs=6.4)
    add_round_box(ax, (89.0, 37.0), 4.7, 3.6, "Router MLP\n+ Softmax", fc="#fff0f4", ec=COLORS["pink"], fs=6.5)
    ax.text(86.5, 34.5, r"$c_{fused}=w_t c_t+w_g c_g+w_p c_p$", fontsize=6.7)
    arrow(ax, (80.6, 40.1), (88.9, 39.2), color=COLORS["pink"])
    arrow(ax, (80.6, 37.5), (88.9, 38.6), color=COLORS["pink"])
    arrow(ax, (87.9, 38.5), (88.9, 38.5), color=COLORS["pink"])

    add_round_box(ax, (74.4, 21.0), 23.5, 11.0, fc="white", ec=COLORS["pink"], lw=0.9)
    badge(ax, 75.6, 30.6, 8, COLORS["pink"])
    ax.text(76.9, 30.6, "Decoder & Prediction Head", fontsize=9.2, weight="bold", va="center")
    add_round_box(ax, (75.5, 25.2), 6.5, 3.3, "Last 12 Hist.\n+\nfuture_padding 24", fc="#fff3f3", ec="#d96b6b", fs=6.4)
    add_round_box(ax, (84.0, 26.0), 5.5, 3.2, "Enhanced\nMemory", fc="#f4efff", ec=COLORS["purple"], fs=6.6)
    ax.text(83.0, 24.5, r"$memory=X_E+\mathrm{Proj}(c_{fused})$", fontsize=6.1)
    add_round_box(ax, (91.0, 25.2), 5.6, 3.3, "Transformer\nDecoder", fc="#fff7dc", ec=COLORS["yellow"], fs=6.8)
    arrow(ax, (82.1, 26.8), (90.9, 26.8), color=COLORS["pink"])
    arrow(ax, (89.6, 27.8), (90.9, 27.8), color=COLORS["purple"])

    add_round_box(ax, (74.4, 5.0), 23.5, 13.4, fc="white", ec=COLORS["pink"], lw=0.9)
    badge(ax, 75.6, 16.7, 9, COLORS["pink"])
    ax.text(76.9, 16.7, "Outputs", fontsize=9.2, weight="bold", va="center")
    add_round_box(ax, (75.5, 13.0), 6.2, 2.4, "Prediction\nHead", fc="#eef5ff", ec=COLORS["blue"], fs=6.7)
    tiny_series(ax, 83.0, 12.6, 6.8, 2.3, color="#c0393b", seed=6)
    ax.text(82.8, 10.8, r"$\hat{y}\in\mathbb{R}^{B\times24}$", fontsize=7.2)
    add_round_box(ax, (91.0, 12.6), 5.9, 2.8, "q10/q50/q90\ninterval", fc="#fff1f5", ec=COLORS["pink"], fs=6.2)
    add_round_box(ax, (76.0, 7.0), 5.6, 2.3, "router\nweights", fc="#f7fff3", ec=COLORS["green"], fs=6.2)
    add_round_box(ax, (82.6, 7.0), 5.6, 2.3, "drift\nscore", fc="#fff7dc", ec=COLORS["yellow"], fs=6.2)
    add_round_box(ax, (89.2, 7.0), 6.2, 2.3, "prototype weights\nadaptive graph", fc="#f4efff", ec=COLORS["purple"], fs=5.8)
    arrow(ax, (96.6, 26.8), (96.0, 15.5), color=COLORS["pink"], rad=-0.1)
    arrow(ax, (81.8, 14.2), (82.9, 14.2), color=COLORS["pink"])
    arrow(ax, (89.9, 14.0), (90.9, 14.0), color=COLORS["pink"])

    # Cross-region arrows.
    arrow(ax, (24.5, 18.0), (27.4, 39.2), color="#4b8a4b", rad=-0.18)
    arrow(ax, (24.5, 28.5), (27.3, 26.2), color="#4b8a4b")
    arrow(ax, (24.5, 20.0), (27.4, 10.4), color="#4b8a4b", rad=0.16)
    arrow(ax, (66.9, 38.8), (75.2, 40.0), color=COLORS["pink"])
    arrow(ax, (70.9, 26.5), (75.2, 34.8), color=COLORS["pink"], rad=-0.15)
    arrow(ax, (69.9, 10.8), (75.2, 37.2), color=COLORS["pink"], rad=0.2)
    arrow(ax, (61.7, 21.8), (75.4, 26.3), color="#d96b6b", ls="--", rad=-0.12)

    fig.tight_layout(pad=0.3)
    pdf_path = OUT_DIR / "mpgformer_architecture.pdf"
    png_path = OUT_DIR / "mpgformer_architecture.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
