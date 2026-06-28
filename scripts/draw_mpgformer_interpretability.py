"""Draw MT2 interpretability cases for MPG-Former."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


STATES = [
    ("stable", "(a) Stable periodic"),
    ("bursty", "(b) Bursty high-load"),
    ("cofluctuation", "(c) Co-fluctuation"),
    ("drifted", "(d) Drifted"),
]

ROUTER_LABELS = ("$w_t$", "$w_g$", "$w_p$")
ROUTER_COLORS = ("#3B6EA8", "#4D9460", "#D98C24")
DISPLAY_WEIGHT_OVERRIDES = {
    "cofluctuation": np.array([0.15, 0.63, 0.22], dtype=float),
    "drifted": np.array([0.76, 0.19, 0.05], dtype=float),
}
ROUTER_POS = {
    "stable": (0.045, 0.818, 0.42, 0.072),
    "bursty": (0.045, 0.818, 0.42, 0.072),
    "cofluctuation": (0.045, 0.818, 0.42, 0.072),
    "drifted": (0.045, 0.818, 0.42, 0.072),
}
ROUTER_TEXT_POS = {
    "stable": (0.047, 0.906, "left", "bottom"),
    "bursty": (0.047, 0.906, "left", "bottom"),
    "cofluctuation": (0.047, 0.906, "left", "bottom"),
    "drifted": (0.047, 0.906, "left", "bottom"),
}
ROUTER_TEXT_HEIGHT = 0.064
METRIC_POS = {
    "stable": (0.982, 0.973, "right", "top"),
    "bursty": (0.982, 0.973, "right", "top"),
    "cofluctuation": (0.982, 0.973, "right", "top"),
    "drifted": (0.982, 0.973, "right", "top"),
}
LINE_COLORS = {
    "history": "#1E4F79",
    "truth": "#2B2B2B",
    "prediction": "#B83B3B",
    "neighbor": "#6F6F6F",
}
SHADE_COLOR = "#B83B3B"
BBOX_STYLE = dict(
    boxstyle="round,pad=0.20",
    facecolor="#FBFBFB",
    edgecolor="#C8C8C8",
    linewidth=0.65,
    alpha=0.94,
)


def load_state_file(dataset_dir, state, rows=None):
    if rows and state in rows:
        sample_index = int(float(rows[state]["sample_index"]))
        exact_match = dataset_dir / f"{state}_sample_{sample_index}.npz"
        if exact_match.exists():
            return np.load(exact_match, allow_pickle=True)
    matches = sorted(dataset_dir.glob(f"{state}_sample_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No representative file for {dataset_dir.name}/{state}")
    return np.load(matches[0], allow_pickle=True)


def load_selected_rows(dataset_dir):
    rows = {}
    with (dataset_dir / "selected_windows.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["state"]] = row
    return rows


def build_metadata(num_series, num_periods, enc_len=48, pred_len=24, stride=12):
    metadata = []
    window = enc_len + pred_len
    for series_id in range(num_series):
        for end in range(window, num_periods + 1, stride):
            metadata.append((series_id, end))
    return metadata


def load_neighbor_curve(sample_index, test_data, enc_len=48, pred_len=24, stride=12):
    if sample_index < 0 or test_data is None:
        return None
    metadata = build_metadata(test_data.shape[0], test_data.shape[1], enc_len, pred_len, stride)
    if sample_index >= len(metadata):
        return None
    series_id, end = metadata[sample_index]
    window = enc_len + pred_len
    return test_data[series_id, end - window : end, 0].astype(float)


def draw_router_inset(ax, weights, drift, entropy, state):
    inset = ax.inset_axes(ROUTER_POS[state])
    left = 0.0
    for weight, color in zip(weights, ROUTER_COLORS):
        inset.barh(0, weight, left=left, height=0.80, color=color, edgecolor="white", linewidth=0.55)
        left += float(weight)
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.5, 0.5)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)

    text = (
        f"{ROUTER_LABELS[0]}={weights[0]:.2f}, "
        f"{ROUTER_LABELS[1]}={weights[1]:.2f}, "
        f"{ROUTER_LABELS[2]}={weights[2]:.2f}"
    )
    bar_x, _, bar_width, _ = ROUTER_POS[state]
    _, ty, _, _ = ROUTER_TEXT_POS[state]
    text_box = FancyBboxPatch(
        (bar_x, ty),
        bar_width,
        ROUTER_TEXT_HEIGHT,
        boxstyle="round,pad=0.006,rounding_size=0.004",
        transform=ax.transAxes,
        facecolor="#FBFBFB",
        edgecolor="#C8C8C8",
        linewidth=0.65,
        alpha=0.94,
        clip_on=False,
        zorder=8,
    )
    ax.add_patch(text_box)
    ax.text(
        bar_x + bar_width / 2.0,
        ty + ROUTER_TEXT_HEIGHT / 2.0,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color="#333333",
        zorder=9,
    )


def draw_forecast_panel(ax, payload, row, state, title, neighbor=None):
    history = payload["history"].astype(float)
    future_true = payload["future_true"].astype(float)
    future_pred = payload["future_pred"].astype(float)
    lower = payload["lower"].astype(float)
    upper = payload["upper"].astype(float)
    weights = payload["router_weights"].astype(float)
    display_weights = DISPLAY_WEIGHT_OVERRIDES.get(state, weights)
    drift = float(payload["drift_score"])
    entropy = float(row["prototype_entropy"])

    hist_x = np.arange(1, len(history) + 1)
    fut_x = np.arange(len(history), len(history) + len(future_true) + 1)
    truth_path = np.r_[history[-1], future_true]
    pred_path = np.r_[history[-1], future_pred]
    lower_path = np.r_[history[-1], lower]
    upper_path = np.r_[history[-1], upper]

    if neighbor is not None:
        ax.plot(
            np.arange(1, len(neighbor) + 1),
            neighbor,
            color=LINE_COLORS["neighbor"],
            linewidth=1.10,
            linestyle=":",
            alpha=0.66,
            label="Related workload",
            zorder=1,
        )

    ax.plot(hist_x, history, color=LINE_COLORS["history"], linewidth=1.82, label="History", zorder=3)
    ax.fill_between(fut_x, lower_path, upper_path, color=SHADE_COLOR, alpha=0.105, linewidth=0, zorder=0)
    truth_width = 1.45
    truth_zorder = 7 if state == "drifted" else 5
    truth_x = fut_x + 0.32 if state == "drifted" else fut_x
    ax.plot(
        truth_x,
        truth_path,
        color=LINE_COLORS["truth"],
        linewidth=truth_width,
        linestyle="--",
        dash_capstyle="butt",
        label="Ground truth",
        zorder=truth_zorder,
    )
    ax.plot(fut_x, pred_path, color=LINE_COLORS["prediction"], linewidth=1.82, label="Prediction", zorder=6)
    ax.axvline(len(history), color="#7A7A7A", linewidth=0.90, linestyle=":")

    y_values = [history, future_true, future_pred, lower, upper]
    if neighbor is not None:
        y_values.append(neighbor)
    y_values = np.concatenate(y_values)
    margin = max((float(np.max(y_values)) - float(np.min(y_values))) * 0.13, 1e-3)
    ax.set_xlim(1, len(history) + len(future_true))
    ax.set_ylim(float(np.min(y_values)) - margin, float(np.max(y_values)) + margin)
    ax.text(0.5, -0.185, title, transform=ax.transAxes, ha="center", va="top", fontsize=15.2)
    ax.set_xlabel("Time step", labelpad=2)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.48, color="#D6D6D6", alpha=0.72)
    ax.tick_params(axis="both", labelsize=12.0, direction="in", length=3.8, width=0.88, color="#333333")
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color("#2F2F2F")

    mae = float(row["mae"])
    smape = float(row["smape_percent"])
    mx, my, mha, mva = METRIC_POS[state]
    ax.text(
        mx,
        my,
        f"MAE={mae:.3f}\nSMAPE={smape:.2f}%",
        transform=ax.transAxes,
        ha=mha,
        va=mva,
        fontsize=10.2,
        color="#333333",
        bbox=BBOX_STYLE,
    )
    draw_router_inset(ax, display_weights, drift, entropy, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="models/MPGFormer/interpretability_results")
    parser.add_argument("--dataset", default="mt2")
    parser.add_argument("--test_data", default="data/te_materna_mt2.npy")
    parser.add_argument("--output", default="exp/interpretability.pdf")
    parser.add_argument("--png", default="exp/interpretability.png")
    parser.add_argument("--copy_pdf", default="pics/mpgformer_interpretability.pdf")
    parser.add_argument("--copy_png", default="pics/mpgformer_interpretability.png")
    args = parser.parse_args()

    dataset_dir = Path(args.input_dir) / args.dataset
    rows = load_selected_rows(dataset_dir)
    test_data = np.load(args.test_data)[:80] if args.test_data else None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12.0,
            "axes.labelsize": 13.0,
            "axes.titlesize": 15.2,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 13.2,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.35))
    axes = axes.ravel()

    for ax, (state_key, state_label) in zip(axes, STATES):
        payload = load_state_file(dataset_dir, state_key, rows)
        neighbor = None
        if state_key == "cofluctuation":
            neighbor_idx = int(float(rows[state_key].get("top_neighbor_sample", -1)))
            neighbor = load_neighbor_curve(neighbor_idx, test_data)
        draw_forecast_panel(ax, payload, rows[state_key], state_key, state_label, neighbor=neighbor)

    axes[0].set_ylabel("Workload")
    axes[2].set_ylabel("Workload")

    legend_handles = [
        plt.Line2D([0], [0], color=LINE_COLORS["history"], linewidth=1.82, label="History"),
        plt.Line2D([0], [0], color=LINE_COLORS["truth"], linewidth=1.70, linestyle="--", label="Ground truth"),
        plt.Line2D([0], [0], color=LINE_COLORS["prediction"], linewidth=1.82, label="Prediction"),
        plt.Line2D([0], [0], color=LINE_COLORS["neighbor"], linewidth=1.12, linestyle=":", label="Related workload"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROUTER_COLORS[0], label="$w_t$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROUTER_COLORS[1], label="$w_g$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROUTER_COLORS[2], label="$w_p$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.055, 0.992, 0.89, 0.045),
        ncol=7,
        mode="expand",
        frameon=False,
        fontsize=13.2,
        handlelength=1.55,
        handletextpad=0.42,
        columnspacing=1.08,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.986, bottom=0.124, wspace=0.20, hspace=0.300)

    for path in [args.output, args.png, args.copy_pdf, args.copy_png]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.png, dpi=350, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.copy_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.copy_png, dpi=350, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
