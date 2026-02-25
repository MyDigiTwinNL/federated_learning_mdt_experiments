import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


# Define fixed colors (same everywhere)
COLOR_LIFELINES = "tab:blue"
COLOR_ROTTERDAM = "tab:orange"


def nice_ylim(values, margin_ratio=0.15):
    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        return vmin - 0.01, vmax + 0.01

    margin = (vmax - vmin) * margin_ratio
    return vmin - margin, vmax + margin


def plot_single(x, values, label, color, output_path, metric="C-statistic"):

    fig = plt.figure(figsize=(4, 3))

    plt.plot(
        x,
        values,
        marker="o",
        linestyle="-",
        color=color,
        label=label
    )

    plt.axhline(
        y=values[0],
        color=color,
        linestyle="--",
        label=f"Before aggregation ({label})"
    )

    ymin, ymax = nice_ylim(values)
    plt.ylim(ymin, ymax)

    yticks = np.linspace(ymin, ymax, 5)
    plt.yticks(yticks, [f"{t:.3f}" for t in yticks])

    plt.legend(fontsize=8)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Update iteration", fontsize=12)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)


def main(json_path, output_path, metric="C-statistic"):

    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    base, ext = os.path.splitext(output_path)

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lifelines = data.get("Lifelines", [])
    rotterdam = data.get("Rotterdam Study", [])

    if not lifelines or not rotterdam:
        raise ValueError("Missing data")

    n = min(len(lifelines), len(rotterdam))
    lifelines = lifelines[:n]
    rotterdam = rotterdam[:n]

    x = list(range(1, n + 1))

    # =========================
    # Combined plot
    # =========================

    fig = plt.figure(figsize=(4, 3))

    plt.plot(
        x,
        lifelines,
        marker="o",
        linestyle="-",
        color=COLOR_LIFELINES,
        label="Lifelines"
    )

    plt.axhline(
        y=lifelines[0],
        color=COLOR_LIFELINES,
        linestyle="--",
        label="Before aggregation (Lifelines)"
    )

    plt.plot(
        x,
        rotterdam,
        marker="o",
        linestyle="-",
        color=COLOR_ROTTERDAM,
        label="Rotterdam Study"
    )

    plt.axhline(
        y=rotterdam[0],
        color=COLOR_ROTTERDAM,
        linestyle="--",
        label="Before aggregation (Rotterdam Study)"
    )

    plt.legend(fontsize=8)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Update iteration", fontsize=12)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)

    # =========================
    # Separate figures
    # =========================

    plot_single(
        x,
        lifelines,
        "Lifelines",
        COLOR_LIFELINES,
        f"{base}_Lifelines{ext}",
        metric
    )

    plot_single(
        x,
        rotterdam,
        "Rotterdam Study",
        COLOR_ROTTERDAM,
        f"{base}_Rotterdam_Study{ext}",
        metric
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", default="fedavg_ci.png")

    args = parser.parse_args()

    main(args.json, args.output)
