#!/usr/bin/env python3

import os
import glob
import json
import argparse
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


COLOR_LIFELINES = "tab:blue"
COLOR_ROTTERDAM = "tab:orange"


# =========================
# Confidence interval
# =========================

def compute_confidence(metric, N_train, N_test, alpha=0.95):

    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric) ** 2.0)
        metric_std = np.sqrt((1.0 / N_iterations + N_test / N_train) * S_uj)
        CI = st.t.interval(alpha, N_iterations - 1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)

    return CI


# =========================
# Load data
# =========================

def load_fold_jsons(files):

    lifelines = []
    rotterdam = []

    for fp in files:
        with open(fp) as f:
            data = json.load(f)

        lifelines.append(np.array(data["Lifelines"], dtype=float))
        rotterdam.append(np.array(data["Rotterdam Study"], dtype=float))

    return lifelines, rotterdam


def summarize(folds, N_train, N_test, alpha=0.95):

    min_len = min(len(f) for f in folds)
    folds = [f[:min_len] for f in folds]

    M = np.stack(folds, axis=0)
    n_iter = M.shape[1]

    mean = np.zeros(n_iter)
    lo = np.zeros(n_iter)
    hi = np.zeros(n_iter)

    for i in range(n_iter):
        vals = M[:, i]
        mean[i] = np.mean(vals)
        ci = compute_confidence(vals, N_train, N_test, alpha)
        lo[i], hi[i] = ci

    return mean, lo, hi


# =========================
# Axis helpers
# =========================

def format_y_ticks():

    ax = plt.gca()
    ticks = ax.get_yticks()
    ax.set_yticklabels([f"{t:.3f}" for t in ticks])


def compute_ylim(mean, lo=None, hi=None, baseline=None, shade=True):

    if shade and lo is not None and hi is not None:
        ymin = np.min(lo)
        ymax = np.max(hi)
    else:
        ymin = np.min(mean)
        ymax = np.max(mean)

    if baseline is not None:
        ymin = min(ymin, baseline)
        ymax = max(ymax, baseline)

    if ymin == ymax:
        return ymin - 0.01, ymax + 0.01

    margin = (ymax - ymin) * 0.15
    return ymin - margin, ymax + margin


# =========================
# Plot functions
# =========================

def plot_combined(x, lif, lif_lo, lif_hi,
                  rs, rs_lo, rs_hi,
                  output, metric, shade=True):

    fig = plt.figure(figsize=(4, 3))

    # Lifelines
    plt.plot(x, lif, marker="o", color=COLOR_LIFELINES, label="Lifelines")
    if shade:
        plt.fill_between(x, lif_lo, lif_hi,
                         color=COLOR_LIFELINES, alpha=0.2)

    plt.axhline(
        y=lif[0],
        color=COLOR_LIFELINES,
        linestyle="--",
        label="Before aggregation (Lifelines)"
    )

    # Rotterdam
    plt.plot(x, rs, marker="o", color=COLOR_ROTTERDAM,
             label="Rotterdam Study")
    if shade:
        plt.fill_between(x, rs_lo, rs_hi,
                         color=COLOR_ROTTERDAM, alpha=0.2)

    plt.axhline(
        y=rs[0],
        color=COLOR_ROTTERDAM,
        linestyle="--",
        label="Before aggregation (Rotterdam Study)"
    )

    ymin, ymax = compute_ylim(
        np.concatenate([lif, rs]),
        np.concatenate([lif_lo, rs_lo]) if shade else None,
        np.concatenate([lif_hi, rs_hi]) if shade else None,
        shade=shade
    )

    plt.ylim(ymin, ymax)

    plt.legend(loc="best", fontsize=8)
    plt.ylabel(metric)
    plt.xlabel("Update iteration")

    format_y_ticks()

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def plot_single(x, mean, lo, hi,
                label, color,
                output, metric, shade=True):

    fig = plt.figure(figsize=(4, 3))

    plt.plot(x, mean, marker="o", color=color, label=label)

    if shade:
        plt.fill_between(x, lo, hi, color=color, alpha=0.2)

    plt.axhline(
        y=mean[0],
        color=color,
        linestyle="--",
        label=f"Before aggregation ({label})"
    )

    ymin, ymax = compute_ylim(
        mean,
        lo if shade else None,
        hi if shade else None,
        baseline=mean[0],
        shade=shade
    )

    plt.ylim(ymin, ymax)

    plt.legend(loc="best", fontsize=8)
    plt.ylabel(metric)
    plt.xlabel("Update iteration")

    format_y_ticks()

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


# =========================
# Print results
# =========================

def print_stats(name, mean, lo, hi):

    print(f"\n=== {name} ===")
    print("Iter\tMean\tCI_low\tCI_high")

    for i in range(len(mean)):
        print(f"{i+1}\t{mean[i]:.6f}\t{lo[i]:.6f}\t{hi[i]:.6f}")


# =========================
# Main
# =========================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--glob",
                        default="ci_results_*_fold_*.json")

    parser.add_argument("--output",
                        default="ci_multi_fold.png")

    parser.add_argument("--shade",
                        dest="shade",
                        action="store_true")

    parser.add_argument("--no-shade",
                        dest="shade",
                        action="store_false")

    parser.set_defaults(shade=True)

    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))

    if not files:
        raise ValueError("No JSON files found")

    print("Found files:")
    for f in files:
        print(" ", f)

    lif_folds, rs_folds = load_fold_jsons(files)

    lif_train, lif_test = 118584, 14823
    rs_train, rs_test = 8084, 1006

    lif_mean, lif_lo, lif_hi = summarize(
        lif_folds, lif_train, lif_test
    )

    rs_mean, rs_lo, rs_hi = summarize(
        rs_folds, rs_train, rs_test
    )

    print_stats("Lifelines", lif_mean, lif_lo, lif_hi)
    print_stats("Rotterdam Study", rs_mean, rs_lo, rs_hi)

    x = list(range(1, len(lif_mean) + 1))

    base, ext = os.path.splitext(args.output)

    plot_combined(
        x,
        lif_mean,
        lif_lo,
        lif_hi,
        rs_mean,
        rs_lo,
        rs_hi,
        args.output,
        "C-statistic",
        shade=args.shade
    )

    plot_single(
        x,
        lif_mean,
        lif_lo,
        lif_hi,
        "Lifelines",
        COLOR_LIFELINES,
        f"{base}_Lifelines{ext}",
        "C-statistic",
        shade=args.shade
    )

    plot_single(
        x,
        rs_mean,
        rs_lo,
        rs_hi,
        "Rotterdam Study",
        COLOR_ROTTERDAM,
        f"{base}_Rotterdam_Study{ext}",
        "C-statistic",
        shade=args.shade
    )


if __name__ == "__main__":
    main()
