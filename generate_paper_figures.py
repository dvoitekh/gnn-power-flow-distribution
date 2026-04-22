"""Generate publication-quality figures for the MPCE paper."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs import FIGURES_DIR, MV_GRIDS, LV_GRIDS, RESULTS_DIR
from analyze_results import load_all_experiments

# IEEE two-column: column width ~3.5in, full width ~7.16in
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "text.usetex": False,
})

PAPER_DIR = FIGURES_DIR.parent / "paper"


def fig1_mvlv_gap_boxplot(experiments):
    """Fig 1: MV/LV gap boxplot — baseline vs combined."""
    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)

    # Baseline
    bl_mv = bl[bl.voltage_level == "MV"].groupby("grid")["mae_vm"].mean().values * 1000
    bl_lv = bl[bl.voltage_level == "LV"].groupby("grid")["mae_vm"].mean().values * 1000
    bp1 = ax1.boxplot([bl_mv, bl_lv], tick_labels=["MV", "LV"],
                       patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", linewidth=1.5))
    bp1["boxes"][0].set_facecolor("#2196F3")
    bp1["boxes"][0].set_alpha(0.6)
    bp1["boxes"][1].set_facecolor("#FF9800")
    bp1["boxes"][1].set_alpha(0.6)
    _, p_bl = stats.mannwhitneyu(bl_mv, bl_lv, alternative="two-sided")
    ratio_bl = bl_lv.mean() / bl_mv.mean()
    ax1.set_title(f"Baseline GraphSAGE\nLV/MV = {ratio_bl:.2f}x (p = {p_bl:.3f})")
    ax1.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)")
    ax1.grid(axis="y", alpha=0.3)

    # Combined
    e5_mv = e5[e5.voltage_level == "MV"].groupby("grid")["mae_vm"].mean().values * 1000
    e5_lv = e5[e5.voltage_level == "LV"].groupby("grid")["mae_vm"].mean().values * 1000
    bp2 = ax2.boxplot([e5_mv, e5_lv], tick_labels=["MV", "LV"],
                       patch_artist=True, widths=0.5,
                       medianprops=dict(color="black", linewidth=1.5))
    bp2["boxes"][0].set_facecolor("#2196F3")
    bp2["boxes"][0].set_alpha(0.6)
    bp2["boxes"][1].set_facecolor("#FF9800")
    bp2["boxes"][1].set_alpha(0.6)
    _, p_e5 = stats.mannwhitneyu(e5_mv, e5_lv, alternative="two-sided")
    ratio_e5 = e5_lv.mean() / e5_mv.mean()
    ax2.set_title(f"Combined Model\nLV/MV = {ratio_e5:.2f}x (p = {p_e5:.3f})")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = PAPER_DIR / "fig1_mvlv_gap_boxplot.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def fig2_pareto_speedup_accuracy(experiments):
    """Fig 2: Pareto frontier — speedup vs accuracy."""
    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    grid_order = sorted(set(bl["grid"].unique()) & set(e5["grid"].unique()))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for grid in grid_order:
        bl_g = bl[bl.grid == grid]
        e5_g = e5[e5.grid == grid]
        is_mv = grid in MV_GRIDS
        color = "#2196F3" if is_mv else "#FF9800"

        # Baseline point
        ax.scatter(bl_g["speedup"].mean(), bl_g["mae_vm"].mean() * 1000,
                   marker="o", color=color, s=30, alpha=0.7, edgecolors="black", linewidths=0.3)
        # Combined point
        ax.scatter(e5_g["speedup"].mean(), e5_g["mae_vm"].mean() * 1000,
                   marker="^", color=color, s=40, alpha=0.7, edgecolors="black", linewidths=0.3)

        # Arrow from baseline to combined
        ax.annotate("", xy=(e5_g["speedup"].mean(), e5_g["mae_vm"].mean() * 1000),
                    xytext=(bl_g["speedup"].mean(), bl_g["mae_vm"].mean() * 1000),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.5))

    # Legend
    ax.scatter([], [], marker="o", color="gray", s=30, label="Baseline", edgecolors="black", linewidths=0.3)
    ax.scatter([], [], marker="^", color="gray", s=40, label="Combined", edgecolors="black", linewidths=0.3)
    ax.scatter([], [], marker="s", color="#2196F3", s=20, label="MV grids")
    ax.scatter([], [], marker="s", color="#FF9800", s=20, label="LV grids")
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel("Speedup vs. NR (x)")
    ax.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PAPER_DIR / "fig2_pareto_speedup_accuracy.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def fig3_per_grid_comparison(experiments):
    """Fig 3: Per-grid MAE comparison — baseline vs combined."""
    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    grid_order = [g for g in MV_GRIDS + LV_GRIDS
                  if g in bl["grid"].unique() and g in e5["grid"].unique()]

    bl_agg = bl.groupby("grid").agg(mean=("mae_vm", "mean"), std=("mae_vm", "std")).reset_index()
    e5_agg = e5.groupby("grid").agg(mean=("mae_vm", "mean"), std=("mae_vm", "std")).reset_index()

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(grid_order))
    width = 0.35

    bl_vals = [bl_agg[bl_agg.grid == g]["mean"].values[0] * 1000 for g in grid_order]
    bl_errs = [bl_agg[bl_agg.grid == g]["std"].values[0] * 1000 for g in grid_order]
    e5_vals = [e5_agg[e5_agg.grid == g]["mean"].values[0] * 1000 for g in grid_order]
    e5_errs = [e5_agg[e5_agg.grid == g]["std"].values[0] * 1000 for g in grid_order]

    ax.bar(x - width/2, bl_vals, width, yerr=bl_errs, label="Baseline",
           color="#757575", alpha=0.85, capsize=2, error_kw=dict(lw=0.5))
    ax.bar(x + width/2, e5_vals, width, yerr=e5_errs, label="Combined",
           color="#F44336", alpha=0.85, capsize=2, error_kw=dict(lw=0.5))

    # Short grid labels
    short_labels = [g.replace("MV_", "MV\n").replace("LV_", "LV\n") for g in grid_order]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=6)
    ax.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # MV/LV separator
    n_mv = len([g for g in MV_GRIDS if g in grid_order])
    ax.axvline(x=n_mv - 0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(n_mv / 2 - 0.5, ax.get_ylim()[1] * 0.95, "MV", ha="center", fontsize=7, alpha=0.6)
    ax.text(n_mv + (len(grid_order) - n_mv) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "LV",
            ha="center", fontsize=7, alpha=0.6)

    fig.tight_layout()
    path = PAPER_DIR / "fig3_per_grid_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    experiments = load_all_experiments()
    fig1_mvlv_gap_boxplot(experiments)
    fig2_pareto_speedup_accuracy(experiments)
    fig3_per_grid_comparison(experiments)
    print("\nAll paper figures generated.")
