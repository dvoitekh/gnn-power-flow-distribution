"""Generate publication-quality figures for the EE&E paper.

Two output paths per figure:
  - PAPER_DIR/<name>.pdf  : LaTeX preprint (column width)
  - ASSETS_DIR/<name>.png : DOCX submission (full-page width, 600 dpi)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs import FIGURES_DIR, MV_GRIDS, LV_GRIDS, RESULTS_DIR
from analyze_results import load_all_experiments

# Wider canvas for both PDF preprint and DOCX submission.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "regular",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,
})

PAPER_DIR = FIGURES_DIR.parent / "paper"
ASSETS_DIR = FIGURES_DIR.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Consistent colour palette across the three figures.
COLOR_MV = "#1f77b4"
COLOR_LV = "#ff7f0e"
COLOR_BASELINE = "#7f7f7f"
COLOR_COMBINED = "#d62728"


def _save_both(fig, name):
    """Save a single matplotlib figure to PDF (paper) and PNG (assets)."""
    pdf_path = PAPER_DIR / f"{name}.pdf"
    png_path = ASSETS_DIR / f"{name}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"Saved {pdf_path.name} + {png_path.name}")


# -----------------------------------------------------------------------------
# Fig 1 — MV/LV gap boxplot (baseline vs. combined)
# -----------------------------------------------------------------------------

def fig1_mvlv_gap_boxplot(experiments):
    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    bl_mv = bl[bl.voltage_level == "MV"].groupby("grid")["mae_vm"].mean().values * 1000
    bl_lv = bl[bl.voltage_level == "LV"].groupby("grid")["mae_vm"].mean().values * 1000
    e5_mv = e5[e5.voltage_level == "MV"].groupby("grid")["mae_vm"].mean().values * 1000
    e5_lv = e5[e5.voltage_level == "LV"].groupby("grid")["mae_vm"].mean().values * 1000

    _, p_bl = stats.mannwhitneyu(bl_mv, bl_lv, alternative="two-sided")
    _, p_e5 = stats.mannwhitneyu(e5_mv, e5_lv, alternative="two-sided")
    ratio_bl = bl_lv.mean() / bl_mv.mean()
    ratio_e5 = e5_lv.mean() / e5_mv.mean()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 3.6), sharey=True, constrained_layout=True
    )

    def _draw(ax, mv_vals, lv_vals, title_main, title_sub):
        bp = ax.boxplot(
            [mv_vals, lv_vals],
            tick_labels=["MV", "LV"],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=4,
                            markerfacecolor="white",
                            markeredgecolor="black", markeredgewidth=0.6),
        )
        for box, c in zip(bp["boxes"], [COLOR_MV, COLOR_LV]):
            box.set_facecolor(c)
            box.set_alpha(0.55)
            box.set_edgecolor("#333333")
            box.set_linewidth(0.8)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(0.8)
            whisker.set_color("#333333")
        for cap in bp["caps"]:
            cap.set_linewidth(0.8)
            cap.set_color("#333333")
        ax.set_title(title_main, pad=24)
        # Stats subtitle placed just below the main title, outside the data area
        ax.text(0.5, 1.02, title_sub,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=9.5, color="#444444")
        ax.grid(axis="y", alpha=0.4)

    _draw(ax1, bl_mv, bl_lv, "Baseline GraphSAGE",
          f"LV/MV = {ratio_bl:.2f}×    per-grid p = {p_bl:.2f}")
    _draw(ax2, e5_mv, e5_lv, "Combined model",
          f"LV/MV = {ratio_e5:.2f}×    per-grid p = {p_e5:.2f}")

    # Shared y-limit covers the largest whisker across both panels.
    y_top = max(bl_mv.max(), bl_lv.max(), e5_mv.max(), e5_lv.max()) * 1.10
    ax1.set_ylim(0, y_top)

    ax1.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)")

    _save_both(fig, "fig1_mvlv_gap_boxplot")


# -----------------------------------------------------------------------------
# Fig 2 — Speedup vs. accuracy Pareto frontier
# -----------------------------------------------------------------------------

def fig2_pareto_speedup_accuracy(experiments):
    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    grid_order = sorted(set(bl["grid"].unique()) & set(e5["grid"].unique()))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)

    for grid in grid_order:
        bl_g = bl[bl.grid == grid]
        e5_g = e5[e5.grid == grid]
        is_mv = grid in MV_GRIDS
        color = COLOR_MV if is_mv else COLOR_LV

        bl_x = bl_g["speedup"].mean()
        bl_y = bl_g["mae_vm"].mean() * 1000
        e5_x = e5_g["speedup"].mean()
        e5_y = e5_g["mae_vm"].mean() * 1000

        ax.scatter(bl_x, bl_y, marker="o", color=color, s=60, alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(e5_x, e5_y, marker="^", color=color, s=80, alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(
            "", xy=(e5_x, e5_y), xytext=(bl_x, bl_y),
            arrowprops=dict(arrowstyle="-|>", color="#aaaaaa",
                            alpha=0.35, lw=0.6,
                            shrinkA=5, shrinkB=5),
            zorder=1,
        )

    # Unified 2x2 legend: every entry has the actual color and shape
    # used in the plot, so readers can match legend ↔ data directly.
    handles = [
        ax.scatter([], [], marker="o", color=COLOR_MV, s=60,
                   edgecolors="black", linewidths=0.5,
                   label="MV — Baseline (d=4)"),
        ax.scatter([], [], marker="^", color=COLOR_MV, s=80,
                   edgecolors="black", linewidths=0.5,
                   label="MV — Combined"),
        ax.scatter([], [], marker="o", color=COLOR_LV, s=60,
                   edgecolors="black", linewidths=0.5,
                   label="LV — Baseline (d=4)"),
        ax.scatter([], [], marker="^", color=COLOR_LV, s=80,
                   edgecolors="black", linewidths=0.5,
                   label="LV — Combined"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2,
              framealpha=0.95, edgecolor="#cccccc", fontsize=9)

    ax.set_xlabel(r"Inference speedup vs. Newton–Raphson ($\times$)")
    ax.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_axisbelow(True)

    _save_both(fig, "fig2_pareto_speedup_accuracy")


# -----------------------------------------------------------------------------
# Fig 3 — Per-grid MAE comparison
# -----------------------------------------------------------------------------

def fig3_per_grid_comparison(experiments):
    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    grid_order = [g for g in MV_GRIDS + LV_GRIDS
                  if g in bl["grid"].unique() and g in e5["grid"].unique()]

    bl_agg = bl.groupby("grid").agg(mean=("mae_vm", "mean"),
                                     std=("mae_vm", "std")).reset_index()
    e5_agg = e5.groupby("grid").agg(mean=("mae_vm", "mean"),
                                     std=("mae_vm", "std")).reset_index()

    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    x = np.arange(len(grid_order))
    width = 0.38

    bl_vals = [bl_agg[bl_agg.grid == g]["mean"].values[0] * 1000 for g in grid_order]
    bl_errs = [bl_agg[bl_agg.grid == g]["std"].values[0] * 1000 for g in grid_order]
    e5_vals = [e5_agg[e5_agg.grid == g]["mean"].values[0] * 1000 for g in grid_order]
    e5_errs = [e5_agg[e5_agg.grid == g]["std"].values[0] * 1000 for g in grid_order]

    ax.bar(x - width/2, bl_vals, width, yerr=bl_errs, label="Baseline",
           color=COLOR_BASELINE, alpha=0.85, capsize=3, error_kw=dict(lw=0.7),
           edgecolor="#333333", linewidth=0.4)
    ax.bar(x + width/2, e5_vals, width, yerr=e5_errs, label="Combined",
           color=COLOR_COMBINED, alpha=0.85, capsize=3, error_kw=dict(lw=0.7),
           edgecolor="#333333", linewidth=0.4)

    # Use full grid names, rotated for readability
    ax.set_xticks(x)
    ax.set_xticklabels(grid_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#cccccc")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)

    # Group separator MV / LV
    n_mv = len([g for g in MV_GRIDS if g in grid_order])
    sep_x = n_mv - 0.5
    ax.axvline(x=sep_x, color="#888888", linewidth=0.8,
               linestyle="--", alpha=0.6, zorder=0)
    y_top = max(bl_vals + e5_vals) * 1.18
    ax.set_ylim(0, y_top)
    ax.text((sep_x) / 2, y_top * 0.95, "MV grids",
            ha="center", fontsize=10, color="#555555", fontweight="bold")
    ax.text(sep_x + (len(grid_order) - n_mv) / 2, y_top * 0.95, "LV grids",
            ha="center", fontsize=10, color="#555555", fontweight="bold")

    _save_both(fig, "fig3_per_grid_comparison")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    experiments = load_all_experiments()
    fig1_mvlv_gap_boxplot(experiments)
    fig2_pareto_speedup_accuracy(experiments)
    fig3_per_grid_comparison(experiments)
    print("\nAll paper figures generated.")
