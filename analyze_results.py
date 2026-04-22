"""Analyze and visualize results from all experiments (E1-E5).

Usage:
    python analyze_results.py                    # Full analysis + all figures
    python analyze_results.py --summary          # Print summary tables only
    python analyze_results.py --pick-best        # Suggest best configs for E5
    python analyze_results.py --experiment E1    # Analyze specific experiment
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from configs import (
    FIGURES_DIR, GRID_CODES, LV_GRIDS, MV_GRIDS, RESULTS_DIR,
    PHYSICS_LAMBDAS, PE_TYPES, DEEP_DEPTHS, DEEP_TECHNIQUES,
)

logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "baseline": "#757575",
    "pi_0.01": "#81D4FA",
    "pi_0.1": "#2196F3",
    "pi_1.0": "#0D47A1",
    "laplacian": "#FF9800",
    "random_walk": "#FF5722",
    "distance_from_slack": "#4CAF50",
    "residual": "#9C27B0",
    "jknet": "#E91E63",
    "dropedge": "#00BCD4",
    "virtual": "#795548",
    "combined": "#F44336",
    "MV": "#2196F3",
    "LV": "#FF9800",
}


# ── Data Loading ────────────────────────────────────────────────────────────

def load_experiment_results(subdir: str) -> pd.DataFrame:
    """Load all JSON results from a subdirectory into a DataFrame."""
    result_dir = RESULTS_DIR / subdir
    if not result_dir.exists():
        return pd.DataFrame()

    data = []
    skip_suffixes = ("_all.json", "_pilot.json", "_full.json", "analysis.json",
                     "scaling.json", "all_results.json")
    for p in result_dir.glob("*.json"):
        if any(p.name.endswith(s) for s in skip_suffixes):
            continue
        try:
            with open(p) as f:
                item = json.load(f)
            if isinstance(item, dict) and "grid" in item and "metrics" in item:
                data.append(item)
        except (json.JSONDecodeError, KeyError):
            continue

    if not data:
        # Try loading combined file
        for combined in ["e1_all.json", "e2_all.json", "e3_pilot.json",
                         "e3_full.json", "e4_all.json", "e5_all.json",
                         "baseline_all.json", "all_results.json"]:
            combined_path = result_dir / combined
            if combined_path.exists():
                with open(combined_path) as f:
                    loaded = json.load(f)
                # Handle nested lists (legacy format)
                if isinstance(loaded, list):
                    for item in loaded:
                        if isinstance(item, dict):
                            data.append(item)
                break

    if not data:
        return pd.DataFrame()

    # For baseline from first paper: filter to GraphSAGE only
    if subdir == "baseline" and data and "experiment_tag" not in data[0]:
        data = [d for d in data if d.get("model") == "GraphSAGE"]

    df = pd.DataFrame(data)
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    df["voltage_level"] = df["grid"].apply(lambda g: "MV" if g in MV_GRIDS else "LV")
    return df


def load_all_experiments() -> dict[str, pd.DataFrame]:
    """Load results from all experiment subdirectories."""
    experiments = {}
    for subdir in ["baseline", "e1_physics", "e2_pe", "e3_deep", "e4_virtual", "e5_combined"]:
        df = load_experiment_results(subdir)
        if not df.empty:
            experiments[subdir] = df
            logger.info(f"  {subdir}: {len(df)} results loaded")
        else:
            logger.info(f"  {subdir}: no results yet")
    return experiments


# ── Summary Tables ──────────────────────────────────────────────────────────

def print_summary(experiments: dict):
    """Print summary tables for all completed experiments."""

    # Baseline
    if "baseline" in experiments:
        df = experiments["baseline"]
        print("\n" + "=" * 80)
        print("BASELINE: GraphSAGE (4 layers)")
        print("=" * 80)
        agg = df.groupby("grid").agg(
            mae_vm_mean=("mae_vm", "mean"),
            mae_vm_std=("mae_vm", "std"),
            mae_va_mean=("mae_va", "mean"),
            mae_va_std=("mae_va", "std"),
            speedup=("speedup", "mean"),
        ).reset_index()
        agg["voltage_level"] = agg["grid"].apply(lambda g: "MV" if g in MV_GRIDS else "LV")
        agg = agg.sort_values(["voltage_level", "grid"])
        print(f"\n{'Grid':<15} {'MAE_Vm (×1e-3)':<18} {'MAE_Va (deg)':<18} {'Speedup':>8}")
        print("-" * 65)
        for _, r in agg.iterrows():
            print(f"{r['grid']:<15} {r['mae_vm_mean']*1000:>6.2f} ± {r['mae_vm_std']*1000:.2f}      "
                  f"{r['mae_va_mean']:>6.3f} ± {r['mae_va_std']:.3f}      {r['speedup']:>6.1f}x")

        # MV vs LV summary
        mv = agg[agg.voltage_level == "MV"]["mae_vm_mean"]
        lv = agg[agg.voltage_level == "LV"]["mae_vm_mean"]
        print(f"\n  MV avg MAE_Vm: {mv.mean()*1000:.2f}×10⁻³  |  LV avg: {lv.mean()*1000:.2f}×10⁻³  "
              f"|  LV/MV ratio: {lv.mean()/mv.mean():.2f}x harder")

    # E1: Physics Loss
    if "e1_physics" in experiments:
        df = experiments["e1_physics"]
        print("\n" + "=" * 80)
        print("E1: PHYSICS-INFORMED LOSS")
        print("=" * 80)
        # Extract lambda from tag
        df["lambda"] = df["experiment_tag"].str.extract(r"_pi([\d.]+)").astype(float)
        agg = df.groupby("lambda").agg(
            mae_vm_mean=("mae_vm", "mean"),
            mae_vm_std=("mae_vm", "std"),
            mae_va_mean=("mae_va", "mean"),
        ).reset_index()
        print(f"\n{'Lambda':<10} {'MAE_Vm (×1e-3)':<18} {'MAE_Va (deg)':<15}")
        print("-" * 45)
        for _, r in agg.iterrows():
            print(f"{r['lambda']:<10.2f} {r['mae_vm_mean']*1000:>6.2f} ± {r['mae_vm_std']*1000:.2f}      "
                  f"{r['mae_va_mean']:>6.3f}")

        # MV vs LV per lambda
        for lam in sorted(df["lambda"].unique()):
            sub = df[df["lambda"] == lam]
            mv = sub[sub.voltage_level == "MV"]["mae_vm"].mean()
            lv = sub[sub.voltage_level == "LV"]["mae_vm"].mean()
            ratio = lv / mv if mv > 0 else float("inf")
            print(f"  λ={lam:.2f}: MV={mv*1000:.2f}e-3, LV={lv*1000:.2f}e-3, LV/MV={ratio:.2f}x")

    # E2: Positional Encodings
    if "e2_pe" in experiments:
        df = experiments["e2_pe"]
        print("\n" + "=" * 80)
        print("E2: POSITIONAL ENCODINGS")
        print("=" * 80)
        df["pe_type"] = df["experiment_tag"].str.extract(r"_pe_(.+)")
        agg = df.groupby("pe_type").agg(
            mae_vm_mean=("mae_vm", "mean"),
            mae_vm_std=("mae_vm", "std"),
            mae_va_mean=("mae_va", "mean"),
        ).reset_index()
        print(f"\n{'PE Type':<22} {'MAE_Vm (×1e-3)':<18} {'MAE_Va (deg)':<15}")
        print("-" * 55)
        for _, r in agg.iterrows():
            print(f"{r['pe_type']:<22} {r['mae_vm_mean']*1000:>6.2f} ± {r['mae_vm_std']*1000:.2f}      "
                  f"{r['mae_va_mean']:>6.3f}")

        # MV/LV gap per PE
        for pe in sorted(df["pe_type"].dropna().unique()):
            sub = df[df["pe_type"] == pe]
            mv = sub[sub.voltage_level == "MV"]["mae_vm"].mean()
            lv = sub[sub.voltage_level == "LV"]["mae_vm"].mean()
            ratio = lv / mv if mv > 0 else float("inf")
            print(f"  {pe}: MV={mv*1000:.2f}e-3, LV={lv*1000:.2f}e-3, LV/MV={ratio:.2f}x")

    # E3: Deeper GNNs
    if "e3_deep" in experiments:
        df = experiments["e3_deep"]
        print("\n" + "=" * 80)
        print("E3: DEEPER GNNs")
        print("=" * 80)
        df["depth"] = df["experiment_tag"].str.extract(r"_d(\d+)").astype(int)
        df["technique"] = df["model"].map({
            "ResidualGraphSAGE": "residual",
            "JKGraphSAGE": "jknet",
            "DropEdgeGraphSAGE": "dropedge",
        })
        agg = df.groupby(["technique", "depth"]).agg(
            mae_vm_mean=("mae_vm", "mean"),
            mae_vm_std=("mae_vm", "std"),
        ).reset_index()
        pivot = agg.pivot(index="technique", columns="depth", values="mae_vm_mean")
        print("\nMAE_Vm (×1e-3) by technique × depth:")
        print("-" * 55)
        print(f"{'Technique':<15}", end="")
        for d in sorted(pivot.columns):
            print(f"{'d=' + str(d):>10}", end="")
        print()
        for tech in pivot.index:
            print(f"{tech:<15}", end="")
            for d in sorted(pivot.columns):
                val = pivot.loc[tech, d] * 1000
                print(f"{val:>10.2f}", end="")
            print()

    # E4: Virtual Node
    if "e4_virtual" in experiments:
        df = experiments["e4_virtual"]
        print("\n" + "=" * 80)
        print("E4: VIRTUAL SLACK NODE")
        print("=" * 80)
        agg = df.groupby("grid").agg(
            mae_vm_mean=("mae_vm", "mean"),
            mae_vm_std=("mae_vm", "std"),
        ).reset_index()
        agg["voltage_level"] = agg["grid"].apply(lambda g: "MV" if g in MV_GRIDS else "LV")
        mv = agg[agg.voltage_level == "MV"]["mae_vm_mean"].mean()
        lv = agg[agg.voltage_level == "LV"]["mae_vm_mean"].mean()
        print(f"\n  MV avg MAE_Vm: {mv*1000:.2f}×10⁻³  |  LV avg: {lv*1000:.2f}×10⁻³  "
              f"|  LV/MV ratio: {lv/mv:.2f}x")

    # E5: Combined
    if "e5_combined" in experiments:
        df = experiments["e5_combined"]
        print("\n" + "=" * 80)
        print("E5: COMBINED BEST")
        print("=" * 80)
        agg = df.groupby("grid").agg(
            mae_vm_mean=("mae_vm", "mean"),
            mae_vm_std=("mae_vm", "std"),
            mae_va_mean=("mae_va", "mean"),
            speedup=("speedup", "mean"),
        ).reset_index()
        agg["voltage_level"] = agg["grid"].apply(lambda g: "MV" if g in MV_GRIDS else "LV")
        agg = agg.sort_values(["voltage_level", "grid"])
        print(f"\n{'Grid':<15} {'MAE_Vm (×1e-3)':<18} {'MAE_Va (deg)':<15} {'Speedup':>8}")
        print("-" * 60)
        for _, r in agg.iterrows():
            print(f"{r['grid']:<15} {r['mae_vm_mean']*1000:>6.2f} ± {r['mae_vm_std']*1000:.2f}      "
                  f"{r['mae_va_mean']:>6.3f}             {r['speedup']:>6.1f}x")


def pick_best(experiments: dict) -> dict:
    """Analyze E1-E4 results and suggest best configuration for E5."""
    best = {}

    if "baseline" in experiments:
        bl = experiments["baseline"]
        bl_mae = bl["mae_vm"].mean()
        print(f"\nBaseline MAE_Vm: {bl_mae*1000:.3f}×10⁻³")

    if "e1_physics" in experiments:
        df = experiments["e1_physics"]
        df["lambda"] = df["experiment_tag"].str.extract(r"_pi([\d.]+)").astype(float)
        best_lam = df.groupby("lambda")["mae_vm"].mean().idxmin()
        improvement = (1 - df[df["lambda"] == best_lam]["mae_vm"].mean() / bl_mae) * 100 if "baseline" in experiments else 0
        best["lambda"] = best_lam
        print(f"E1 best λ: {best_lam} (improvement: {improvement:+.1f}%)")

    if "e2_pe" in experiments:
        df = experiments["e2_pe"]
        df["pe_type"] = df["experiment_tag"].str.extract(r"_pe_(.+)")
        best_pe = df.groupby("pe_type")["mae_vm"].mean().idxmin()
        improvement = (1 - df[df["pe_type"] == best_pe]["mae_vm"].mean() / bl_mae) * 100 if "baseline" in experiments else 0
        best["pe"] = best_pe
        print(f"E2 best PE: {best_pe} (improvement: {improvement:+.1f}%)")

    if "e3_deep" in experiments:
        df = experiments["e3_deep"]
        df["depth"] = df["experiment_tag"].str.extract(r"_d(\d+)").astype(int)
        df["technique"] = df["model"].map({
            "ResidualGraphSAGE": "residual",
            "JKGraphSAGE": "jknet",
            "DropEdgeGraphSAGE": "dropedge",
        })
        best_idx = df.groupby(["technique", "depth"])["mae_vm"].mean().idxmin()
        best["technique"] = best_idx[0]
        best["depth"] = best_idx[1]
        print(f"E3 best: {best_idx[0]} d={best_idx[1]}")

    if "e4_virtual" in experiments:
        df = experiments["e4_virtual"]
        improvement = (1 - df["mae_vm"].mean() / bl_mae) * 100 if "baseline" in experiments else 0
        best["virtual"] = improvement > 0
        print(f"E4 virtual node: {'helpful' if improvement > 0 else 'not helpful'} ({improvement:+.1f}%)")

    print(f"\nSuggested E5 command:")
    lam = best.get("lambda", 0.1)
    pe = best.get("pe", "distance_from_slack")
    tech = best.get("technique", "residual")
    depth = best.get("depth", 16)
    virt = "" if best.get("virtual", True) else " --no-virtual"
    print(f"  python run_all_experiments.py --experiment E5 "
          f"--best-lambda {lam} --best-pe {pe} "
          f"--best-technique {tech} --best-depth {depth}{virt}")

    return best


# ── Visualization ────────────────────────────────────────────────────────────

def plot_e1_lambda_sweep(experiments: dict):
    """E1: Bar chart comparing λ values with baseline."""
    if "e1_physics" not in experiments:
        return

    df = experiments["e1_physics"].copy()
    df["lambda"] = df["experiment_tag"].str.extract(r"_pi([\d.]+)").astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Overall comparison
    groups = {"baseline": None}
    groups.update({f"λ={l}": l for l in sorted(df["lambda"].unique())})

    means, stds, labels = [], [], []
    if "baseline" in experiments:
        bl = experiments["baseline"]
        means.append(bl["mae_vm"].mean() * 1000)
        stds.append(bl["mae_vm"].std() * 1000)
        labels.append("baseline")

    for label, lam in list(groups.items())[1:]:
        sub = df[df["lambda"] == lam]
        means.append(sub["mae_vm"].mean() * 1000)
        stds.append(sub["mae_vm"].std() * 1000)
        labels.append(label)

    x = np.arange(len(labels))
    colors = [COLORS.get("baseline", "#757575")] + [COLORS.get(f"pi_{l}", "#2196F3")
              for l in sorted(df["lambda"].unique())]
    ax1.bar(x, means, yerr=stds, capsize=3, color=colors[:len(x)], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax1.set_title("(a) Physics Loss: λ Sweep")
    ax1.grid(axis="y", alpha=0.3)

    # MV vs LV gap per lambda
    gap_data = []
    if "baseline" in experiments:
        bl = experiments["baseline"]
        mv = bl[bl.voltage_level == "MV"]["mae_vm"].mean()
        lv = bl[bl.voltage_level == "LV"]["mae_vm"].mean()
        gap_data.append(("baseline", lv / mv))
    for lam in sorted(df["lambda"].unique()):
        sub = df[df["lambda"] == lam]
        mv = sub[sub.voltage_level == "MV"]["mae_vm"].mean()
        lv = sub[sub.voltage_level == "LV"]["mae_vm"].mean()
        gap_data.append((f"λ={lam}", lv / mv if mv > 0 else 0))

    ax2.bar(range(len(gap_data)), [g[1] for g in gap_data],
            color=colors[:len(gap_data)], alpha=0.85)
    ax2.set_xticks(range(len(gap_data)))
    ax2.set_xticklabels([g[0] for g in gap_data])
    ax2.set_ylabel("LV/MV Error Ratio")
    ax2.set_title("(b) MV/LV Gap Reduction")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "e1_physics_loss.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_e2_pe_comparison(experiments: dict):
    """E2: Grouped bar chart of PE types per grid."""
    if "e2_pe" not in experiments:
        return

    df = experiments["e2_pe"].copy()
    df["pe_type"] = df["experiment_tag"].str.extract(r"_pe_(.+)")

    grid_order = [g for g in MV_GRIDS + LV_GRIDS if g in df["grid"].unique()]
    pe_types = sorted(df["pe_type"].dropna().unique())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(grid_order))
    n_groups = len(pe_types) + (1 if "baseline" in experiments else 0)
    width = 0.8 / n_groups
    i = 0

    if "baseline" in experiments:
        bl = experiments["baseline"]
        bl_agg = bl.groupby("grid")["mae_vm"].mean()
        vals = [bl_agg.get(g, 0) * 1000 for g in grid_order]
        ax.bar(x + i * width, vals, width, label="baseline", color=COLORS["baseline"], alpha=0.85)
        i += 1

    for pe in pe_types:
        sub = df[df["pe_type"] == pe]
        agg = sub.groupby("grid")["mae_vm"].mean()
        vals = [agg.get(g, 0) * 1000 for g in grid_order]
        ax.bar(x + i * width, vals, width, label=pe, color=COLORS.get(pe, "#999"), alpha=0.85)
        i += 1

    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(grid_order, rotation=45, ha="right")
    ax.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax.set_title("Positional Encodings: Per-Grid Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    n_mv = len([g for g in MV_GRIDS if g in grid_order])
    ax.axvline(x=n_mv - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = FIGURES_DIR / "e2_pe_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_e3_depth_heatmap(experiments: dict):
    """E3: Heatmap of technique × depth."""
    if "e3_deep" not in experiments:
        return

    df = experiments["e3_deep"].copy()
    df["depth"] = df["experiment_tag"].str.extract(r"_d(\d+)").astype(int)
    df["technique"] = df["model"].map({
        "ResidualGraphSAGE": "residual",
        "JKGraphSAGE": "jknet",
        "DropEdgeGraphSAGE": "dropedge",
    })

    pivot = df.groupby(["technique", "depth"])["mae_vm"].mean().unstack() * 1000

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"d={d}" for d in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("MAE Vm (×10⁻³ p.u.)")
    ax.set_xlabel("Depth (layers)")
    ax.set_title("Deeper GNNs: Technique × Depth")

    fig.tight_layout()
    path = FIGURES_DIR / "e3_depth_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_e5_comparison(experiments: dict):
    """E5: Combined vs baseline comparison across all grids."""
    if "e5_combined" not in experiments or "baseline" not in experiments:
        return

    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    grid_order = [g for g in MV_GRIDS + LV_GRIDS
                  if g in bl["grid"].unique() and g in e5["grid"].unique()]

    bl_agg = bl.groupby("grid").agg(mae_vm_mean=("mae_vm", "mean"),
                                     mae_vm_std=("mae_vm", "std")).reset_index()
    e5_agg = e5.groupby("grid").agg(mae_vm_mean=("mae_vm", "mean"),
                                     mae_vm_std=("mae_vm", "std")).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Per-grid comparison
    x = np.arange(len(grid_order))
    width = 0.35
    bl_vals = [bl_agg[bl_agg.grid == g]["mae_vm_mean"].values[0] * 1000 for g in grid_order]
    bl_errs = [bl_agg[bl_agg.grid == g]["mae_vm_std"].values[0] * 1000 for g in grid_order]
    e5_vals = [e5_agg[e5_agg.grid == g]["mae_vm_mean"].values[0] * 1000 for g in grid_order]
    e5_errs = [e5_agg[e5_agg.grid == g]["mae_vm_std"].values[0] * 1000 for g in grid_order]

    ax1.bar(x - width/2, bl_vals, width, yerr=bl_errs, label="Baseline GraphSAGE",
            color=COLORS["baseline"], alpha=0.85, capsize=2)
    ax1.bar(x + width/2, e5_vals, width, yerr=e5_errs, label="PI-GNN (combined)",
            color=COLORS["combined"], alpha=0.85, capsize=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(grid_order, rotation=45, ha="right")
    ax1.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax1.set_title("(a) Per-Grid Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    n_mv = len([g for g in MV_GRIDS if g in grid_order])
    ax1.axvline(x=n_mv - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)

    # Panel 2: MV/LV gap boxplot
    bl_mv = bl[bl.voltage_level == "MV"]["mae_vm"].values * 1000
    bl_lv = bl[bl.voltage_level == "LV"]["mae_vm"].values * 1000
    e5_mv = e5[e5.voltage_level == "MV"]["mae_vm"].values * 1000
    e5_lv = e5[e5.voltage_level == "LV"]["mae_vm"].values * 1000

    bp = ax2.boxplot([bl_mv, bl_lv, e5_mv, e5_lv],
                     tick_labels=["BL-MV", "BL-LV", "PI-MV", "PI-LV"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [COLORS["MV"], COLORS["LV"], COLORS["MV"], COLORS["LV"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax2.set_title("(b) MV/LV Gap: Before vs After")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "e5_combined_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_ablation_summary(experiments: dict):
    """Ablation: summary bar chart showing each component's contribution."""
    if "baseline" not in experiments:
        return

    bl_mae = experiments["baseline"]["mae_vm"].mean() * 1000
    components = [("Baseline", bl_mae, COLORS["baseline"])]

    if "e1_physics" in experiments:
        df = experiments["e1_physics"]
        df["lambda"] = df["experiment_tag"].str.extract(r"_pi([\d.]+)").astype(float)
        best_lam = df.groupby("lambda")["mae_vm"].mean().idxmin()
        best_mae = df[df["lambda"] == best_lam]["mae_vm"].mean() * 1000
        components.append((f"+ PI (λ={best_lam})", best_mae, COLORS["pi_0.1"]))

    if "e2_pe" in experiments:
        df = experiments["e2_pe"]
        df["pe_type"] = df["experiment_tag"].str.extract(r"_pe_(.+)")
        best_pe = df.groupby("pe_type")["mae_vm"].mean().idxmin()
        best_mae = df[df["pe_type"] == best_pe]["mae_vm"].mean() * 1000
        components.append((f"+ PE ({best_pe})", best_mae, COLORS.get(best_pe, "#4CAF50")))

    if "e3_deep" in experiments:
        best_mae = experiments["e3_deep"]["mae_vm"].mean() * 1000
        components.append(("+ Deep GNN", best_mae, COLORS["residual"]))

    if "e4_virtual" in experiments:
        best_mae = experiments["e4_virtual"]["mae_vm"].mean() * 1000
        components.append(("+ Virtual", best_mae, COLORS["virtual"]))

    if "e5_combined" in experiments:
        best_mae = experiments["e5_combined"]["mae_vm"].mean() * 1000
        components.append(("Combined", best_mae, COLORS["combined"]))

    if len(components) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = [c[2] for c in components]

    bars = ax.bar(range(len(components)), values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax.set_title("Ablation: Each Component's Contribution")
    ax.grid(axis="y", alpha=0.3)

    # Annotate improvement
    for i, (bar, val) in enumerate(zip(bars, values)):
        pct = (1 - val / bl_mae) * 100 if i > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}\n({pct:+.0f}%)" if i > 0 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "ablation_summary.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def statistical_tests(experiments: dict):
    """Run statistical significance tests."""
    if "baseline" not in experiments or "e5_combined" not in experiments:
        return

    bl = experiments["baseline"]
    e5 = experiments["e5_combined"]

    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    # Paired comparison per grid (aggregate across seeds)
    bl_per_grid = bl.groupby("grid")["mae_vm"].mean()
    e5_per_grid = e5.groupby("grid")["mae_vm"].mean()
    common_grids = sorted(set(bl_per_grid.index) & set(e5_per_grid.index))

    if len(common_grids) >= 5:
        bl_vals = [bl_per_grid[g] for g in common_grids]
        e5_vals = [e5_per_grid[g] for g in common_grids]

        # Wilcoxon signed-rank test (paired)
        stat, p_val = stats.wilcoxon(bl_vals, e5_vals)
        print(f"\nWilcoxon signed-rank (baseline vs combined):")
        print(f"  statistic={stat:.1f}, p={p_val:.4f}")
        print(f"  {'Significant' if p_val < 0.05 else 'Not significant'} at α=0.05")

        # Effect size (mean improvement)
        improvements = [(b - e) / b * 100 for b, e in zip(bl_vals, e5_vals)]
        print(f"  Mean improvement: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")

    # MV/LV gap test
    bl_lv = bl[bl.voltage_level == "LV"]["mae_vm"].values
    bl_mv = bl[bl.voltage_level == "MV"]["mae_vm"].values
    e5_lv = e5[e5.voltage_level == "LV"]["mae_vm"].values
    e5_mv = e5[e5.voltage_level == "MV"]["mae_vm"].values

    if len(bl_lv) > 0 and len(bl_mv) > 0:
        _, p_bl = stats.mannwhitneyu(bl_mv, bl_lv, alternative="two-sided")
        print(f"\nMann-Whitney U (MV vs LV gap):")
        print(f"  Baseline: p={p_bl:.4f}, gap ratio={bl_lv.mean()/bl_mv.mean():.2f}x")
    if len(e5_lv) > 0 and len(e5_mv) > 0:
        _, p_e5 = stats.mannwhitneyu(e5_mv, e5_lv, alternative="two-sided")
        print(f"  Combined: p={p_e5:.4f}, gap ratio={e5_lv.mean()/e5_mv.mean():.2f}x")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Paper v2 Results")
    parser.add_argument("--summary", action="store_true", help="Print summary tables only")
    parser.add_argument("--pick-best", action="store_true", help="Suggest best configs for E5")
    parser.add_argument("--experiment", choices=["E1", "E2", "E3", "E4", "E5", "all"],
                        default="all", help="Which experiment to analyze")
    args = parser.parse_args()

    logger.info("Loading experiment results...")
    experiments = load_all_experiments()

    if not experiments:
        logger.error("No experiment results found. Run experiments first.")
        return

    print_summary(experiments)

    if args.pick_best:
        pick_best(experiments)
        return

    if args.summary:
        return

    # Generate all figures
    logger.info("\nGenerating figures...")
    plot_e1_lambda_sweep(experiments)
    plot_e2_pe_comparison(experiments)
    plot_e3_depth_heatmap(experiments)
    plot_e5_comparison(experiments)
    plot_ablation_summary(experiments)
    statistical_tests(experiments)

    logger.info(f"\nAll figures saved to {FIGURES_DIR}")
    logger.info("Analysis complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
