"""Compute structural-property correlations for the MV/LV gap analysis.

Per-grid data:
  - diameter (hops)
  - R/X ratio
  - bridge fraction
  - baseline MAE V_m (averaged over seeds)
  - E4 virtual-only MAE V_m
  - E5 combined MAE V_m
  - absolute improvement, relative improvement

Computes Spearman and Pearson correlations of structural properties against
baseline MAE and improvement, to provide evidence for the "diameter causes gap"
claim and to show generalization across grids.
"""
import json
import glob
from pathlib import Path
from scipy import stats
import numpy as np

RESULTS_DIR = Path("/Users/dmitryvoitekh/projects/university/gnn_paper_v2/results")

# Structural properties from SimBench (as reported in paper Table 1)
GRID_PROPS = {
    "MV_rural":    {"buses": 97,  "diameter": 22, "rx": 1.02, "bridge": 0.96},
    "MV_semiurb":  {"buses": 116, "diameter": 14, "rx": 0.98, "bridge": 0.89},
    "MV_urban":    {"buses": 144, "diameter": 12, "rx": 0.85, "bridge": 0.88},
    "MV_comm":     {"buses": 89,  "diameter": 20, "rx": 1.05, "bridge": 0.97},
    "LV_rural1":   {"buses": 15,  "diameter": 13, "rx": 5.21, "bridge": 1.00},
    "LV_rural2":   {"buses": 96,  "diameter": 56, "rx": 4.87, "bridge": 1.00},
    "LV_rural3":   {"buses": 60,  "diameter": 38, "rx": 4.92, "bridge": 1.00},
    "LV_semiurb4": {"buses": 44,  "diameter": 28, "rx": 3.45, "bridge": 1.00},
    "LV_semiurb5": {"buses": 55,  "diameter": 32, "rx": 3.68, "bridge": 1.00},
    "LV_urban6":   {"buses": 40,  "diameter": 18, "rx": 2.95, "bridge": 0.95},
}
GRIDS = list(GRID_PROPS)


def load_mae(folder: str, suffix: str) -> dict[str, float]:
    """Return mean MAE_V_m per grid, averaged over 3 seeds."""
    out = {}
    for grid in GRIDS:
        pattern = str(RESULTS_DIR / folder / f"{grid}_{suffix}_s*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"WARNING: no files matched {pattern}")
            continue
        vals = []
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            vals.append(d["metrics"]["mae_vm"])
        out[grid] = float(np.mean(vals)) * 1000.0  # p.u. -> x10^-3 p.u.
    return out


def report_corr(name: str, x: list[float], y: list[float]) -> None:
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    print(f"  {name:<30s} Spearman rho = {sp.statistic:+.3f} (p={sp.pvalue:.4f}), "
          f"Pearson r = {pe.statistic:+.3f} (p={pe.pvalue:.4f})")


def main() -> None:
    baseline = load_mae("baseline", "GraphSAGE")
    virtual = load_mae("e4_virtual", "GraphSAGE_virtual")
    combined = load_mae("e5_combined", "ResidualGraphSAGE_combined")

    print(f"\n{'Grid':<14s} {'Diam':>5s} {'R/X':>6s} {'Bridge':>7s} "
          f"{'Base':>6s} {'Virt':>6s} {'Comb':>6s} "
          f"{'Δ_virt':>7s} {'Δ_comb':>7s} {'%comb':>6s}")
    print("-" * 90)

    diameters, rxs, bridges = [], [], []
    base_mae, virt_mae, comb_mae = [], [], []
    abs_imp_virt, abs_imp_comb, rel_imp_comb = [], [], []

    for g in GRIDS:
        p = GRID_PROPS[g]
        b, v, c = baseline[g], virtual[g], combined[g]
        diameters.append(p["diameter"])
        rxs.append(p["rx"])
        bridges.append(p["bridge"])
        base_mae.append(b)
        virt_mae.append(v)
        comb_mae.append(c)
        abs_imp_virt.append(b - v)
        abs_imp_comb.append(b - c)
        rel_imp_comb.append(100 * (b - c) / b)
        print(f"{g:<14s} {p['diameter']:>5d} {p['rx']:>6.2f} {p['bridge']:>7.2f} "
              f"{b:>6.2f} {v:>6.2f} {c:>6.2f} "
              f"{b-v:>7.2f} {b-c:>7.2f} {100*(b-c)/b:>5.1f}%")

    print("\n\n=== CORRELATIONS: Structural property vs Baseline MAE ===")
    report_corr("Diameter vs Baseline MAE", diameters, base_mae)
    report_corr("R/X ratio vs Baseline MAE", rxs, base_mae)
    report_corr("Bridge fraction vs Baseline MAE", bridges, base_mae)

    print("\n=== CORRELATIONS: Structural property vs Absolute improvement (virtual slack) ===")
    report_corr("Diameter vs Δ_virtual", diameters, abs_imp_virt)
    report_corr("R/X ratio vs Δ_virtual", rxs, abs_imp_virt)
    report_corr("Bridge fraction vs Δ_virtual", bridges, abs_imp_virt)

    print("\n=== CORRELATIONS: Structural property vs Absolute improvement (combined) ===")
    report_corr("Diameter vs Δ_combined", diameters, abs_imp_comb)
    report_corr("R/X ratio vs Δ_combined", rxs, abs_imp_comb)
    report_corr("Bridge fraction vs Δ_combined", bridges, abs_imp_comb)

    print("\n=== CORRELATIONS: Structural property vs Relative improvement (combined) ===")
    report_corr("Diameter vs %improv", diameters, rel_imp_comb)
    report_corr("R/X ratio vs %improv", rxs, rel_imp_comb)
    report_corr("Bridge fraction vs %improv", bridges, rel_imp_comb)

    print("\n=== MV vs LV summary (per-grid, n=4 vs n=6) ===")
    mv_base = [base_mae[i] for i, g in enumerate(GRIDS) if g.startswith("MV")]
    lv_base = [base_mae[i] for i, g in enumerate(GRIDS) if g.startswith("LV")]
    mv_comb = [comb_mae[i] for i, g in enumerate(GRIDS) if g.startswith("MV")]
    lv_comb = [comb_mae[i] for i, g in enumerate(GRIDS) if g.startswith("LV")]
    print(f"Baseline:  MV avg={np.mean(mv_base):.3f}, LV avg={np.mean(lv_base):.3f}, "
          f"ratio={np.mean(lv_base)/np.mean(mv_base):.2f}")
    print(f"Combined:  MV avg={np.mean(mv_comb):.3f}, LV avg={np.mean(lv_comb):.3f}, "
          f"ratio={np.mean(lv_comb)/np.mean(mv_comb):.2f}")
    u_b = stats.mannwhitneyu(mv_base, lv_base, alternative="two-sided")
    u_c = stats.mannwhitneyu(mv_comb, lv_comb, alternative="two-sided")
    print(f"Mann-Whitney U, Baseline:  U={u_b.statistic:.1f}, p={u_b.pvalue:.4f}")
    print(f"Mann-Whitney U, Combined:  U={u_c.statistic:.1f}, p={u_c.pvalue:.4f}")

    print("\n=== MV vs LV summary (per-seed, n=12 vs n=18) ===")

    def per_seed(folder: str, suffix: str, prefix: str) -> list[float]:
        out = []
        for g in GRIDS:
            if not g.startswith(prefix):
                continue
            for fp in sorted(glob.glob(str(RESULTS_DIR / folder / f"{g}_{suffix}_s*.json"))):
                with open(fp) as f:
                    d = json.load(f)
                out.append(d["metrics"]["mae_vm"] * 1000.0)
        return out

    mv_base_s = per_seed("baseline", "GraphSAGE", "MV")
    lv_base_s = per_seed("baseline", "GraphSAGE", "LV")
    mv_comb_s = per_seed("e5_combined", "ResidualGraphSAGE_combined", "MV")
    lv_comb_s = per_seed("e5_combined", "ResidualGraphSAGE_combined", "LV")
    u_b2 = stats.mannwhitneyu(mv_base_s, lv_base_s, alternative="two-sided")
    u_c2 = stats.mannwhitneyu(mv_comb_s, lv_comb_s, alternative="two-sided")
    print(f"Baseline:  n_MV={len(mv_base_s)}, n_LV={len(lv_base_s)}")
    print(f"Mann-Whitney U, Baseline (per-seed):  U={u_b2.statistic:.1f}, p={u_b2.pvalue:.4f}")
    print(f"Mann-Whitney U, Combined (per-seed):  U={u_c2.statistic:.1f}, p={u_c2.pvalue:.4f}")

    print("\n=== Wilcoxon signed-rank: paired per-grid improvement ===")
    w = stats.wilcoxon(base_mae, comb_mae)
    print(f"Wilcoxon (baseline vs combined): W={w.statistic:.1f}, p={w.pvalue:.4f}")
    mean_imp = np.mean(rel_imp_comb)
    std_imp = np.std(rel_imp_comb, ddof=1)
    print(f"Mean relative improvement: {mean_imp:.1f}% ± {std_imp:.1f}%")


if __name__ == "__main__":
    main()
