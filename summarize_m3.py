"""Summarize M3 ablation results once complete.

Prints the row to add to Table 2 of the paper and computes per-grid
statistics for LV vs MV comparison.
"""

import json
import glob
import os
from collections import defaultdict
from scipy.stats import mannwhitneyu, wilcoxon


def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    g = sum(1 for a in x for b in y if a > b)
    l = sum(1 for a in x for b in y if a < b)
    return (g - l) / (n1 * n2)


def main():
    rdir = "/Users/dmitryvoitekh/projects/university/gnn_paper_v2/results/m3_no_virtual"
    files = [f for f in glob.glob(os.path.join(rdir, "*.json"))
             if not f.endswith("m3_all.json")]

    by_grid = defaultdict(list)
    for f in sorted(files):
        with open(f) as fp:
            d = json.load(fp)
        by_grid[d["grid"]].append(d["metrics"]["mae_vm"])

    print(f"M3 experiments completed: {len(files)}/30")
    print(f"Grids with all 3 seeds: "
          f"{sum(1 for g in by_grid if len(by_grid[g]) == 3)}/10")

    mv_keys = [k for k in by_grid if k.startswith("MV_")]
    lv_keys = [k for k in by_grid if k.startswith("LV_")]

    # Per-grid MAE (mean across seeds) in 1e-3 p.u.
    mv_per_grid = [sum(by_grid[g]) / len(by_grid[g]) * 1000 for g in mv_keys
                   if len(by_grid[g]) == 3]
    lv_per_grid = [sum(by_grid[g]) / len(by_grid[g]) * 1000 for g in lv_keys
                   if len(by_grid[g]) == 3]

    all_mae = mv_per_grid + lv_per_grid
    overall_mean = sum(all_mae) / len(all_mae) if all_mae else 0.0
    overall_std = (sum((x - overall_mean) ** 2 for x in all_mae) / len(all_mae))**0.5 \
        if all_mae else 0.0

    print()
    print(f"{'Grid':<14s} {'MAE x1e-3':>10s}")
    for g in sorted(by_grid):
        vals = by_grid[g]
        m = sum(vals) / len(vals) * 1000
        mark = " (incomplete)" if len(vals) < 3 else ""
        print(f"{g:<14s} {m:>10.2f}{mark}")

    if mv_per_grid and lv_per_grid:
        u, p = mannwhitneyu(lv_per_grid, mv_per_grid, alternative="two-sided")
        d = cliffs_delta(lv_per_grid, mv_per_grid)
        mv_mean = sum(mv_per_grid) / len(mv_per_grid)
        lv_mean = sum(lv_per_grid) / len(lv_per_grid)
        ratio = lv_mean / mv_mean
        print()
        print(f"MV mean: {mv_mean:.2f}  LV mean: {lv_mean:.2f}  "
              f"ratio: {ratio:.2f}x")
        print(f"Per-grid Mann-Whitney: U={u}, p={p:.3f}; Cliff's delta={d:.3f}")

        # Improvement over baseline (baseline overall mean = 0.88)
        baseline_mean = 0.88
        improv = (baseline_mean - overall_mean) / baseline_mean * 100
        print()
        print(f"Overall MAE: {overall_mean:.2f} ± {overall_std:.2f} "
              f"(x1e-3 p.u.)")
        print(f"Improvement over baseline (0.88): {improv:.0f}%")
        print()
        print("=== Table 2 row for paper.tex / generate_eie_docx.py ===")
        print(f'"Residual d=8 + RW-PE (no virtual)", '
              f'"{overall_mean:.2f} ± {overall_std:.2f}", '
              f'"{ratio:.2f}×", "{improv:.0f}"')


if __name__ == "__main__":
    main()
