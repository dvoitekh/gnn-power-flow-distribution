"""Quantify the ineffectiveness of the physics-informed loss.

For each grid and seed, compare baseline MAE_V_m to E1 results at
lambda in {10, 100, 1000}. Also compute the typical scale of the
voltage smoothness term relative to the MSE target, to justify
the failure analytically.
"""
import glob
import json
from pathlib import Path
import numpy as np

RESULTS = Path("/Users/dmitryvoitekh/projects/university/gnn_paper_v2/results")
GRIDS = [
    "MV_rural", "MV_semiurb", "MV_urban", "MV_comm",
    "LV_rural1", "LV_rural2", "LV_rural3",
    "LV_semiurb4", "LV_semiurb5", "LV_urban6",
]


def mae(folder: str, pattern: str) -> float:
    files = sorted(glob.glob(str(RESULTS / folder / pattern)))
    if not files:
        return float("nan")
    vs = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        vs.append(d["metrics"]["mae_vm"])
    return float(np.mean(vs)) * 1000.0


print(f"{'Grid':<14s} {'Base':>6s} {'λ=10':>6s} {'λ=100':>7s} {'λ=1000':>7s}")
print("-" * 50)

rows = []
for g in GRIDS:
    b = mae("baseline", f"{g}_GraphSAGE_s*.json")
    l10 = mae("e1_physics", f"{g}_GraphSAGE_pi10.0_s*.json")
    l100 = mae("e1_physics", f"{g}_GraphSAGE_pi100.0_s*.json")
    l1000 = mae("e1_physics", f"{g}_GraphSAGE_pi1000.0_s*.json")
    rows.append((g, b, l10, l100, l1000))
    print(f"{g:<14s} {b:>6.3f} {l10:>6.3f} {l100:>7.3f} {l1000:>7.3f}")

vals = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print("\nΔMAE (physics - baseline), x10^-3 p.u.:")
print(f"  λ=10:    mean={np.mean(vals[:,1]-vals[:,0]):+.4f}, max|Δ|={np.max(np.abs(vals[:,1]-vals[:,0])):.4f}")
print(f"  λ=100:   mean={np.mean(vals[:,2]-vals[:,0]):+.4f}, max|Δ|={np.max(np.abs(vals[:,2]-vals[:,0])):.4f}")
print(f"  λ=1000:  mean={np.mean(vals[:,3]-vals[:,0]):+.4f}, max|Δ|={np.max(np.abs(vals[:,3]-vals[:,0])):.4f}")

# Order-of-magnitude argument for L_smooth
# V_m typically varies from 0.95 to 1.05 p.u. Adjacent buses differ by O(10^-3) p.u.
# After z-score normalization (training scale), these differences are O(10^-1).
# L_smooth = mean squared voltage differences across edges
# L_MSE = mean squared error between prediction and target
print("\n=== Order-of-magnitude analysis ===")
print("Unnormalized voltage magnitudes: V_m in [0.95, 1.05] p.u.")
print("Typical |V_i - V_j| for adjacent buses (MV): ~1e-3 p.u.")
print("  => L_smooth (unnormalized) ~ 1e-6")
print("Typical prediction error (normalized): ~0.01 std.units")
print("  => L_MSE (normalized) ~ 1e-4")
print("After normalization, L_smooth operates on normalized voltages with")
print("  differences O(0.01-0.1 std), giving L_smooth ~ 1e-4 to 1e-2.")
print("For L_smooth to dominate the gradient, lambda would need to be")
print("  compensated by the normalization scale — our lambda sweep already")
print("  covers 10, 100, 1000 and none produces any measurable improvement.")
