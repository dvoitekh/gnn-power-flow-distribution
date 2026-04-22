"""Generate additional figures for the GitHub README."""
import glob
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS = Path("/Users/dmitryvoitekh/projects/university/gnn_paper_v2/results")
ASSETS = Path("/Users/dmitryvoitekh/projects/university/gnn_paper_v2/assets")
ASSETS.mkdir(exist_ok=True)

GRID_PROPS = {
    "MV_rural":    {"diameter": 22, "type": "MV"},
    "MV_semiurb":  {"diameter": 14, "type": "MV"},
    "MV_urban":    {"diameter": 12, "type": "MV"},
    "MV_comm":     {"diameter": 20, "type": "MV"},
    "LV_rural1":   {"diameter": 13, "type": "LV"},
    "LV_rural2":   {"diameter": 56, "type": "LV"},
    "LV_rural3":   {"diameter": 38, "type": "LV"},
    "LV_semiurb4": {"diameter": 28, "type": "LV"},
    "LV_semiurb5": {"diameter": 32, "type": "LV"},
    "LV_urban6":   {"diameter": 18, "type": "LV"},
}


def mean_mae(pattern: str) -> float:
    files = sorted(glob.glob(str(RESULTS / pattern)))
    return float(np.mean([json.load(open(f))["metrics"]["mae_vm"] for f in files])) * 1000


# Collect data
grids = list(GRID_PROPS)
diameters = [GRID_PROPS[g]["diameter"] for g in grids]
baseline_mae = [mean_mae(f"baseline/{g}_GraphSAGE_s*.json") for g in grids]
combined_mae = [mean_mae(f"e5_combined/{g}_ResidualGraphSAGE_combined_s*.json") for g in grids]
types = [GRID_PROPS[g]["type"] for g in grids]


# === Figure: Diameter vs MAE with correlation ===
fig, ax = plt.subplots(figsize=(7, 5))
mv_mask = np.array(types) == "MV"
lv_mask = np.array(types) == "LV"
diam = np.array(diameters)
base = np.array(baseline_mae)
comb = np.array(combined_mae)

ax.scatter(diam[mv_mask], base[mv_mask], s=120, c="#2E86AB", marker="o",
           edgecolors="black", linewidth=1.2, label="Baseline (MV)", zorder=3)
ax.scatter(diam[lv_mask], base[lv_mask], s=120, c="#E63946", marker="o",
           edgecolors="black", linewidth=1.2, label="Baseline (LV)", zorder=3)
ax.scatter(diam[mv_mask], comb[mv_mask], s=120, c="#2E86AB", marker="^",
           edgecolors="black", linewidth=1.2, label="Combined (MV)", zorder=3)
ax.scatter(diam[lv_mask], comb[lv_mask], s=120, c="#E63946", marker="^",
           edgecolors="black", linewidth=1.2, label="Combined (LV)", zorder=3)

# Trend line for baseline
z = np.polyfit(diameters, baseline_mae, 1)
xs = np.linspace(min(diameters), max(diameters), 100)
ax.plot(xs, np.polyval(z, xs), "--", color="gray", alpha=0.6, zorder=1,
        label=f"Baseline trend (ρ={stats.spearmanr(diameters, baseline_mae).statistic:+.2f})")

ax.set_xlabel("Graph diameter (hops)", fontsize=12)
ax.set_ylabel(r"MAE $V_m$ ($\times 10^{-3}$ p.u.)", fontsize=12)
ax.set_title("Baseline error tracks grid diameter; combined model flattens it",
             fontsize=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / "fig_diameter_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_diameter_correlation.png")


# === Figure: Virtual slack edges concept illustration ===
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

def draw_feeder(ax, title, virtual=False):
    # Simple 7-node radial feeder
    xs = [0, 1, 2, 3, 4, 5, 6]
    ys = [0, 0, 0, 0, 0, 0, 0]
    ax.plot(xs, ys, "-", color="black", linewidth=2, zorder=2)
    # Slack node (star)
    ax.scatter([0], [0], s=400, c="#FFD700", marker="*", edgecolors="black",
               linewidth=1.5, zorder=5)
    ax.annotate("slack", (0, 0), xytext=(0, 0.35), ha="center", fontsize=10,
                fontweight="bold")
    # Other nodes
    ax.scatter(xs[1:], ys[1:], s=180, c="#87CEEB", edgecolors="black",
               linewidth=1.2, zorder=4)
    for i, (x, y) in enumerate(zip(xs[1:], ys[1:]), 1):
        ax.annotate(f"{i}", (x, y), ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=6)

    if virtual:
        # Add curved virtual edges from slack to each other node
        for i in range(1, 7):
            ax.annotate("", xy=(xs[i], ys[i] - 0.015),
                        xytext=(xs[0], ys[0] - 0.015),
                        arrowprops=dict(arrowstyle="-", color="#E63946",
                                        linewidth=1.5, alpha=0.6,
                                        connectionstyle=f"arc3,rad=-{0.25 + 0.06*i}"),
                        zorder=1)

    ax.set_xlim(-0.8, 7)
    ax.set_ylim(-2.0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Diameter annotation
    diam = 2 if virtual else 6
    ax.annotate(f"diameter = {diam}", (3, -1.5), ha="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray"))


draw_feeder(axes[0], "Original radial feeder\n(receptive field bottleneck)",
            virtual=False)
draw_feeder(axes[1], "Augmented graph with virtual slack edges\n(2-hop diameter)",
            virtual=True)
plt.tight_layout()
plt.savefig(ASSETS / "fig_virtual_slack_concept.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_virtual_slack_concept.png")
