"""Publication-quality figures for the GNN Power Flow benchmark."""

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from configs import FIGURES_DIR, GRID_CODES, LV_GRIDS, MODEL_NAMES, MODELS_DIR, MV_GRIDS, RESULTS_DIR

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


def load_results() -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    results_file = RESULTS_DIR / "all_results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
    else:
        data = []
        for p in RESULTS_DIR.glob("*.json"):
            if p.name in ("all_results.json", "e2_analysis.json", "e3_scaling.json"):
                continue
            with open(p) as f:
                data.append(json.load(f))

    if not data:
        raise FileNotFoundError("No results found. Run run_experiments.py first.")

    df = pd.DataFrame(data)
    metrics_df = pd.json_normalize(df["metrics"])
    df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
    return df


def load_graph_properties() -> pd.DataFrame:
    """Load graph properties from E2 analysis."""
    path = RESULTS_DIR / "e2_analysis.json"
    if not path.exists():
        raise FileNotFoundError("No E2 analysis found. Run run_analysis.py first.")
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data["graph_properties"])


# ── E1 Figures ─────────────────────────────────────────────────────────────────

def plot_accuracy_heatmap(df: pd.DataFrame):
    """Heatmap of MAE_Vm across grids × models."""
    pivot = df.groupby(["grid", "model"])["mae_vm"].mean().unstack()
    # Order grids: MV first, then LV
    grid_order = [g for g in MV_GRIDS + LV_GRIDS if g in pivot.index]
    model_order = [m for m in MODEL_NAMES if m in pivot.columns]
    pivot = pivot.loc[grid_order, model_order]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pivot.values * 1000, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order)
    ax.set_yticks(range(len(grid_order)))
    ax.set_yticklabels(grid_order)

    # Annotate cells
    for i in range(len(grid_order)):
        for j in range(len(model_order)):
            val = pivot.values[i, j] * 1000
            color = "white" if val > pivot.values.max() * 1000 * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE Vm (×10⁻³ p.u.)")
    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Distribution Grid")
    ax.set_title("Voltage Magnitude Prediction Accuracy")

    # Add separator line between MV and LV
    n_mv = len([g for g in MV_GRIDS if g in grid_order])
    ax.axhline(y=n_mv - 0.5, color="black", linewidth=1.5, linestyle="--")

    path = FIGURES_DIR / "e1_accuracy_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_speedup_bars(df: pd.DataFrame):
    """Bar chart of speedup (NR time / GNN time) per grid and model."""
    agg = df.groupby(["grid", "model"]).agg(
        speedup_mean=("speedup", "mean"),
        speedup_std=("speedup", "std"),
    ).reset_index()

    grid_order = [g for g in MV_GRIDS + LV_GRIDS if g in agg["grid"].unique()]
    model_order = [m for m in MODEL_NAMES if m in agg["model"].unique()]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(grid_order))
    n_models = len(model_order)
    width = 0.8 / n_models
    colors = ["#757575", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for i, model in enumerate(model_order):
        model_data = agg[agg["model"] == model].set_index("grid")
        vals = [model_data.loc[g, "speedup_mean"] if g in model_data.index else 0
                for g in grid_order]
        errs = [model_data.loc[g, "speedup_std"] if g in model_data.index else 0
                for g in grid_order]
        ax.bar(x + i * width, vals, width, yerr=errs, label=model,
               color=colors[i], alpha=0.85, capsize=2)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(grid_order, rotation=45, ha="right")
    ax.set_ylabel("Speedup (NR / GNN)")
    ax.set_title("GNN Inference Speedup over Newton-Raphson")
    ax.legend(loc="upper left")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    path = FIGURES_DIR / "e1_speedup.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── E2 Figures ─────────────────────────────────────────────────────────────────

def plot_mv_lv_boxplots(df: pd.DataFrame):
    """Box plots comparing MV vs LV prediction errors."""
    df = df.copy()
    df["voltage_level"] = df["grid"].apply(lambda g: "MV" if g in MV_GRIDS else "LV")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Vm
    mv_vm = df[df.voltage_level == "MV"]["mae_vm"].values * 1000
    lv_vm = df[df.voltage_level == "LV"]["mae_vm"].values * 1000
    bp1 = ax1.boxplot([mv_vm, lv_vm], tick_labels=["MV", "LV"], patch_artist=True,
                       widths=0.5)
    bp1["boxes"][0].set_facecolor("#2196F3")
    bp1["boxes"][1].set_facecolor("#FF9800")
    ax1.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax1.set_title("Voltage Magnitude Error")

    # Va
    mv_va = df[df.voltage_level == "MV"]["mae_va"].values
    lv_va = df[df.voltage_level == "LV"]["mae_va"].values
    bp2 = ax2.boxplot([mv_va, lv_va], tick_labels=["MV", "LV"], patch_artist=True,
                       widths=0.5)
    bp2["boxes"][0].set_facecolor("#2196F3")
    bp2["boxes"][1].set_facecolor("#FF9800")
    ax2.set_ylabel("MAE Va (degrees)")
    ax2.set_title("Voltage Angle Error")

    fig.suptitle("MV vs LV Prediction Error Distribution", y=1.02)
    fig.tight_layout()

    path = FIGURES_DIR / "e2_mv_lv_boxplots.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_rx_vs_error(df: pd.DataFrame, props: pd.DataFrame):
    """Scatter plot of average R/X ratio vs MAE_Vm."""
    agg = df.groupby("grid").agg(mae_vm=("mae_vm", "mean")).reset_index()
    merged = agg.merge(props[["grid", "avg_rx_ratio", "voltage_level"]], on="grid")

    fig, ax = plt.subplots(figsize=(5, 4))
    mv_mask = merged.voltage_level == "MV"
    ax.scatter(merged[mv_mask]["avg_rx_ratio"], merged[mv_mask]["mae_vm"] * 1000,
               c="#2196F3", s=80, label="MV", zorder=3, edgecolors="black", linewidth=0.5)
    ax.scatter(merged[~mv_mask]["avg_rx_ratio"], merged[~mv_mask]["mae_vm"] * 1000,
               c="#FF9800", s=80, label="LV", zorder=3, edgecolors="black", linewidth=0.5)

    for _, row in merged.iterrows():
        ax.annotate(row["grid"], (row["avg_rx_ratio"], row["mae_vm"] * 1000),
                    fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    ax.set_xlabel("Average R/X Ratio")
    ax.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax.set_title("Grid R/X Ratio vs Prediction Error")
    ax.legend()
    ax.grid(alpha=0.3)

    path = FIGURES_DIR / "e2_rx_vs_error.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_mv_lv_combined(df: pd.DataFrame, props: pd.DataFrame):
    """Combined MV/LV analysis: boxplots (left) + R/X scatter (right)."""
    df = df.copy()
    df["voltage_level"] = df["grid"].apply(lambda g: "MV" if g in MV_GRIDS else "LV")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Boxplots of MAE Vm
    mv_vm = df[df.voltage_level == "MV"]["mae_vm"].values * 1000
    lv_vm = df[df.voltage_level == "LV"]["mae_vm"].values * 1000
    bp = ax1.boxplot([mv_vm, lv_vm], tick_labels=["MV", "LV"], patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][1].set_facecolor("#FF9800")
    ax1.set_ylabel("MAE Vm (\u00d710\u207b\u00b3 p.u.)")
    ax1.set_title("(a) MV vs LV Error Distribution")

    # Right: R/X vs error scatter
    agg = df.groupby("grid").agg(mae_vm=("mae_vm", "mean")).reset_index()
    merged = agg.merge(props[["grid", "avg_rx_ratio", "voltage_level"]], on="grid")
    mv_mask = merged.voltage_level == "MV"
    ax2.scatter(merged[mv_mask]["avg_rx_ratio"], merged[mv_mask]["mae_vm"] * 1000,
                c="#2196F3", s=80, label="MV", zorder=3, edgecolors="black", linewidth=0.5)
    ax2.scatter(merged[~mv_mask]["avg_rx_ratio"], merged[~mv_mask]["mae_vm"] * 1000,
                c="#FF9800", s=80, label="LV", zorder=3, edgecolors="black", linewidth=0.5)
    for _, row in merged.iterrows():
        ax2.annotate(row["grid"], (row["avg_rx_ratio"], row["mae_vm"] * 1000),
                     fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                     textcoords="offset points")
    ax2.set_xlabel("Average R/X Ratio")
    ax2.set_ylabel("MAE Vm (\u00d710\u207b\u00b3 p.u.)")
    ax2.set_title("(b) R/X Ratio vs Prediction Error")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "e2_mv_lv_combined.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── E3 Figures ─────────────────────────────────────────────────────────────────

def plot_scalability(df: pd.DataFrame):
    """Scalability curves: buses vs inference time, buses vs error."""
    from data_generation import load_simbench_net

    bus_counts = {}
    for grid_name in GRID_CODES:
        net = load_simbench_net(grid_name)
        bus_counts[grid_name] = len(net.bus)
    df = df.copy()
    df["num_buses"] = df["grid"].map(bus_counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"MLP": "#757575", "GCN": "#2196F3", "GAT": "#FF9800", "GraphSAGE": "#4CAF50", "MPNN": "#9C27B0"}

    # Panel 1: Time vs buses
    for model_name in MODEL_NAMES:
        model_data = df[df["model"] == model_name].groupby("grid").agg(
            num_buses=("num_buses", "first"),
            gnn_time=("gnn_time_ms", "mean"),
        ).sort_values("num_buses")
        ax1.plot(model_data["num_buses"], model_data["gnn_time"],
                 "o-", color=colors.get(model_name, "gray"), label=model_name,
                 markersize=4)

    # NR time
    nr_data = df.groupby("grid").agg(
        num_buses=("num_buses", "first"),
        nr_time=("nr_time_ms", "mean"),
    ).sort_values("num_buses")
    ax1.plot(nr_data["num_buses"], nr_data["nr_time"],
             "s--", color="red", label="NR", markersize=5)

    ax1.set_xlabel("Number of Buses")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Inference Time Scaling")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2: Error vs buses
    for model_name in MODEL_NAMES:
        model_data = df[df["model"] == model_name].groupby("grid").agg(
            num_buses=("num_buses", "first"),
            mae_vm=("mae_vm", "mean"),
        ).sort_values("num_buses")
        ax2.plot(model_data["num_buses"], model_data["mae_vm"] * 1000,
                 "o-", color=colors.get(model_name, "gray"), label=model_name,
                 markersize=4)

    ax2.set_xlabel("Number of Buses")
    ax2.set_ylabel("MAE Vm (×10⁻³ p.u.)")
    ax2.set_title("Prediction Error Scaling")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "e3_scalability.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── New Figures ────────────────────────────────────────────────────────────────

def plot_grid_topology():
    """Visualize 2 example grid topologies (1 MV, 1 LV) as NetworkX graphs."""
    import networkx as nx
    from data_generation import load_simbench_net, build_graph_topology

    grids = [("MV_urban", "MV_urban (144 buses, meshed)"),
             ("LV_rural3", "LV_rural3 (129 buses, radial)")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (grid_name, title) in zip(axes, grids):
        net = load_simbench_net(grid_name)
        topo = build_graph_topology(net)

        G = nx.Graph()
        G.add_nodes_from(range(topo.num_nodes))
        edge_index = topo.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            s, d = edge_index[0, i], edge_index[1, i]
            if s < d:
                G.add_edge(s, d)

        pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(topo.num_nodes))

        # Color nodes by type
        ext_grid_idx = set()
        bus_to_idx = {b: i for i, b in enumerate(topo.bus_indices)}
        for bus in topo.ext_grid_buses:
            if bus in bus_to_idx:
                ext_grid_idx.add(bus_to_idx[bus])

        slack_nodes = [i for i in range(topo.num_nodes) if i in ext_grid_idx]
        pq_nodes = [i for i in range(topo.num_nodes) if i not in ext_grid_idx]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.5, edge_color="#999999")
        # PQ buses: circles (default)
        nx.draw_networkx_nodes(G, pos, nodelist=pq_nodes, ax=ax, node_size=15,
                               node_color="#42A5F5", node_shape="o",
                               edgecolors="black", linewidths=0.3)
        # Slack buses: squares, larger
        nx.draw_networkx_nodes(G, pos, nodelist=slack_nodes, ax=ax, node_size=60,
                               node_color="#E53935", node_shape="s",
                               edgecolors="black", linewidths=0.5)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E53935',
               markeredgecolor='black', markersize=8, label='Slack bus'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#42A5F5',
               markeredgecolor='black', markersize=8, label='PQ bus'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    path = FIGURES_DIR / "grid_topology.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_per_node_error():
    """Per-node error heatmap on graph topology for one example grid."""
    import networkx as nx
    import torch
    from data_generation import load_simbench_net, build_graph_topology, build_dataset
    from models import create_model

    grid_name = "LV_rural3"
    model_name = "GraphSAGE"
    seed = 42

    # Load data and model
    train_data, val_data, test_data, normalizer = build_dataset(grid_name, seed=seed)
    net = load_simbench_net(grid_name)
    topo = build_graph_topology(net)

    kwargs = {}
    model = create_model(model_name, **kwargs)
    ckpt_path = MODELS_DIR / f"{grid_name}_{model_name}_s{seed}.pt"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    model.eval()

    # Compute per-node MAE across test set
    all_errors = []
    with torch.no_grad():
        for data in test_data:
            pred = model(data.x, data.edge_index, data.edge_attr, None)
            pred_denorm = normalizer.denormalize_y(pred)
            true_denorm = normalizer.denormalize_y(data.y)
            err = torch.abs(pred_denorm[:, 0] - true_denorm[:, 0])  # Vm error
            all_errors.append(err.numpy())

    per_node_mae = np.mean(all_errors, axis=0) * 1000  # ×10⁻³ p.u.

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(topo.num_nodes))
    edge_index = topo.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        s, d = edge_index[0, i], edge_index[1, i]
        if s < d:
            G.add_edge(s, d)

    pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(topo.num_nodes))

    fig, ax = plt.subplots(figsize=(7, 6))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5, edge_color="#999999")

    norm = Normalize(vmin=per_node_mae.min(), vmax=per_node_mae.max())
    cmap = plt.cm.YlOrRd
    node_colors = [cmap(norm(v)) for v in per_node_mae]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=40, node_color=node_colors,
                           edgecolors="black", linewidths=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("MAE Vm (×10⁻³ p.u.)")
    ax.set_title(f"Per-node prediction error: {grid_name} ({model_name})", fontsize=11)
    ax.axis("off")

    path = FIGURES_DIR / "per_node_error.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_learning_curves():
    """Learning curves (train/val loss) for all models on one grid.

    Uses loss data from result JSONs if available (train_losses/val_losses keys).
    Falls back to re-training if no loss data is found.
    """
    grid_name = "MV_rural"
    seed = 42
    colors = {"MLP": "#757575", "GCN": "#2196F3", "GAT": "#FF9800",
              "GraphSAGE": "#4CAF50", "MPNN": "#9C27B0"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    found_any = False

    for model_name in MODEL_NAMES:
        result_path = RESULTS_DIR / f"{grid_name}_{model_name}_s{seed}.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            result = json.load(f)

        train_losses = result.get("train_losses")
        val_losses = result.get("val_losses")
        if not train_losses or not val_losses:
            continue

        found_any = True
        epochs = range(1, len(train_losses) + 1)
        c = colors.get(model_name, "gray")
        ax1.plot(epochs, train_losses, color=c, label=model_name, linewidth=1.2)
        ax2.plot(epochs, val_losses, color=c, label=model_name, linewidth=1.2)

    if not found_any:
        logger.warning("No loss data found in result JSONs. "
                        "Re-run experiments with updated run_experiments.py.")
        plt.close(fig)
        return

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_yscale("log")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Validation Loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle(f"Learning Curves ({grid_name}, seed={seed})", y=1.02)
    fig.tight_layout()

    path = FIGURES_DIR / "learning_curves.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        df = load_results()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # E1 figures
    plot_accuracy_heatmap(df)
    plot_speedup_bars(df)

    # E2 figures
    try:
        props = load_graph_properties()
        plot_mv_lv_boxplots(df)
        plot_rx_vs_error(df, props)
        plot_mv_lv_combined(df, props)
    except FileNotFoundError:
        logger.warning("E2 analysis not found. Skipping E2 figures.")

    # E3 figure
    plot_scalability(df)

    # New figures
    plot_grid_topology()
    plot_per_node_error()
    plot_learning_curves()

    logger.info(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
