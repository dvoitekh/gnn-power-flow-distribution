"""E2: MV vs LV gap analysis. E3: Scalability analysis."""

import json
import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from configs import GRID_CODES, LV_GRIDS, MV_GRIDS, RESULTS_DIR
from data_generation import build_graph_topology, load_simbench_net

logger = logging.getLogger(__name__)


# ── E2: Graph Properties & MV vs LV Gap ───────────────────────────────────────

def compute_graph_properties(grid_name: str) -> dict:
    """Compute structural properties of a grid using NetworkX."""
    net = load_simbench_net(grid_name)
    topo = build_graph_topology(net)

    # Build undirected NetworkX graph
    G = nx.Graph()
    edge_index = topo.edge_index.numpy()
    edge_attr = topo.edge_attr.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if not G.has_edge(src, dst):
            G.add_edge(src, dst, r=edge_attr[i, 0], x=edge_attr[i, 1])

    num_buses = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = np.mean([d for _, d in G.degree()])
    clustering = nx.average_clustering(G)

    # Diameter (of largest connected component)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        diameter = nx.diameter(G.subgraph(largest_cc))

    # R/X ratio statistics
    rx_ratios = []
    for _, _, data in G.edges(data=True):
        if data["x"] > 1e-10:
            rx_ratios.append(data["r"] / data["x"])
    avg_rx_ratio = np.mean(rx_ratios) if rx_ratios else 0

    # Bridges (edges whose removal disconnects the graph)
    bridges = list(nx.bridges(G))
    fraction_bridges = len(bridges) / num_edges if num_edges > 0 else 0

    return {
        "grid": grid_name,
        "num_buses": num_buses,
        "num_edges": num_edges,
        "avg_degree": float(avg_degree),
        "clustering_coeff": float(clustering),
        "diameter": int(diameter),
        "avg_rx_ratio": float(avg_rx_ratio),
        "fraction_bridges": float(fraction_bridges),
        "voltage_level": "MV" if grid_name in MV_GRIDS else "LV",
    }


def load_all_results() -> list[dict]:
    """Load all experiment results from JSON files."""
    results_file = RESULTS_DIR / "all_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    # Fallback: load individual files
    results = []
    for p in RESULTS_DIR.glob("*.json"):
        if p.name == "all_results.json":
            continue
        with open(p) as f:
            results.append(json.load(f))
    return results


def run_e2_analysis():
    """E2: MV vs LV gap analysis with statistical tests."""
    logger.info("Running E2: MV vs LV gap analysis")

    # Compute graph properties for all grids
    props = []
    for grid_name in GRID_CODES:
        p = compute_graph_properties(grid_name)
        props.append(p)
        logger.info(f"  {grid_name}: buses={p['num_buses']}, "
                     f"avg_degree={p['avg_degree']:.2f}, "
                     f"R/X={p['avg_rx_ratio']:.2f}, "
                     f"bridges={p['fraction_bridges']:.2f}")

    props_df = pd.DataFrame(props)

    # Load experiment results and merge
    results = load_all_results()
    if not results:
        logger.warning("No experiment results found. Run run_experiments.py first.")
        return props_df, None

    results_df = pd.DataFrame(results)
    # Flatten metrics
    metrics_df = pd.json_normalize(results_df["metrics"])
    results_df = pd.concat([results_df.drop(columns=["metrics"]), metrics_df], axis=1)

    # Aggregate across seeds (mean ± std)
    agg = results_df.groupby(["grid", "model"]).agg(
        mae_vm_mean=("mae_vm", "mean"),
        mae_vm_std=("mae_vm", "std"),
        mae_va_mean=("mae_va", "mean"),
        mae_va_std=("mae_va", "std"),
    ).reset_index()

    # Add voltage level
    agg["voltage_level"] = agg["grid"].apply(
        lambda g: "MV" if g in MV_GRIDS else "LV"
    )

    # Mann-Whitney U test: MV vs LV MAE_Vm
    mv_errors = agg[agg.voltage_level == "MV"]["mae_vm_mean"].values
    lv_errors = agg[agg.voltage_level == "LV"]["mae_vm_mean"].values

    if len(mv_errors) > 0 and len(lv_errors) > 0:
        u_stat, p_value = stats.mannwhitneyu(mv_errors, lv_errors, alternative="two-sided")
        logger.info(f"\nMann-Whitney U test (MV vs LV MAE_Vm):")
        logger.info(f"  U={u_stat:.1f}, p={p_value:.4f}")
        logger.info(f"  MV mean MAE_Vm: {mv_errors.mean():.6f}")
        logger.info(f"  LV mean MAE_Vm: {lv_errors.mean():.6f}")

    # Spearman correlations: graph property vs GNN error
    merged = agg.merge(props_df, on="grid")
    property_cols = ["avg_degree", "clustering_coeff", "diameter",
                     "avg_rx_ratio", "fraction_bridges", "num_buses"]
    logger.info("\nSpearman correlations (property vs mae_vm_mean):")
    correlations = {}
    for col in property_cols:
        rho, p = stats.spearmanr(merged[col], merged["mae_vm_mean"])
        correlations[col] = {"rho": float(rho), "p": float(p)}
        logger.info(f"  {col:<20}: rho={rho:+.3f}, p={p:.4f}")

    # Save analysis results
    analysis = {
        "graph_properties": props,
        "correlations": correlations,
    }
    if len(mv_errors) > 0 and len(lv_errors) > 0:
        analysis["mann_whitney"] = {"U": float(u_stat), "p_value": float(p_value)}

    analysis_path = RESULTS_DIR / "e2_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"\nE2 analysis saved to {analysis_path}")

    return props_df, agg


# ── E3: Scalability Analysis ──────────────────────────────────────────────────

def run_e3_analysis():
    """E3: Scaling of inference time and error with network size."""
    logger.info("Running E3: Scalability analysis")

    results = load_all_results()
    if not results:
        logger.warning("No experiment results found. Run run_experiments.py first.")
        return None

    results_df = pd.DataFrame(results)
    metrics_df = pd.json_normalize(results_df["metrics"])
    results_df = pd.concat([results_df.drop(columns=["metrics"]), metrics_df], axis=1)

    # Get num_buses per grid
    bus_counts = {}
    for grid_name in GRID_CODES:
        net = load_simbench_net(grid_name)
        bus_counts[grid_name] = len(net.bus)

    results_df["num_buses"] = results_df["grid"].map(bus_counts)

    # Aggregate across seeds per (grid, model)
    scaling = results_df.groupby(["grid", "model"]).agg(
        num_buses=("num_buses", "first"),
        gnn_time_mean=("gnn_time_ms", "mean"),
        gnn_time_std=("gnn_time_ms", "std"),
        nr_time_mean=("nr_time_ms", "mean"),
        mae_vm_mean=("mae_vm", "mean"),
    ).reset_index()

    # Log-log regression: log(time) vs log(buses)
    for model_name in scaling["model"].unique():
        model_data = scaling[scaling["model"] == model_name]
        log_buses = np.log(model_data["num_buses"].values)
        log_time = np.log(model_data["gnn_time_mean"].values)
        slope, intercept, r, p, se = stats.linregress(log_buses, log_time)
        logger.info(f"  {model_name}: time ~ buses^{slope:.2f} (R²={r**2:.3f})")

    # NR scaling
    nr_data = scaling.groupby("grid").agg(
        num_buses=("num_buses", "first"),
        nr_time=("nr_time_mean", "mean"),
    ).reset_index()
    log_buses = np.log(nr_data["num_buses"].values)
    log_time = np.log(nr_data["nr_time"].values)
    slope, intercept, r, p, se = stats.linregress(log_buses, log_time)
    logger.info(f"  NR: time ~ buses^{slope:.2f} (R²={r**2:.3f})")

    # Save
    scaling_path = RESULTS_DIR / "e3_scaling.json"
    scaling.to_json(scaling_path, orient="records", indent=2)
    logger.info(f"\nE3 scaling data saved to {scaling_path}")

    return scaling


def main():
    props_df, agg = run_e2_analysis()
    scaling = run_e3_analysis()

    if props_df is not None:
        print("\n=== Graph Properties ===")
        print(props_df.to_string(index=False))

    if agg is not None:
        print("\n=== Aggregated Results (mean across seeds) ===")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
