"""Unified experiment runner for Paper v2: E1-E5.

Usage:
    python run_all_experiments.py --experiment E1          # Physics-informed loss
    python run_all_experiments.py --experiment E2          # Positional encodings
    python run_all_experiments.py --experiment E3          # Deeper GNNs (pilot)
    python run_all_experiments.py --experiment E3_full     # Best depth config → all grids
    python run_all_experiments.py --experiment E4          # Virtual slack node
    python run_all_experiments.py --experiment E5          # Combined best
    python run_all_experiments.py --experiment baseline    # Reproduce baseline (GraphSAGE)
"""

import argparse
import json
import logging
import time

import torch

from configs import (
    DEEP_DEPTHS,
    DEEP_TECHNIQUES,
    EXPERIMENT_SEEDS,
    GRID_CODES,
    PHYSICS_LAMBDAS,
    PE_TYPES,
    PILOT_GRIDS,
    RESULTS_DIR,
    NUM_SAMPLES,
)
from data_generation import (
    build_dataset,
    build_graph_topology,
    load_simbench_net,
    compute_positional_encodings,
    add_virtual_slack_edges,
    Normalizer,
)
from evaluate import compute_metrics, measure_gnn_time, measure_nr_time
from models import create_model
from train import set_seed, train_model

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _augment_with_pe(data_list: list, topo, pe_type: str) -> list:
    """Concatenate positional encodings to node features for all samples."""
    pe = compute_positional_encodings(topo, pe_type)
    augmented = []
    for data in data_list:
        new_x = torch.cat([data.x, pe.expand(1, -1, -1).squeeze(0)
                           if data.x.shape[0] == pe.shape[0]
                           else pe[:data.x.shape[0]]], dim=1)
        from torch_geometric.data import Data
        augmented.append(Data(
            x=new_x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
        ))
    return augmented


def _augment_with_virtual(data_list: list, topo) -> list:
    """Add virtual slack edges to all samples."""
    return [add_virtual_slack_edges(d, topo) for d in data_list]


def _run_single(grid_name, model_name, seed, result_dir,
                extra_tag="", model_kwargs=None, train_kwargs=None,
                pe_type=None, virtual_node=False):
    """Run one experiment and save result JSON."""
    tag = f"{grid_name}_{model_name}{extra_tag}_s{seed}"
    result_path = result_dir / f"{tag}.json"

    if result_path.exists():
        logger.info(f"  Skipping {tag} (already done)")
        with open(result_path) as f:
            return json.load(f)

    set_seed(seed)
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}

    # Load dataset
    train_data, val_data, test_data, normalizer = build_dataset(
        grid_name, n_samples=NUM_SAMPLES, seed=seed
    )

    # Load topology for PE / virtual node
    net = load_simbench_net(grid_name)
    topo = build_graph_topology(net)

    # Apply positional encodings
    if pe_type:
        train_data = _augment_with_pe(train_data, topo, pe_type)
        val_data = _augment_with_pe(val_data, topo, pe_type)
        test_data = _augment_with_pe(test_data, topo, pe_type)
        # Update in_dim
        model_kwargs["in_dim"] = train_data[0].x.shape[1]

    # Apply virtual slack edges
    if virtual_node:
        train_data = _augment_with_virtual(train_data, topo)
        val_data = _augment_with_virtual(val_data, topo)
        test_data = _augment_with_virtual(test_data, topo)

    # Create and train model
    model = create_model(model_name, **model_kwargs)
    t0 = time.time()
    train_result = train_model(
        model, train_data, val_data, grid_name, model_name, seed,
        normalizer=normalizer, **train_kwargs
    )
    train_time = time.time() - t0

    # Evaluate
    metrics = compute_metrics(model, test_data, normalizer)
    gnn_time = measure_gnn_time(model, test_data)
    nr_time = measure_nr_time(net, n_runs=50)

    result = {
        "grid": grid_name,
        "model": model_name,
        "seed": seed,
        "experiment_tag": extra_tag,
        "metrics": metrics.to_dict(),
        "gnn_time_ms": gnn_time,
        "nr_time_ms": nr_time,
        "speedup": nr_time / gnn_time if gnn_time > 0 else 0,
        "num_epochs": train_result.num_epochs,
        "best_epoch": train_result.best_epoch,
        "best_val_loss": train_result.best_val_loss,
        "train_time_s": train_time,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  {tag}: MAE_Vm={metrics.mae_vm:.6f} | MAE_Va={metrics.mae_va:.4f} | "
                f"Speedup={result['speedup']:.1f}x | {train_time:.0f}s")
    return result


# ── E1: Physics-Informed Loss ──────────────────────────────────────────────

def run_e1():
    """E1: GraphSAGE + physics loss with different λ values."""
    result_dir = RESULTS_DIR / "e1_physics"
    grids = list(GRID_CODES.keys())

    total = len(grids) * len(PHYSICS_LAMBDAS) * len(EXPERIMENT_SEEDS)
    logger.info(f"E1: {len(grids)} grids × {len(PHYSICS_LAMBDAS)} lambdas × "
                f"{len(EXPERIMENT_SEEDS)} seeds = {total} experiments")

    results = []
    for lam in PHYSICS_LAMBDAS:
        for grid in grids:
            for seed in EXPERIMENT_SEEDS:
                r = _run_single(
                    grid, "GraphSAGE", seed, result_dir,
                    extra_tag=f"_pi{lam}",
                    train_kwargs={"physics_weight": lam},
                )
                results.append(r)

    combined_path = result_dir / "e1_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E1 complete: {len(results)} results saved to {combined_path}")
    return results


# ── E2: Positional Encodings ────────────────────────────────────────────────

def run_e2():
    """E2: GraphSAGE + different positional encodings."""
    result_dir = RESULTS_DIR / "e2_pe"
    grids = list(GRID_CODES.keys())

    total = len(grids) * len(PE_TYPES) * len(EXPERIMENT_SEEDS)
    logger.info(f"E2: {len(grids)} grids × {len(PE_TYPES)} PE types × "
                f"{len(EXPERIMENT_SEEDS)} seeds = {total} experiments")

    results = []
    for pe_type in PE_TYPES:
        for grid in grids:
            for seed in EXPERIMENT_SEEDS:
                r = _run_single(
                    grid, "GraphSAGE", seed, result_dir,
                    extra_tag=f"_pe_{pe_type}",
                    pe_type=pe_type,
                )
                results.append(r)

    combined_path = result_dir / "e2_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E2 complete: {len(results)} results saved to {combined_path}")
    return results


# ── E3: Deeper GNNs ─────────────────────────────────────────────────────────

def run_e3_pilot():
    """E3 pilot: depth × technique sweep on 2 grids."""
    result_dir = RESULTS_DIR / "e3_deep"
    model_map = {
        "residual": "ResidualGraphSAGE",
        "jknet": "JKGraphSAGE",
        "dropedge": "DropEdgeGraphSAGE",
    }

    total = len(PILOT_GRIDS) * len(DEEP_DEPTHS) * len(DEEP_TECHNIQUES) * len(EXPERIMENT_SEEDS)
    logger.info(f"E3 pilot: {len(PILOT_GRIDS)} grids × {len(DEEP_DEPTHS)} depths × "
                f"{len(DEEP_TECHNIQUES)} techniques × {len(EXPERIMENT_SEEDS)} seeds = {total}")

    results = []
    for depth in DEEP_DEPTHS:
        for tech in DEEP_TECHNIQUES:
            model_name = model_map[tech]
            for grid in PILOT_GRIDS:
                for seed in EXPERIMENT_SEEDS:
                    r = _run_single(
                        grid, model_name, seed, result_dir,
                        extra_tag=f"_d{depth}",
                        model_kwargs={"num_layers": depth},
                    )
                    results.append(r)

    combined_path = result_dir / "e3_pilot.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E3 pilot complete: {len(results)} results")
    return results


def run_e3_full(best_technique: str = "residual", best_depth: int = 16):
    """E3 full: best config on all 10 grids."""
    result_dir = RESULTS_DIR / "e3_deep"
    model_map = {
        "residual": "ResidualGraphSAGE",
        "jknet": "JKGraphSAGE",
        "dropedge": "DropEdgeGraphSAGE",
    }
    model_name = model_map[best_technique]
    grids = list(GRID_CODES.keys())

    total = len(grids) * len(EXPERIMENT_SEEDS)
    logger.info(f"E3 full: {model_name} d={best_depth} on {len(grids)} grids × "
                f"{len(EXPERIMENT_SEEDS)} seeds = {total}")

    results = []
    for grid in grids:
        for seed in EXPERIMENT_SEEDS:
            r = _run_single(
                grid, model_name, seed, result_dir,
                extra_tag=f"_d{best_depth}_full",
                model_kwargs={"num_layers": best_depth},
            )
            results.append(r)

    combined_path = result_dir / "e3_full.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E3 full complete: {len(results)} results")
    return results


# ── E4: Virtual Slack Node ──────────────────────────────────────────────────

def run_e4():
    """E4: GraphSAGE + virtual slack edges."""
    result_dir = RESULTS_DIR / "e4_virtual"
    grids = list(GRID_CODES.keys())

    total = len(grids) * len(EXPERIMENT_SEEDS)
    logger.info(f"E4: {len(grids)} grids × {len(EXPERIMENT_SEEDS)} seeds = {total}")

    results = []
    for grid in grids:
        for seed in EXPERIMENT_SEEDS:
            r = _run_single(
                grid, "GraphSAGE", seed, result_dir,
                extra_tag="_virtual",
                virtual_node=True,
            )
            results.append(r)

    combined_path = result_dir / "e4_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E4 complete: {len(results)} results")
    return results


# ── E5: Combined Best ────────────────────────────────────────────────────────

def run_e5(best_lambda: float = 0.1, best_pe: str = "distance_from_slack",
           best_technique: str = "residual", best_depth: int = 16,
           use_virtual: bool = True):
    """E5: Combined best components from E1-E4."""
    result_dir = RESULTS_DIR / "e5_combined"
    model_map = {
        "residual": "ResidualGraphSAGE",
        "jknet": "JKGraphSAGE",
        "dropedge": "DropEdgeGraphSAGE",
    }
    model_name = model_map[best_technique]
    grids = list(GRID_CODES.keys())

    total = len(grids) * len(EXPERIMENT_SEEDS)
    logger.info(f"E5 combined: {model_name} d={best_depth} + PI(λ={best_lambda}) + "
                f"PE({best_pe}) + virtual={use_virtual}")
    logger.info(f"  {len(grids)} grids × {len(EXPERIMENT_SEEDS)} seeds = {total}")

    results = []
    for grid in grids:
        for seed in EXPERIMENT_SEEDS:
            r = _run_single(
                grid, model_name, seed, result_dir,
                extra_tag=f"_combined",
                model_kwargs={"num_layers": best_depth},
                train_kwargs={"physics_weight": best_lambda},
                pe_type=best_pe,
                virtual_node=use_virtual,
            )
            results.append(r)

    combined_path = result_dir / "e5_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"E5 complete: {len(results)} results")
    return results


# ── Baseline (GraphSAGE without enhancements) ──────────────────────────────

def run_baseline():
    """Reproduce baseline GraphSAGE on all grids for fair comparison."""
    result_dir = RESULTS_DIR / "baseline"
    grids = list(GRID_CODES.keys())

    total = len(grids) * len(EXPERIMENT_SEEDS)
    logger.info(f"Baseline: GraphSAGE on {len(grids)} grids × "
                f"{len(EXPERIMENT_SEEDS)} seeds = {total}")

    results = []
    for grid in grids:
        for seed in EXPERIMENT_SEEDS:
            r = _run_single(grid, "GraphSAGE", seed, result_dir)
            results.append(r)

    combined_path = result_dir / "baseline_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline complete: {len(results)} results")
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper v2 Experiments (E1-E5)")
    parser.add_argument("--experiment", required=True,
                        choices=["baseline", "E1", "E2", "E3", "E3_full", "E4", "E5", "all"],
                        help="Which experiment to run")
    parser.add_argument("--best-lambda", type=float, default=0.1,
                        help="Best λ for E5 (from E1 results)")
    parser.add_argument("--best-pe", type=str, default="distance_from_slack",
                        help="Best PE type for E5 (from E2 results)")
    parser.add_argument("--best-technique", type=str, default="residual",
                        help="Best technique for E3_full/E5")
    parser.add_argument("--best-depth", type=int, default=16,
                        help="Best depth for E3_full/E5")
    parser.add_argument("--no-virtual", action="store_true",
                        help="Disable virtual node in E5")
    args = parser.parse_args()

    if args.experiment == "baseline":
        run_baseline()
    elif args.experiment == "E1":
        run_e1()
    elif args.experiment == "E2":
        run_e2()
    elif args.experiment == "E3":
        run_e3_pilot()
    elif args.experiment == "E3_full":
        run_e3_full(args.best_technique, args.best_depth)
    elif args.experiment == "E4":
        run_e4()
    elif args.experiment == "E5":
        run_e5(args.best_lambda, args.best_pe, args.best_technique,
               args.best_depth, not args.no_virtual)
    elif args.experiment == "all":
        logger.info("Running ALL experiments sequentially: baseline → E1 → E2 → E3 → E4")
        run_baseline()
        run_e1()
        run_e2()
        run_e3_pilot()
        run_e4()
        logger.info("\n" + "=" * 60)
        logger.info("E1-E4 complete. Analyze results to pick best configs for E3_full and E5.")
        logger.info("Then run: python run_all_experiments.py --experiment E3_full")
        logger.info("Then run: python run_all_experiments.py --experiment E5")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
