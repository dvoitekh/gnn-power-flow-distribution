"""R10 ablation: generic virtual node vs random hub vs slack-bus shortcut.

Aim: prove that the slack-bus choice matters, not just any global shortcut.
Pilot on 2 grids x 3 hub_modes x 3 seeds = 18 experiments.
"""

import json
import logging
import time

import torch
from torch_geometric.data import Data

from configs import EXPERIMENT_SEEDS, NUM_SAMPLES, RESULTS_DIR
from data_generation import (
    add_virtual_slack_edges, build_dataset, build_graph_topology,
    load_simbench_net,
)
from evaluate import compute_metrics, measure_gnn_time, measure_nr_time
from models import create_model
from train import set_seed, train_model

logger = logging.getLogger(__name__)

PILOT_GRIDS = ["MV_rural", "LV_rural2"]
HUB_MODES = ["slack", "random", "new_node"]


def _augment_with_virtual(data_list, topo, hub_mode):
    return [add_virtual_slack_edges(d, topo, hub_mode=hub_mode) for d in data_list]


def run_one(grid_name, hub_mode, seed, result_dir):
    tag = f"{grid_name}_GraphSAGE_r10_{hub_mode}_s{seed}"
    result_path = result_dir / f"{tag}.json"
    if result_path.exists():
        logger.info(f"  Skipping {tag} (already done)")
        with open(result_path) as f:
            return json.load(f)

    set_seed(seed)
    train_data, val_data, test_data, normalizer = build_dataset(
        grid_name, n_samples=NUM_SAMPLES, seed=seed
    )
    net = load_simbench_net(grid_name)
    topo = build_graph_topology(net)

    train_data = _augment_with_virtual(train_data, topo, hub_mode)
    val_data = _augment_with_virtual(val_data, topo, hub_mode)
    test_data = _augment_with_virtual(test_data, topo, hub_mode)

    model = create_model("GraphSAGE")
    t0 = time.time()
    train_result = train_model(
        model, train_data, val_data, grid_name, "GraphSAGE", seed,
        normalizer=normalizer,
    )
    train_time = time.time() - t0

    metrics = compute_metrics(model, test_data, normalizer)
    gnn_time = measure_gnn_time(model, test_data)
    nr_time = measure_nr_time(net, n_runs=50)

    result = {
        "grid": grid_name,
        "model": "GraphSAGE",
        "seed": seed,
        "hub_mode": hub_mode,
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
    logger.info(f"  {tag}: MAE_Vm={metrics.mae_vm:.6f} | "
                f"Speedup={result['speedup']:.1f}x | {train_time:.0f}s")
    return result


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    result_dir = RESULTS_DIR / "r10_hub_ablation"
    result_dir.mkdir(exist_ok=True)
    total = len(PILOT_GRIDS) * len(HUB_MODES) * len(EXPERIMENT_SEEDS)
    logger.info(f"R10: {total} experiments "
                f"({len(PILOT_GRIDS)} grids x {len(HUB_MODES)} modes x "
                f"{len(EXPERIMENT_SEEDS)} seeds)")
    results = []
    for hub in HUB_MODES:
        for grid in PILOT_GRIDS:
            for seed in EXPERIMENT_SEEDS:
                results.append(run_one(grid, hub, seed, result_dir))
    with open(result_dir / "r10_all.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"R10 complete: {len(results)} results")


if __name__ == "__main__":
    main()
