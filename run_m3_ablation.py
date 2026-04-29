"""M3 ablation: ResidualGraphSAGE d=8 + RW-PE k=16 + NO virtual edges.

Purpose: isolate the contribution of virtual slack edges in the combined model.
Without this ablation, reviewers cannot attribute the 43.8% improvement to any
single component. Runs 10 grids x 3 seeds = 30 experiments.
"""

import json
import logging

from configs import EXPERIMENT_SEEDS, GRID_CODES, RESULTS_DIR
from run_all_experiments import _run_single


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result_dir = RESULTS_DIR / "m3_no_virtual"
    result_dir.mkdir(exist_ok=True)

    grids = list(GRID_CODES.keys())
    total = len(grids) * len(EXPERIMENT_SEEDS)
    logging.info(f"M3: ResidualGraphSAGE d=8 + RW-PE k=16, NO virtual edges")
    logging.info(f"  {len(grids)} grids x {len(EXPERIMENT_SEEDS)} seeds = {total}")

    results = []
    for grid in grids:
        for seed in EXPERIMENT_SEEDS:
            r = _run_single(
                grid, "ResidualGraphSAGE", seed, result_dir,
                extra_tag="_m3_novirtual",
                model_kwargs={"num_layers": 8},
                pe_type="random_walk",
                virtual_node=False,
            )
            results.append(r)

    combined_path = result_dir / "m3_all.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"M3 complete: {len(results)} results -> {combined_path}")


if __name__ == "__main__":
    main()
