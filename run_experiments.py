"""E1: Main benchmark — grid × model × seed."""

import argparse
import json
import logging
import time

from configs import (
    EXPERIMENT_SEEDS,
    GRID_CODES,
    MODEL_NAMES,
    NUM_SAMPLES,
    RESULTS_DIR,
)
from data_generation import build_dataset, load_simbench_net
from evaluate import compute_metrics, measure_gnn_time, measure_nr_time
from models import create_model
from train import set_seed, train_model

logger = logging.getLogger(__name__)


def run_single(grid_name: str, model_name: str, seed: int) -> dict:
    """Run one (grid, model, seed) experiment. Returns result dict."""
    set_seed(seed)

    # Load / build dataset (cached per grid+seed)
    train_data, val_data, test_data, normalizer = build_dataset(
        grid_name, n_samples=NUM_SAMPLES, seed=seed
    )

    # Create and train model
    # MPNN uses fewer layers (2) to keep training time manageable
    kwargs = {"num_layers": 2} if model_name == "MPNN" else {}
    model = create_model(model_name, **kwargs)
    t0 = time.time()
    train_result = train_model(
        model, train_data, val_data, grid_name, model_name, seed
    )
    train_time = time.time() - t0

    # Evaluate
    metrics = compute_metrics(model, test_data, normalizer)
    gnn_time = measure_gnn_time(model, test_data)

    # NR timing (only need once per grid, but inexpensive)
    net = load_simbench_net(grid_name)
    nr_time = measure_nr_time(net, n_runs=100)

    result = {
        "grid": grid_name,
        "model": model_name,
        "seed": seed,
        "metrics": metrics.to_dict(),
        "gnn_time_ms": gnn_time,
        "nr_time_ms": nr_time,
        "speedup": nr_time / gnn_time if gnn_time > 0 else 0,
        "num_epochs": train_result.num_epochs,
        "best_epoch": train_result.best_epoch,
        "best_val_loss": train_result.best_val_loss,
        "train_time_s": train_time,
        "train_losses": train_result.train_losses,
        "val_losses": train_result.val_losses,
    }

    # Save result
    result_path = RESULTS_DIR / f"{grid_name}_{model_name}_s{seed}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved result to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="GNN Power Flow Benchmark (E1)")
    parser.add_argument("--grids", nargs="+", default=list(GRID_CODES.keys()),
                        help="Grid names to benchmark")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES,
                        help="Model names to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=EXPERIMENT_SEEDS,
                        help="Random seeds")
    args = parser.parse_args()

    total = len(args.grids) * len(args.models) * len(args.seeds)
    logger.info(f"Starting E1: {len(args.grids)} grids × {len(args.models)} models "
                f"× {len(args.seeds)} seeds = {total} experiments")

    results = []
    for i, grid in enumerate(args.grids):
        for model in args.models:
            for seed in args.seeds:
                idx = len(results) + 1
                logger.info(f"\n{'='*60}")
                logger.info(f"[{idx}/{total}] {grid} / {model} / seed={seed}")
                logger.info(f"{'='*60}")

                # Skip if already completed
                result_path = RESULTS_DIR / f"{grid}_{model}_s{seed}.json"
                if result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                    logger.info(f"  Skipping (already done)")
                    results.append(result)
                    m = result["metrics"]
                    logger.info(f"  MAE_Vm={m['mae_vm']:.6f} | MAE_Va={m['mae_va']:.4f} | "
                                f"Speedup={result['speedup']:.1f}x")
                    continue

                result = run_single(grid, model, seed)
                results.append(result)

                m = result["metrics"]
                logger.info(f"  MAE_Vm={m['mae_vm']:.6f} | MAE_Va={m['mae_va']:.4f} | "
                            f"Speedup={result['speedup']:.1f}x")

    # Save combined results
    combined_path = RESULTS_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll results saved to {combined_path}")

    # Print summary table
    print(f"\n{'Grid':<15} {'Model':<12} {'MAE_Vm':>10} {'MAE_Va':>10} {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        m = r["metrics"]
        print(f"{r['grid']:<15} {r['model']:<12} {m['mae_vm']:>10.6f} "
              f"{m['mae_va']:>10.4f} {r['speedup']:>7.1f}x")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
