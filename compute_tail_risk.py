"""R11: tail-risk metrics for baseline vs combined models.

Computes per-grid:
- 95th percentile absolute Vm error
- max absolute Vm error
- Voltage-violation detection precision/recall (V_m outside [0.95, 1.05] p.u.)

Uses cached datasets and trained model checkpoints; no retraining.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from configs import (BATCH_SIZE, EXPERIMENT_SEEDS, GRID_CODES, MODELS_DIR,
                     RESULTS_DIR)
from data_generation import (add_virtual_slack_edges, build_dataset,
                              build_graph_topology, compute_positional_encodings,
                              load_simbench_net)
from models import create_model

logger = logging.getLogger(__name__)


def _augment(data_list, topo, *, pe_type=None, virtual_node=False):
    """Apply same augmentations as during training (RW-PE then virtual edges)."""
    out = data_list
    if pe_type:
        pe = compute_positional_encodings(topo, pe_type)
        from torch_geometric.data import Data as PyGData
        new = []
        for data in out:
            new_x = torch.cat([data.x, pe[:data.x.shape[0]]], dim=1)
            new.append(PyGData(x=new_x, edge_index=data.edge_index,
                               edge_attr=data.edge_attr, y=data.y))
        out = new
    if virtual_node:
        out = [add_virtual_slack_edges(d, topo) for d in out]
    return out


def predict_test(model, test_data, normalizer):
    """Return concatenated predicted+true Vm/Va arrays for the test set."""
    model.eval()
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    pred_all, true_all = [], []
    with torch.no_grad():
        for batch in loader:
            pred_norm = model(batch.x, batch.edge_index, batch.edge_attr,
                              batch.batch)
            pred = normalizer.denormalize_y(pred_norm.cpu())
            true = normalizer.denormalize_y(batch.y.cpu())
            pred_all.append(pred)
            true_all.append(true)
    pred_all = torch.cat(pred_all, dim=0).numpy()
    true_all = torch.cat(true_all, dim=0).numpy()
    return pred_all, true_all


def violation_metrics(pred_vm, true_vm, low=0.98, high=1.02):
    """Precision/recall/F1 for detecting voltage outside [low, high]."""
    true_violation = (true_vm < low) | (true_vm > high)
    pred_violation = (pred_vm < low) | (pred_vm > high)

    tp = int(np.sum(true_violation & pred_violation))
    fp = int(np.sum(~true_violation & pred_violation))
    fn = int(np.sum(true_violation & ~pred_violation))
    tn = int(np.sum(~true_violation & ~pred_violation))

    n_true = int(np.sum(true_violation))
    n_pred = int(np.sum(pred_violation))

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0)
          else float("nan"))
    return {
        "n_true_violations": n_true,
        "n_pred_violations": n_pred,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }


def compute_for_one(grid, seed, model_name, *, pe_type=None,
                     virtual_node=False, model_kwargs=None):
    """Load the trained model checkpoint and compute tail-risk metrics."""
    from train import set_seed
    set_seed(seed)
    train_data, val_data, test_data, normalizer = build_dataset(
        grid, n_samples=2000, seed=seed
    )
    net = load_simbench_net(grid)
    topo = build_graph_topology(net)
    test_data = _augment(test_data, topo, pe_type=pe_type,
                         virtual_node=virtual_node)

    in_dim = test_data[0].x.shape[1]
    model_kwargs = dict(model_kwargs or {})
    model_kwargs["in_dim"] = in_dim
    model = create_model(model_name, **model_kwargs)
    ckpt_path = MODELS_DIR / f"{grid}_{model_name}_s{seed}.pt"
    if not ckpt_path.exists():
        logger.warning(f"Missing checkpoint: {ckpt_path}")
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    pred, true = predict_test(model, test_data, normalizer)
    vm_err = np.abs(pred[:, 0] - true[:, 0])

    violations = violation_metrics(pred[:, 0], true[:, 0])
    return {
        "grid": grid, "seed": seed, "model": model_name,
        "p95_vm": float(np.percentile(vm_err, 95)),
        "p99_vm": float(np.percentile(vm_err, 99)),
        "max_vm": float(vm_err.max()),
        "mean_vm": float(vm_err.mean()),
        **violations,
    }


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = RESULTS_DIR / "tail_risk"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for grid in GRID_CODES:
        for seed in EXPERIMENT_SEEDS:
            r = compute_for_one(grid, seed, "GraphSAGE")
            if r:
                r["config"] = "baseline"
                rows.append(r)
            r = compute_for_one(grid, seed, "ResidualGraphSAGE",
                                pe_type="random_walk", virtual_node=True,
                                model_kwargs={"num_layers": 8})
            if r:
                r["config"] = "combined"
                rows.append(r)

    out_path = out_dir / "tail_risk_per_seed.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved {len(rows)} rows -> {out_path}")

    # Summary per (grid, config)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["grid"], r["config"])].append(r)
    summary = {}
    for (grid, cfg), rs in agg.items():
        summary[f"{grid}__{cfg}"] = {
            "p95_vm_mean": float(np.mean([x["p95_vm"] for x in rs])),
            "max_vm_mean": float(np.mean([x["max_vm"] for x in rs])),
            "violations_true_total": int(sum(x["n_true_violations"] for x in rs)),
            "violations_pred_total": int(sum(x["n_pred_violations"] for x in rs)),
            "precision_mean": float(np.nanmean([x["precision"] for x in rs])),
            "recall_mean": float(np.nanmean([x["recall"] for x in rs])),
            "f1_mean": float(np.nanmean([x["f1"] for x in rs])),
        }
    with open(out_dir / "tail_risk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print per-grid table
    print()
    print(f"{'Grid':<14s} {'config':<10s} {'p95_vm x1e-3':>14s} {'max_vm x1e-3':>14s} "
          f"{'true_v':>7s} {'precision':>10s} {'recall':>8s}")
    for grid in GRID_CODES:
        for cfg in ["baseline", "combined"]:
            key = f"{grid}__{cfg}"
            if key not in summary: continue
            s = summary[key]
            print(f"{grid:<14s} {cfg:<10s} "
                  f"{s['p95_vm_mean']*1000:>14.2f} "
                  f"{s['max_vm_mean']*1000:>14.2f} "
                  f"{s['violations_true_total']:>7d} "
                  f"{s['precision_mean']:>10.3f} "
                  f"{s['recall_mean']:>8.3f}")


if __name__ == "__main__":
    main()
