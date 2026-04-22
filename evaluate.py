"""Evaluation: metrics (MAE/RMSE/MaxAE for Vm, Va) + timing (GNN vs NR)."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from configs import BATCH_SIZE
from data_generation import Normalizer

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Evaluation metrics for voltage magnitude and angle."""
    mae_vm: float       # MAE of Vm in p.u.
    mae_va: float       # MAE of Va in degrees
    rmse_vm: float      # RMSE of Vm in p.u.
    rmse_va: float      # RMSE of Va in degrees
    max_ae_vm: float    # Max absolute error of Vm in p.u.
    max_ae_va: float    # Max absolute error of Va in degrees

    def to_dict(self) -> dict:
        return {
            "mae_vm": self.mae_vm,
            "mae_va": self.mae_va,
            "rmse_vm": self.rmse_vm,
            "rmse_va": self.rmse_va,
            "max_ae_vm": self.max_ae_vm,
            "max_ae_va": self.max_ae_va,
        }


def compute_metrics(model: nn.Module, test_data: list,
                    normalizer: Normalizer,
                    batch_size: int = BATCH_SIZE) -> Metrics:
    """Compute denormalized metrics on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # Denormalize
            pred = normalizer.denormalize_y(pred_norm.cpu())
            true = normalizer.denormalize_y(batch.y.cpu())
            all_pred.append(pred)
            all_true.append(true)

    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_true = torch.cat(all_true, dim=0).numpy()

    # Vm (column 0), Va (column 1)
    vm_err = np.abs(all_pred[:, 0] - all_true[:, 0])
    va_err = np.abs(all_pred[:, 1] - all_true[:, 1])

    return Metrics(
        mae_vm=float(vm_err.mean()),
        mae_va=float(va_err.mean()),
        rmse_vm=float(np.sqrt((vm_err ** 2).mean())),
        rmse_va=float(np.sqrt((va_err ** 2).mean())),
        max_ae_vm=float(vm_err.max()),
        max_ae_va=float(va_err.max()),
    )


def measure_gnn_time(model: nn.Module, test_data: list,
                     n_runs: int = 100, batch_size: int = 1) -> float:
    """Measure GNN inference time per sample in milliseconds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    sample = next(iter(loader)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(sample.x, sample.edge_index, sample.edge_attr, sample.batch)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample.x, sample.edge_index, sample.edge_attr, sample.batch)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return float(np.median(times))


def measure_nr_time(net: pp.pandapowerNet, n_runs: int = 100) -> float:
    """Measure Newton-Raphson power flow time per solve in milliseconds."""
    # Warmup
    for _ in range(5):
        try:
            pp.runpp(net, algorithm="nr", init="flat", numba=True)
        except pp.LoadflowNotConverged:
            pass

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            pp.runpp(net, algorithm="nr", init="flat", numba=True)
        except pp.LoadflowNotConverged:
            pass
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return float(np.median(times))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from data_generation import build_dataset, load_simbench_net
    from models import create_model
    from train import set_seed, train_model

    set_seed(42)
    train_data, val_data, test_data, norm = build_dataset(
        "LV_rural1", n_samples=500, seed=42, force=True
    )

    model = create_model("GCN")
    train_model(model, train_data, val_data, "LV_rural1", "GCN", seed=42, epochs=100)

    metrics = compute_metrics(model, test_data, norm)
    print(f"\nMetrics: MAE_Vm={metrics.mae_vm:.6f} p.u., MAE_Va={metrics.mae_va:.4f} deg")
    print(f"         RMSE_Vm={metrics.rmse_vm:.6f}, RMSE_Va={metrics.rmse_va:.4f}")
    print(f"         MaxAE_Vm={metrics.max_ae_vm:.6f}, MaxAE_Va={metrics.max_ae_va:.4f}")

    gnn_time = measure_gnn_time(model, test_data)
    print(f"\nGNN time: {gnn_time:.3f} ms/sample")

    net = load_simbench_net("LV_rural1")
    nr_time = measure_nr_time(net, n_runs=50)
    print(f"NR time:  {nr_time:.3f} ms/solve")
    print(f"Speedup:  {nr_time / gnn_time:.1f}x")
