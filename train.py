"""Training loop: AdamW + OneCycleLR + early stopping."""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader

from configs import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    PATIENCE,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Training outcome."""
    best_val_loss: float
    best_epoch: int
    num_epochs: int
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def physics_loss(pred_norm: torch.Tensor, data, normalizer, topo_info: dict) -> torch.Tensor:
    """Compute voltage consistency loss as a physics-inspired soft constraint.

    Instead of full Kirchhoff power balance (which requires careful impedance
    normalization), we use a simpler but effective constraint: predicted voltages
    of connected nodes should be consistent — the voltage drop across an edge
    should be proportional to the power flow and impedance.

    Specifically, for each edge (i,j) with impedance z_ij:
        |V_i - V_j| should be small for low-impedance edges

    This is a soft smoothness constraint weighted by inverse impedance,
    encouraging physically plausible voltage profiles.

    Additionally, we penalize deviation of Vm from 1.0 p.u. (voltage should
    be close to nominal in well-operated grids).
    """
    # Denormalize predictions
    pred = pred_norm * (normalizer.y_std + 1e-8) + normalizer.y_mean
    vm = pred[:, 0]  # voltage magnitude in p.u.
    va = pred[:, 1]  # voltage angle in degrees

    edge_index = data.edge_index
    edge_attr = data.edge_attr  # [E, 3]: r_ohm, x_ohm, b
    src, dst = edge_index[0], edge_index[1]

    # Edge impedance magnitude (for weighting)
    r = edge_attr[:, 0]
    x = edge_attr[:, 1]
    z_mag = torch.sqrt(r**2 + x**2 + 1e-10)

    # Inverse impedance weight: low-Z edges should have small voltage drops
    # Normalize weights to mean=1 to keep loss scale stable
    w = 1.0 / (z_mag + 1e-6)
    w = w / (w.mean() + 1e-8)

    # Term 1: Weighted voltage magnitude consistency across edges
    vm_diff = (vm[src] - vm[dst]) ** 2
    loss_vm_smooth = torch.mean(w * vm_diff)

    # Term 2: Weighted angle consistency across edges
    va_diff = (va[src] - va[dst]) ** 2
    loss_va_smooth = torch.mean(w * va_diff)

    # Term 3: Voltage magnitude should be close to 1.0 p.u. (soft bound)
    # Penalize deviations > 0.05 p.u. more heavily
    vm_dev = torch.clamp(torch.abs(vm - 1.0) - 0.05, min=0.0) ** 2
    loss_vm_bound = torch.mean(vm_dev)

    return loss_vm_smooth + 0.01 * loss_va_smooth + loss_vm_bound


def train_model(model: nn.Module, train_data: list, val_data: list,
                grid_name: str, model_name: str, seed: int,
                epochs: int = EPOCHS, lr: float = LEARNING_RATE,
                batch_size: int = BATCH_SIZE, patience: int = PATIENCE,
                weight_decay: float = WEIGHT_DECAY,
                physics_weight: float = 0.0,
                normalizer=None) -> TrainResult:
    """Train a GNN model with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                           steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y)
            if physics_weight > 0.0 and normalizer is not None:
                p_loss = physics_loss(pred, batch, normalizer, None)
                # Linear warmup: ramp λ from 0 to target over first 30% of epochs
                warmup_frac = min(epoch / max(epochs * 0.3, 1), 1.0)
                loss = loss + physics_weight * warmup_frac * p_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item() * batch.num_graphs
            n_train += batch.num_graphs

        avg_train_loss = total_train_loss / n_train

        # ── Validate ──
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(pred, batch.y)
                total_val_loss += loss.item() * batch.num_graphs
                n_val += batch.num_graphs

        avg_val_loss = total_val_loss / n_val
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ── Early Stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"[{grid_name}/{model_name}/s{seed}] "
                        f"Epoch {epoch}/{epochs} | "
                        f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(best: {best_epoch}, val_loss: {best_val_loss:.6f})")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model = model.to("cpu")

    # Save checkpoint
    ckpt_path = MODELS_DIR / f"{grid_name}_{model_name}_s{seed}.pt"
    torch.save(best_state, ckpt_path)
    logger.info(f"Saved best model to {ckpt_path}")

    return TrainResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        num_epochs=epoch,
        train_losses=train_losses,
        val_losses=val_losses,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from data_generation import build_dataset
    from models import create_model

    set_seed(42)
    train_data, val_data, test_data, norm = build_dataset(
        "LV_rural1", n_samples=200, seed=42, force=True
    )

    model = create_model("GCN")
    result = train_model(model, train_data, val_data,
                         "LV_rural1", "GCN", seed=42, epochs=50)
    print(f"\nBest val loss: {result.best_val_loss:.6f} at epoch {result.best_epoch}")
    print(f"Final train losses: {result.train_losses[-3:]}")