"""Data generation: SimBench grids → pandapower AC PF → PyG Data objects."""

import logging
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import simbench as sb
import torch
from pandapower.toolbox import create_continuous_bus_index
from torch_geometric.data import Data

from collections import deque

from scipy.sparse import coo_matrix, eye as speye
from scipy.sparse.linalg import eigsh

from configs import (
    DATA_DIR,
    FALLBACK_VARIATION,
    GRID_CODES,
    LAPLACIAN_PE_DIM,
    LOAD_VARIATION,
    MAX_FAIL_RATE,
    NUM_SAMPLES,
    RANDOM_WALK_PE_DIM,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ── Network Loading ────────────────────────────────────────────────────────────

def load_simbench_net(grid_name: str) -> pp.pandapowerNet:
    """Load a SimBench network and apply convergence fixes."""
    code = GRID_CODES[grid_name]
    net = sb.get_simbench_net(code)
    # Fix transformer phase shift for flat-start NR convergence
    if len(net.trafo) > 0:
        net.trafo["shift_degree"] = 0
    # Contiguous bus indices (required by pandapower's internal arrays)
    create_continuous_bus_index(net)
    # Fix NaN voltage-dependent load params → constant power
    for col in ["const_z_p_percent", "const_i_p_percent",
                "const_z_q_percent", "const_i_q_percent"]:
        if col in net.load.columns:
            net.load[col] = net.load[col].fillna(0)
    # Ensure all switches are closed (keep bus-bus switches that maintain connectivity)
    net.switch["closed"] = True
    return net


# ── Graph Topology ─────────────────────────────────────────────────────────────

@dataclass
class GraphTopology:
    """Static graph structure extracted once per grid."""
    bus_indices: np.ndarray       # contiguous 0..N-1
    edge_index: torch.LongTensor  # [2, E] undirected
    edge_attr: torch.FloatTensor  # [E, 3] r, x, b
    vn_kv: torch.FloatTensor      # [N] nominal voltage per bus
    ext_grid_buses: set           # slack bus indices
    num_nodes: int
    num_edges: int


def build_graph_topology(net: pp.pandapowerNet) -> GraphTopology:
    """Extract static graph topology from a pandapower network."""
    bus_in_service = net.bus[net.bus.in_service].index.values
    num_nodes = len(bus_in_service)
    bus_set = set(bus_in_service)

    edges_src, edges_dst = [], []
    edge_r, edge_x, edge_b = [], [], []

    # Lines (undirected: add both directions)
    for _, row in net.line[net.line.in_service].iterrows():
        fb, tb = int(row.from_bus), int(row.to_bus)
        if fb not in bus_set or tb not in bus_set:
            continue
        r = row.r_ohm_per_km * row.length_km
        x = row.x_ohm_per_km * row.length_km
        b = row.c_nf_per_km * row.length_km * 2 * np.pi * 50 * 1e-9  # susceptance in S
        for s, d in [(fb, tb), (tb, fb)]:
            edges_src.append(s)
            edges_dst.append(d)
            edge_r.append(r)
            edge_x.append(x)
            edge_b.append(b)

    # Transformers as additional edges
    for _, row in net.trafo[net.trafo.in_service].iterrows():
        hv, lv = int(row.hv_bus), int(row.lv_bus)
        if hv not in bus_set or lv not in bus_set:
            continue
        # Convert trafo impedance to ohms (referred to HV side)
        z_base = row.vn_hv_kv ** 2 / row.sn_mva
        r = row.vkr_percent / 100 * z_base
        x_sq = (row.vk_percent / 100) ** 2 - (row.vkr_percent / 100) ** 2
        x = np.sqrt(max(x_sq, 0)) * z_base
        b = row.i0_percent / 100 / z_base if z_base > 0 else 0
        for s, d in [(hv, lv), (lv, hv)]:
            edges_src.append(s)
            edges_dst.append(d)
            edge_r.append(r)
            edge_x.append(x)
            edge_b.append(b)

    # Bus-bus switches as zero-impedance edges (maintain connectivity)
    if len(net.switch) > 0:
        bb_switches = net.switch[(net.switch.et == "b") & net.switch.closed]
        for _, row in bb_switches.iterrows():
            fb, tb = int(row.bus), int(row.element)
            if fb not in bus_set or tb not in bus_set:
                continue
            for s, d in [(fb, tb), (tb, fb)]:
                edges_src.append(s)
                edges_dst.append(d)
                edge_r.append(row.z_ohm if row.z_ohm > 0 else 1e-6)
                edge_x.append(1e-6)
                edge_b.append(0.0)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(
        np.column_stack([edge_r, edge_x, edge_b]), dtype=torch.float32
    )
    vn_kv = torch.tensor(
        net.bus.loc[bus_in_service, "vn_kv"].values, dtype=torch.float32
    )
    ext_grid_buses = set(net.ext_grid[net.ext_grid.in_service].bus.values)

    return GraphTopology(
        bus_indices=bus_in_service,
        edge_index=edge_index,
        edge_attr=edge_attr,
        vn_kv=vn_kv,
        ext_grid_buses=ext_grid_buses,
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
    )


# ── Sample Generation ──────────────────────────────────────────────────────────

def _get_bus_type_onehot(net: pp.pandapowerNet, topo: GraphTopology) -> np.ndarray:
    """Compute bus type one-hot: [slack, PV, PQ] for each bus."""
    n = topo.num_nodes
    onehot = np.zeros((n, 3), dtype=np.float32)
    gen_buses = set(net.gen[net.gen.in_service].bus.values) if len(net.gen) > 0 else set()
    for i, bus_idx in enumerate(topo.bus_indices):
        if bus_idx in topo.ext_grid_buses:
            onehot[i, 0] = 1.0  # slack
        elif bus_idx in gen_buses:
            onehot[i, 1] = 1.0  # PV
        else:
            onehot[i, 2] = 1.0  # PQ
    return onehot


def _compute_scheduled_injections(net: pp.pandapowerNet,
                                  topo: GraphTopology) -> np.ndarray:
    """Aggregate P/Q injections per bus from load and sgen tables."""
    n = topo.num_nodes
    bus_to_idx = {b: i for i, b in enumerate(topo.bus_indices)}
    p = np.zeros(n, dtype=np.float32)
    q = np.zeros(n, dtype=np.float32)

    # Loads (consume power → negative injection convention, but we keep sign as-is:
    # loads are positive P draw, sgens are positive P injection)
    for _, row in net.load[net.load.in_service].iterrows():
        idx = bus_to_idx.get(int(row.bus))
        if idx is not None:
            p[idx] -= row.p_mw * row.scaling
            q[idx] -= row.q_mvar * row.scaling

    for _, row in net.sgen[net.sgen.in_service].iterrows():
        idx = bus_to_idx.get(int(row.bus))
        if idx is not None:
            p[idx] += row.p_mw * row.scaling
            q_val = row.q_mvar if hasattr(row, "q_mvar") and not np.isnan(row.q_mvar) else 0
            q[idx] += q_val * row.scaling

    return np.column_stack([p, q])


def generate_samples(net: pp.pandapowerNet, topo: GraphTopology,
                     n_samples: int, seed: int,
                     variation: float = LOAD_VARIATION) -> list[Data]:
    """Generate power flow samples with random load/sgen scaling."""
    rng = np.random.RandomState(seed)
    bus_type_onehot = _get_bus_type_onehot(net, topo)

    # Store original values
    orig_load_p = net.load.p_mw.values.copy()
    orig_load_q = net.load.q_mvar.values.copy()
    orig_sgen_p = net.sgen.p_mw.values.copy() if len(net.sgen) > 0 else np.array([])

    samples = []
    n_failed = 0

    for i in range(n_samples):
        # Random scaling factors
        load_scale = rng.uniform(1 - variation, 1 + variation, size=len(net.load))
        net.load["p_mw"] = orig_load_p * load_scale
        net.load["q_mvar"] = orig_load_q * load_scale
        if len(net.sgen) > 0:
            sgen_scale = rng.uniform(1 - variation, 1 + variation, size=len(net.sgen))
            net.sgen["p_mw"] = orig_sgen_p * sgen_scale

        # Run AC power flow
        try:
            pp.runpp(net, algorithm="nr", init="flat", numba=True,
                     max_iteration=30)
        except pp.LoadflowNotConverged:
            n_failed += 1
            continue

        # Extract features
        pq = _compute_scheduled_injections(net, topo)
        vn = topo.vn_kv.numpy().reshape(-1, 1)
        x = np.hstack([bus_type_onehot, pq, vn])  # [N, 6]

        # Extract targets
        vm = net.res_bus.loc[topo.bus_indices, "vm_pu"].values.astype(np.float32)
        va = net.res_bus.loc[topo.bus_indices, "va_degree"].values.astype(np.float32)
        y = np.column_stack([vm, va])  # [N, 2]

        data = Data(
            x=torch.from_numpy(x),
            edge_index=topo.edge_index,
            edge_attr=topo.edge_attr,
            y=torch.from_numpy(y),
        )
        samples.append(data)

    # Restore original values
    net.load["p_mw"] = orig_load_p
    net.load["q_mvar"] = orig_load_q
    if len(net.sgen) > 0:
        net.sgen["p_mw"] = orig_sgen_p

    fail_rate = n_failed / n_samples
    logger.info(f"Generated {len(samples)}/{n_samples} samples "
                f"({n_failed} failed, {fail_rate:.1%} fail rate)")

    return samples, fail_rate


# ── Normalization ──────────────────────────────────────────────────────────────

@dataclass
class Normalizer:
    """Per-grid z-score normalizer for features and targets."""
    x_mean: torch.FloatTensor
    x_std: torch.FloatTensor
    y_mean: torch.FloatTensor
    y_std: torch.FloatTensor

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / (self.x_std + 1e-8)

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.y_std + 1e-8) + self.y_mean


def compute_normalizer(samples: list[Data]) -> Normalizer:
    """Compute z-score statistics from a list of Data objects."""
    all_x = torch.cat([s.x for s in samples], dim=0)
    all_y = torch.cat([s.y for s in samples], dim=0)
    return Normalizer(
        x_mean=all_x.mean(dim=0),
        x_std=all_x.std(dim=0),
        y_mean=all_y.mean(dim=0),
        y_std=all_y.std(dim=0),
    )


def apply_normalization(samples: list[Data], normalizer: Normalizer) -> list[Data]:
    """Apply z-score normalization to features and targets."""
    normalized = []
    for s in samples:
        data = Data(
            x=normalizer.normalize_x(s.x),
            edge_index=s.edge_index,
            edge_attr=s.edge_attr,
            y=normalizer.normalize_y(s.y),
        )
        normalized.append(data)
    return normalized


# ── Positional Encodings ──────────────────────────────────────────────────────

def compute_laplacian_pe(topo: GraphTopology, k: int = LAPLACIAN_PE_DIM) -> torch.FloatTensor:
    """Compute Laplacian positional encodings (k smallest non-trivial eigenvectors).

    Builds the normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2} and returns
    the eigenvectors corresponding to the k smallest non-zero eigenvalues.
    For disconnected graphs with fewer than k+1 eigenvalues, pads with zeros.

    Returns shape [N, k].
    """
    N = topo.num_nodes
    ei = topo.edge_index.numpy()
    src, dst = ei[0], ei[1]

    # Build sparse adjacency matrix (undirected edges already in edge_index)
    data = np.ones(len(src), dtype=np.float64)
    A = coo_matrix((data, (src, dst)), shape=(N, N)).tocsr()
    # Ensure symmetric (in case of duplicates, sum them)
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)  # binary adjacency

    # Degree matrix
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros(N, dtype=np.float64)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    from scipy.sparse import diags
    D_inv_sqrt = diags(deg_inv_sqrt)
    L = speye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    # Number of eigenvectors to request (k+1 to skip the trivial zero eigenvalue)
    num_eigs = min(k + 1, N - 1) if N > 1 else 0
    if num_eigs < 2:
        return torch.zeros(N, k, dtype=torch.float32)

    try:
        eigenvalues, eigenvectors = eigsh(L.tocsc(), k=num_eigs, which="SM", tol=1e-6)
    except Exception:
        return torch.zeros(N, k, dtype=torch.float32)

    # Sort by eigenvalue and skip the first (trivial, eigenvalue ≈ 0)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    # Skip the first eigenvector (constant, eigenvalue 0)
    pe = eigenvectors[:, 1:]  # shape [N, num_eigs-1]

    # Pad with zeros if we got fewer than k eigenvectors
    if pe.shape[1] < k:
        padding = np.zeros((N, k - pe.shape[1]), dtype=np.float64)
        pe = np.hstack([pe, padding])

    return torch.tensor(pe[:, :k], dtype=torch.float32)


def compute_random_walk_pe(topo: GraphTopology, k: int = RANDOM_WALK_PE_DIM) -> torch.FloatTensor:
    """Compute Random Walk positional encodings.

    Computes RW landing probabilities: [diag(A^1), diag(A^2), ..., diag(A^k)]
    where A is the row-normalized adjacency (transition) matrix.

    Returns shape [N, k].
    """
    N = topo.num_nodes
    ei = topo.edge_index.numpy()
    src, dst = ei[0], ei[1]

    # Build sparse adjacency matrix
    data = np.ones(len(src), dtype=np.float64)
    A = coo_matrix((data, (src, dst)), shape=(N, N)).tocsr()
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)

    # Row-normalize to get transition matrix
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.zeros(N, dtype=np.float64)
    nonzero = deg > 0
    deg_inv[nonzero] = 1.0 / deg[nonzero]
    from scipy.sparse import diags
    D_inv = diags(deg_inv)
    T = D_inv @ A  # transition matrix (row-stochastic)

    pe = np.zeros((N, k), dtype=np.float64)
    # T_power starts as identity, then we multiply by T each step
    T_power = T.copy()
    for i in range(k):
        pe[:, i] = T_power.diagonal()
        if i < k - 1:
            T_power = T_power @ T

    return torch.tensor(pe, dtype=torch.float32)


def compute_distance_from_slack_pe(topo: GraphTopology) -> torch.FloatTensor:
    """Compute distance-from-slack positional encoding via BFS.

    For each node, computes the shortest path distance to the nearest slack bus.
    Returns shape [N, 2] where:
      - column 0: normalized distance (distance / max_distance)
      - column 1: inverse distance (1 / distance, clipped to 1.0 for distance=0)
    """
    N = topo.num_nodes
    ei = topo.edge_index.numpy()

    # Build adjacency list
    adj = [[] for _ in range(N)]
    for idx in range(ei.shape[1]):
        s, d = int(ei[0, idx]), int(ei[1, idx])
        adj[s].append(d)

    # BFS from each slack bus
    dist = np.full(N, fill_value=np.inf, dtype=np.float64)
    for slack_bus in topo.ext_grid_buses:
        visited = np.full(N, fill_value=False)
        queue = deque()
        queue.append((int(slack_bus), 0))
        visited[int(slack_bus)] = True
        while queue:
            node, d = queue.popleft()
            dist[node] = min(dist[node], d)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, d + 1))

    # Handle unreachable nodes (set to max distance + 1)
    max_dist = dist[np.isfinite(dist)].max() if np.any(np.isfinite(dist)) else 1.0
    dist[~np.isfinite(dist)] = max_dist + 1
    max_dist = dist.max()

    # Normalized distance
    norm_dist = dist / max_dist if max_dist > 0 else np.zeros(N)

    # Inverse distance (clipped: 1/d, with d=0 → 1.0)
    inv_dist = np.zeros(N, dtype=np.float64)
    nonzero = dist > 0
    inv_dist[nonzero] = 1.0 / dist[nonzero]
    inv_dist[~nonzero] = 1.0  # slack bus itself

    pe = np.column_stack([norm_dist, inv_dist])
    return torch.tensor(pe, dtype=torch.float32)


def compute_positional_encodings(topo: GraphTopology, pe_type: str) -> torch.FloatTensor:
    """Dispatcher for positional encoding computation.

    Args:
        topo: Graph topology.
        pe_type: One of "laplacian", "random_walk", "distance_from_slack".

    Returns:
        Positional encoding tensor of shape [N, pe_dim].
    """
    if pe_type == "laplacian":
        return compute_laplacian_pe(topo)
    elif pe_type == "random_walk":
        return compute_random_walk_pe(topo)
    elif pe_type == "distance_from_slack":
        return compute_distance_from_slack_pe(topo)
    else:
        raise ValueError(f"Unknown PE type: {pe_type!r}. "
                         f"Choose from: laplacian, random_walk, distance_from_slack")


# ── Virtual Slack Node ────────────────────────────────────────────────────────

def add_virtual_slack_edges(data: Data, topo: GraphTopology) -> Data:
    """Add shortcut edges from slack bus(es) to all other buses.

    This reduces the effective graph diameter to at most 2 hops from any
    node to the slack bus, addressing the limited receptive field problem
    on radial LV feeders.

    Args:
        data: PyG Data object
        topo: GraphTopology with ext_grid_buses info

    Returns:
        New Data object with additional edges.
    """
    bus_to_idx = {b: i for i, b in enumerate(topo.bus_indices)}
    slack_indices = [bus_to_idx[b] for b in topo.ext_grid_buses if b in bus_to_idx]

    if not slack_indices:
        return data

    num_nodes = data.x.shape[0]
    new_src, new_dst = [], []

    for slack_idx in slack_indices:
        for i in range(num_nodes):
            if i == slack_idx:
                continue
            # Check if edge already exists
            edge_mask = (data.edge_index[0] == slack_idx) & (data.edge_index[1] == i)
            if not edge_mask.any():
                # Add bidirectional edges
                new_src.extend([slack_idx, i])
                new_dst.extend([i, slack_idx])

    if not new_src:
        return data

    new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
    # Virtual edges get zero impedance features (direct electrical connection)
    new_edge_attr = torch.zeros(len(new_src), data.edge_attr.shape[1])

    combined_edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    combined_edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

    return Data(
        x=data.x,
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        y=data.y,
    )


# ── Dataset Building ───────────────────────────────────────────────────────────

def build_dataset(grid_name: str, n_samples: int = NUM_SAMPLES,
                  seed: int = 42, force: bool = False):
    """Build and cache a PyG dataset for a single grid.

    Returns (train_data, val_data, test_data, normalizer).
    """
    cache_path = DATA_DIR / f"{grid_name}_seed{seed}.pt"
    if cache_path.exists() and not force:
        logger.info(f"Loading cached dataset: {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["train"], cached["val"], cached["test"], cached["normalizer"]

    logger.info(f"Building dataset for {grid_name} ({n_samples} samples, seed={seed})")

    net = load_simbench_net(grid_name)
    topo = build_graph_topology(net)

    # Generate samples (with fallback on high failure rate)
    samples, fail_rate = generate_samples(net, topo, n_samples, seed, LOAD_VARIATION)
    if fail_rate > MAX_FAIL_RATE:
        logger.warning(f"High fail rate {fail_rate:.1%} for {grid_name}, "
                       f"retrying with ±{FALLBACK_VARIATION*100:.0f}% variation")
        samples, fail_rate = generate_samples(
            net, topo, n_samples, seed, FALLBACK_VARIATION
        )

    # Compute normalization on training split only
    n_train = int(len(samples) * TRAIN_RATIO)
    n_val = int(len(samples) * VAL_RATIO)
    train_raw = samples[:n_train]
    val_raw = samples[n_train:n_train + n_val]
    test_raw = samples[n_train + n_val:]

    normalizer = compute_normalizer(train_raw)
    train_data = apply_normalization(train_raw, normalizer)
    val_data = apply_normalization(val_raw, normalizer)
    test_data = apply_normalization(test_raw, normalizer)

    # Cache
    torch.save({
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "normalizer": normalizer,
        "topology": topo,
        "grid_name": grid_name,
    }, cache_path)
    logger.info(f"Cached dataset to {cache_path} "
                f"(train={len(train_data)}, val={len(val_data)}, test={len(test_data)})")

    return train_data, val_data, test_data, normalizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Quick test: build dataset for smallest grid
    train, val, test, norm = build_dataset("LV_rural1", n_samples=100, seed=42, force=True)
    sample = train[0]
    print(f"Sample shapes: x={sample.x.shape}, edge_index={sample.edge_index.shape}, "
          f"edge_attr={sample.edge_attr.shape}, y={sample.y.shape}")
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Normalizer y_mean={norm.y_mean}, y_std={norm.y_std}")
