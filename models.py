"""GNN architectures for AC power flow approximation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, NNConv

from configs import (
    DROPEDGE_P,
    DROPOUT,
    GAT_HEADS,
    HIDDEN_DIM,
    NUM_EDGE_FEATURES,
    NUM_LAYERS,
    NUM_NODE_FEATURES,
    NUM_TARGETS,
)


class MLP(nn.Module):
    """Multi-Layer Perceptron — non-graph baseline, processes each node independently."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        return self.net(x)


class GCN(nn.Module):
    """Graph Convolutional Network — simplest baseline, no edge features."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


class GAT(nn.Module):
    """Graph Attention Network — multi-head attention with edge features."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 heads=GAT_HEADS, edge_dim=NUM_EDGE_FEATURES):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim // heads, heads=heads,
                    edge_dim=edge_dim, concat=True
                ))
            else:
                # Last layer: single head, full hidden_dim output
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim, heads=1,
                    edge_dim=edge_dim, concat=False
                ))
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


class GraphSAGE(nn.Module):
    """GraphSAGE — mean aggregation, no edge features."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


class MPNN(nn.Module):
    """Message Passing Neural Network — edge NN maps impedance to message weights.

    Uses a reduced inner dimension for stability: edge features are mapped to
    hidden_dim * msg_dim, where msg_dim << hidden_dim. Includes batch
    normalization and residual connections.
    """

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 edge_dim=NUM_EDGE_FEATURES, msg_dim=16):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, msg_dim * hidden_dim),
            )
            self.convs.append(NNConv(hidden_dim, msg_dim, edge_nn, aggr="mean"))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.msg_proj = nn.Linear(msg_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x_msg = conv(x, edge_index, edge_attr)
            x = residual + self.msg_proj(x_msg)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


class ResidualGraphSAGE(nn.Module):
    """GraphSAGE with pre-norm residual connections for deeper architectures."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(norm(x), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


class JKGraphSAGE(nn.Module):
    """GraphSAGE with Jumping Knowledge aggregation across all layers."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 jk_mode="max"):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.jk_mode = jk_mode
        if jk_mode == "cat":
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(norm(x), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        if self.jk_mode == "max":
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == "cat":
            x = self.jk_proj(torch.cat(layer_outputs, dim=-1))
        return self.output_proj(x)


class DropEdgeGraphSAGE(nn.Module):
    """GraphSAGE with DropEdge regularization and residual connections."""

    def __init__(self, in_dim=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                 out_dim=NUM_TARGETS, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 drop_edge_p=DROPEDGE_P):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p

    def _drop_edges(self, edge_index):
        """Randomly drop a fraction of edges during training."""
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges, device=edge_index.device) >= self.drop_edge_p
        return edge_index[:, mask]

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            ei = self._drop_edges(edge_index) if self.training else edge_index
            x = x + conv(norm(x), ei)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)


# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "MLP": MLP,
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
    "MPNN": MPNN,
    "ResidualGraphSAGE": ResidualGraphSAGE,
    "JKGraphSAGE": JKGraphSAGE,
    "DropEdgeGraphSAGE": DropEdgeGraphSAGE,
}


def create_model(name: str, **kwargs) -> nn.Module:
    """Create a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Verify all models on dummy data
    from configs import NUM_NODE_FEATURES, NUM_EDGE_FEATURES

    num_nodes, num_edges = 15, 28
    x = torch.randn(num_nodes, NUM_NODE_FEATURES)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, NUM_EDGE_FEATURES)

    for name in MODEL_REGISTRY:
        model = create_model(name)
        out = model(x, edge_index, edge_attr)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:<12} output={out.shape}  params={n_params:,}")
