"""Configuration for GNN Power Flow Benchmark on Distribution Networks."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "trained_models"
FIGURES_DIR = PROJECT_ROOT / "figures"

for d in [DATA_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)

# ── SimBench Grid Codes ────────────────────────────────────────────────────────
GRID_CODES = {
    # MV grids (4)
    "MV_rural": "1-MV-rural--0-sw",
    "MV_semiurb": "1-MV-semiurb--0-sw",
    "MV_urban": "1-MV-urban--0-sw",
    "MV_comm": "1-MV-comm--0-sw",
    # LV grids (6)
    "LV_rural1": "1-LV-rural1--0-sw",
    "LV_rural2": "1-LV-rural2--0-sw",
    "LV_rural3": "1-LV-rural3--0-sw",
    "LV_semiurb4": "1-LV-semiurb4--0-sw",
    "LV_semiurb5": "1-LV-semiurb5--0-sw",
    "LV_urban6": "1-LV-urban6--0-sw",
}

MV_GRIDS = [k for k in GRID_CODES if k.startswith("MV")]
LV_GRIDS = [k for k in GRID_CODES if k.startswith("LV")]

# ── Data Generation ────────────────────────────────────────────────────────────
NUM_SAMPLES = 2_000
LOAD_VARIATION = 0.30          # ±30% uniform scaling of loads/sgens
MAX_FAIL_RATE = 0.10           # if >10% fail, reduce variation
FALLBACK_VARIATION = 0.20      # reduced variation fallback
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Node Features (6) ─────────────────────────────────────────────────────────
# bus_type one-hot (3): slack, PV, PQ
# P_scheduled (MW), Q_scheduled (Mvar), Vn_kv
NUM_NODE_FEATURES = 6

# ── Edge Features (3) ─────────────────────────────────────────────────────────
# r_ohm_total, x_ohm_total, b_us_total
NUM_EDGE_FEATURES = 3

# ── Targets (2) ───────────────────────────────────────────────────────────────
# vm_pu, va_degree
NUM_TARGETS = 2

# ── GNN Hyperparameters (fixed across all models) ─────────────────────────────
HIDDEN_DIM = 64
NUM_LAYERS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 25

# ── Model Names ────────────────────────────────────────────────────────────────
MODEL_NAMES = ["MLP", "GCN", "GAT", "GraphSAGE", "MPNN"]

# ── Experiment Seeds ───────────────────────────────────────────────────────────
EXPERIMENT_SEEDS = [42, 123, 456]

# ── GAT-specific ───────────────────────────────────────────────────────────────
GAT_HEADS = 4

# ══════════════════════════════════════════════════════════════════════════════
# Paper v2: Physics-Informed GNN experiments
# ══════════════════════════════════════════════════════════════════════════════

# ── E1: Physics-Informed Loss ────────────────────────────────────────────────
PHYSICS_LAMBDAS = [10.0, 100.0, 1000.0]

# ── E2: Positional Encodings ────────────────────────────────────────────────
PE_TYPES = ["laplacian", "random_walk", "distance_from_slack"]
LAPLACIAN_PE_DIM = 8       # number of eigenvectors
RANDOM_WALK_PE_DIM = 16    # number of RW steps

# ── E3: Deeper GNNs ─────────────────────────────────────────────────────────
DEEP_DEPTHS = [4, 8, 16, 32]
DEEP_TECHNIQUES = ["residual", "jknet", "dropedge"]
DROPEDGE_P = 0.2           # fraction of edges to drop during training
# Pilot grids for depth sweep (1 MV + 1 LV with large diameter)
PILOT_GRIDS = ["MV_rural", "LV_rural2"]

# ── E4: Virtual Slack Node ──────────────────────────────────────────────────
VIRTUAL_NODE_EDGE_ATTR = "zero"  # "zero" or "learned"

# ── Experiment result subdirectories ────────────────────────────────────────
for subdir in ["baseline", "e1_physics", "e2_pe", "e3_deep", "e4_virtual", "e5_combined"]:
    (RESULTS_DIR / subdir).mkdir(exist_ok=True)
