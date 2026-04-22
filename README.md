# Bridging the MV/LV Gap: Virtual Slack Nodes and Positional Encodings for GNN-Based Power Flow

Code and data for the paper **"Bridging the MV/LV Gap: Virtual Slack Nodes and Positional Encodings for GNN-Based Power Flow on Radial Distribution Networks"** by D. Voitekh and A. Tymoshenko.

## What this is

A GNN surrogate for AC power flow on SimBench distribution grids. Combines three techniques to eliminate the MV/LV accuracy gap:

1. **Virtual slack edges** — bidirectional shortcut connections from the slack bus to every node (reduces effective graph diameter to 2).
2. **Random walk positional encodings** (k=16) — topology-aware node features.
3. **Residual GraphSAGE** with 8 layers — deep architecture without oversmoothing.

On 10 SimBench grids × 3 seeds, the combined model reduces voltage-magnitude MAE by 43.8% (Wilcoxon p=0.002) and renders the LV/MV gap statistically non-significant (p=0.182).

## Requirements

- Python 3.12
- PyTorch 2.2
- PyTorch Geometric 2.5
- pandapower 2.14
- simbench
- scipy, numpy, matplotlib

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Generate data for all grids (≈30 min on M1 Pro CPU)
python data_generation.py --all

# 2. Run all experiments (baseline, E1 physics, E2 PE, E3 depth, E4 virtual, E5 combined)
python run_all_experiments.py --experiment all --seeds 42 123 456

# 3. Analyze and generate figures
python analyze_results.py
python compute_correlations.py
python generate_paper_figures.py
```

## Repository layout

- `configs.py` — grids, hyperparameters, seeds, paths
- `data_generation.py` — SimBench → pandapower NR → PyG Data objects
- `models.py` — GCN, GAT, GraphSAGE, ResidualGraphSAGE, MPNN, MLP
- `train.py` — AdamW + OneCycleLR + early stopping
- `evaluate.py` — MAE/RMSE/MaxAE + GNN vs NR timing
- `run_all_experiments.py` — full experiment runner
- `analyze_results.py` — statistical tests, comparisons
- `compute_correlations.py` — Spearman/Pearson correlations between grid structure and MAE
- `analyze_physics_loss.py` — physics-loss ineffectiveness quantification
- `generate_paper_figures.py` — PDF figures
- `results/` — raw JSON results (baseline, e1_physics, e2_pe, e3_deep, e4_virtual, e5_combined)
- `paper/` — LaTeX source

## Critical SimBench/pandapower fixes

The following corrections are essential for NR convergence on the full set of grids:

1. Transformer shift degrees set to zero (Yzn5 divergence fix)
2. `pandapower.create_continuous_bus_index(net)` (singular Jacobian fix)
3. NaN voltage-dependent load parameters replaced with zero
4. All switches removed: `net.switch.drop(net.switch.index)`

## License

MIT — see `LICENSE`.

## Citation

To be added upon acceptance.
