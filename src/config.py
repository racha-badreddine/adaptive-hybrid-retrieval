"""
Central configuration for the adaptive hybrid retrieval project.
Import these constants from here; never hardcode them in individual scripts.
"""

# Final 5-dataset set for CIKM 2026 analysis.
# trec-covid was dropped (see DROPPED.md); MIRAGE covers the medical signal in Week 2.
DATASETS = ["scifact", "arguana", "nfcorpus", "fiqa", "scidocs"]

# Canonical 21-point alpha grid: 0.00, 0.05, 0.10, ..., 1.00
# Used for both oracle grid search and static-best validation sweep.
# Feature-based regressors predict a continuous alpha and ignore this grid.
ALPHA_GRID = [round(i / 20.0, 2) for i in range(21)]
