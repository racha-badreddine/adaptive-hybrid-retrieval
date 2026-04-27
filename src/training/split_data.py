"""
split_data.py

Two split strategies for the adaptive model:

Strategy A -- Per-dataset split (primary):
  70% train / 15% val / 15% test, stratified by oracle-alpha bins,
  applied independently per dataset then combined.

Strategy B -- Leave-one-dataset-out (cross-dataset transfer):
  Train on 5 datasets, test on the held-out 6th.  Repeated for every
  dataset as the held-out set.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path("results")
SPLITS_DIR = RESULTS_DIR / "splits"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _alpha_bins(alphas: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Map oracle alpha values to integer bin labels [0, n_bins-1]."""
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    return np.digitize(alphas, edges) - 1


def _split_indices(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/val/test split on df.
    Returns (train_idx, val_idx, test_idx) as positional index arrays into df.
    """
    alphas = df["oracle_alpha"].values
    bins = _alpha_bins(alphas)
    all_idx = np.arange(len(df))

    # Clip bin values so every sample has a valid stratum label
    unique_bins, bin_counts = np.unique(bins, return_counts=True)
    # For bins with a single sample, stratification breaks; fall back to non-stratified
    rare_bins = set(unique_bins[bin_counts < 2])
    stratify_bins = np.where(np.isin(bins, list(rare_bins)), -1, bins)
    use_stratify = len(np.unique(stratify_bins)) > 1

    # First split: trainval vs test
    try:
        trainval_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_frac,
            stratify=stratify_bins if use_stratify else None,
            random_state=random_state,
        )
    except ValueError:
        trainval_idx, test_idx = train_test_split(
            all_idx, test_size=test_frac, random_state=random_state
        )

    # Second split: train vs val (from trainval only)
    val_frac_adj = val_frac / (1.0 - test_frac)
    tv_bins = stratify_bins[trainval_idx]
    unique_tv, tv_counts = np.unique(tv_bins, return_counts=True)
    use_stratify_tv = len(unique_tv) > 1 and all(c >= 2 for c in tv_counts)

    try:
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_frac_adj,
            stratify=tv_bins if use_stratify_tv else None,
            random_state=random_state,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac_adj, random_state=random_state
        )

    return train_idx, val_idx, test_idx


# ── Strategy A ────────────────────────────────────────────────────────────────

def strategy_a(df: pd.DataFrame, random_state: int = 42) -> Dict:
    """
    Per-dataset stratified 70/15/15 split.
    Returns {train_uids, val_uids, test_uids} where each uid is "dataset::query_id"
    to avoid false positives from numeric IDs shared across datasets.
    For backward compat also returns train_ids/val_ids/test_ids (plain query_ids).
    """
    train_uids, val_uids, test_uids = [], [], []

    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds].reset_index(drop=True)
        if len(sub) < 10:
            train_uids.extend([f"{ds}::{q}" for q in sub["query_id"].tolist()])
            print(f"  [WARN] {ds}: only {len(sub)} queries, all assigned to train.")
            continue

        tr_idx, val_idx, te_idx = _split_indices(sub, random_state=random_state)

        train_uids.extend([f"{ds}::{q}" for q in sub.iloc[tr_idx]["query_id"].tolist()])
        val_uids.extend([f"{ds}::{q}" for q in sub.iloc[val_idx]["query_id"].tolist()])
        test_uids.extend([f"{ds}::{q}" for q in sub.iloc[te_idx]["query_id"].tolist()])

        print(
            f"  {ds:<20}: {len(tr_idx):4d} train / "
            f"{len(val_idx):3d} val / {len(te_idx):3d} test"
        )

    _verify_no_overlap(train_uids, val_uids, test_uids)

    # Also expose plain query_id lists (without dataset prefix) for code that
    # filters df by query_id. Since query_ids ARE unique within a dataset, the
    # per-dataset split guarantees no within-dataset overlap.
    def _qids(uids):
        return [u.split("::", 1)[1] for u in uids]

    return {
        "train_ids": _qids(train_uids),
        "val_ids":   _qids(val_uids),
        "test_ids":  _qids(test_uids),
        "train_uids": train_uids,
        "val_uids":   val_uids,
        "test_uids":  test_uids,
    }


# ── Strategy B ────────────────────────────────────────────────────────────────

def strategy_b(df: pd.DataFrame) -> Dict:
    """
    Leave-one-dataset-out cross-dataset transfer splits.
    For each held-out dataset: train on all others, test on held-out.
    Stores plain query_ids since each split stays within a dataset.
    """
    datasets = sorted(df["dataset"].unique().tolist())
    splits = {}

    for held_out in datasets:
        train_df = df[df["dataset"] != held_out]
        test_df  = df[df["dataset"] == held_out]

        splits[held_out] = {
            "train_ids": train_df["query_id"].tolist(),
            "test_ids":  test_df["query_id"].tolist(),
        }

        print(
            f"  held-out={held_out:<20}: "
            f"{len(train_df):5d} train  {len(test_df):5d} test"
        )

    return splits


# ── Verification ──────────────────────────────────────────────────────────────

def _verify_no_overlap(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    s_tr = set(train_ids)
    s_val = set(val_ids)
    s_te = set(test_ids)

    tv_overlap = s_tr & s_val
    tt_overlap = s_tr & s_te
    vt_overlap = s_val & s_te

    if tv_overlap or tt_overlap or vt_overlap:
        raise AssertionError(
            f"Split leakage detected: "
            f"train-val overlap={len(tv_overlap)}, "
            f"train-test overlap={len(tt_overlap)}, "
            f"val-test overlap={len(vt_overlap)}"
        )
    print(
        f"\n  [OK] No leakage: "
        f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test  "
        f"(total={len(train_ids)+len(val_ids)+len(test_ids)})"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def create_splits(
    df: pd.DataFrame,
    random_state: int = 42,
    suffix: str = "",
) -> Tuple[Dict, Dict]:
    """
    Run both split strategies and save to results/splits/.

    Args:
        df: informative queries DataFrame
        random_state: RNG seed for reproducibility
        suffix: optional suffix appended to the saved filenames, e.g. "_bge"
            produces per_dataset_split_bge.json / cross_dataset_splits_bge.json.
            Leave empty for the default MiniLM run.

    Returns (strategy_a_split, strategy_b_splits).
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Strategy A: Per-dataset 70/15/15 split ---")
    split_a = strategy_a(df, random_state=random_state)

    print("\n--- Strategy B: Leave-one-dataset-out ---")
    split_b = strategy_b(df)

    # Save — use suffix to keep BGE and MiniLM splits separate
    path_a = SPLITS_DIR / f"per_dataset_split{suffix}.json"
    path_b = SPLITS_DIR / f"cross_dataset_splits{suffix}.json"

    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(split_a, f, indent=2)
    print(f"\nStrategy A split saved -> {path_a}")

    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(split_b, f, indent=2)
    print(f"Strategy B splits saved -> {path_b}")

    return split_a, split_b


if __name__ == "__main__":
    from src.training.prepare_data import prepare_data

    _, df_inf = prepare_data()
    create_splits(df_inf)
