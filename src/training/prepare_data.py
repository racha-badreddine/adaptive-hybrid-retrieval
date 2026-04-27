"""
prepare_data.py

Loads features.json from all processed datasets, filters to informative queries,
builds a combined DataFrame, and saves to results/combined_training_data.csv.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from src.features.query_features import FEATURE_NAMES


# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "combined_training_data.csv"


def load_features_for_dataset(
    dataset_name: str,
    features_filename: str = "features.json",
) -> List[Dict]:
    """
    Load a features JSON file for one dataset.  Returns a list of flat dicts,
    one per query, with keys: query_id, dataset, oracle_alpha, oracle_ndcg,
    is_informative, + all 17 feature keys.

    Args:
        dataset_name: name of the dataset subdirectory under results/
        features_filename: filename to load (default "features.json";
            pass "features_bge.json" for the BGE variant)
    """
    path = RESULTS_DIR / dataset_name / features_filename
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for qid, entry in data["queries"].items():
        feats = entry.get("features", {})
        # Guard: must have all 17 features
        if not all(fn in feats for fn in FEATURE_NAMES):
            continue
        rows.append(
            {
                "query_id": qid,
                "dataset": dataset_name,
                "oracle_alpha": entry.get("oracle_alpha"),
                "oracle_ndcg": entry.get("oracle_ndcg"),
                "is_informative": entry.get("is_informative", False),
                **{fn: feats[fn] for fn in FEATURE_NAMES},
            }
        )
    return rows


def load_all_datasets(
    dataset_names: List[str],
    features_filename: str = "features.json",
) -> pd.DataFrame:
    """Load and concatenate all datasets into one DataFrame."""
    all_rows = []
    per_dataset_counts = {}

    for ds in dataset_names:
        rows = load_features_for_dataset(ds, features_filename=features_filename)
        per_dataset_counts[ds] = len(rows)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df, per_dataset_counts


def print_alpha_distribution(alphas: np.ndarray, label: str = "Oracle alpha") -> None:
    bins = [
        ("0.0-0.2", 0.0, 0.2),
        ("0.2-0.4", 0.2, 0.4),
        ("0.4-0.6", 0.4, 0.6),
        ("0.6-0.8", 0.6, 0.8),
        ("0.8-1.0", 0.8, 1.0),
    ]
    print(f"\n  {label} distribution:")
    total = len(alphas)
    for label_str, lo, hi in bins:
        if lo == 0.0:
            count = np.sum((alphas >= lo) & (alphas <= hi))
        else:
            count = np.sum((alphas > lo) & (alphas <= hi))
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"    [{label_str}]: {count:5d}  ({pct:.1f}%)")


def prepare_data(
    dataset_names: List[str] | None = None,
    informative_only: bool = True,
    features_filename: str = "features.json",
    output_csv_name: str = "combined_training_data.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point: load all datasets, optionally filter to informative
    queries, print stats, save combined CSV.

    Args:
        dataset_names: list of dataset names; auto-discovered if None
        informative_only: (unused; kept for API compat — always filters)
        features_filename: filename to load per dataset (default "features.json";
            pass "features_bge.json" for the BGE variant)
        output_csv_name: filename for the combined CSV saved under results/

    Returns:
        (df_all, df_informative) — full and filtered DataFrames.
    """
    if dataset_names is None:
        # Auto-discover: any results/{name}/{features_filename} that exists
        dataset_names = sorted(
            p.parent.name
            for p in RESULTS_DIR.glob(f"*/{features_filename}")
            if p.parent.name != "splits"
        )

    print(f"\nLoading features from {len(dataset_names)} datasets: {dataset_names}")
    print(f"  Features file: {features_filename}")
    df_all, per_dataset_counts = load_all_datasets(dataset_names, features_filename=features_filename)

    if df_all.empty:
        print("ERROR: No data loaded. Check that features.json files exist.")
        return df_all, df_all

    df_informative = df_all[df_all["is_informative"] == True].copy()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(df_all)
    n_inf = len(df_informative)
    pct_inf = 100.0 * n_inf / total if total > 0 else 0.0

    print(f"\nTotal queries across all datasets: {total}")
    print(f"Informative queries: {n_inf} ({pct_inf:.1f}%)")
    print("\nPer-dataset breakdown:")
    for ds in dataset_names:
        ds_all = len(df_all[df_all["dataset"] == ds])
        ds_inf = len(df_informative[df_informative["dataset"] == ds])
        pct = 100.0 * ds_inf / ds_all if ds_all > 0 else 0.0
        print(f"  {ds:<20}: {ds_all:5d} total   {ds_inf:5d} informative ({pct:.1f}%)")

        if ds_inf < 50:
            print(f"    [FLAG] {ds} has fewer than 50 informative queries.")

    # ── Feature statistics ────────────────────────────────────────────────────
    df_feats = df_informative[FEATURE_NAMES]
    print("\nFeature statistics (informative queries):")
    for fn in FEATURE_NAMES:
        col = df_feats[fn]
        print(
            f"  {fn:<25}  mean={col.mean():.4f}  std={col.std():.4f}"
            f"  min={col.min():.4f}  max={col.max():.4f}"
        )

    # ── Alpha distribution ────────────────────────────────────────────────────
    alphas = df_informative["oracle_alpha"].dropna().values
    print_alpha_distribution(alphas, label="Oracle alpha (informative queries)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / output_csv_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_informative.to_csv(out_csv, index=False)
    print(f"\nCombined training data saved -> {out_csv}")
    print(f"  Shape: {df_informative.shape}  ({len(df_informative)} rows x {len(df_informative.columns)} cols)")

    return df_all, df_informative


if __name__ == "__main__":
    df_all, df_inf = prepare_data()
