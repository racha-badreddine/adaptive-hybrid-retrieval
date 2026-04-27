"""
train_adaptive.py

Trains and evaluates four adaptive alpha-prediction model variants:
  1. Ridge Regression (RidgeCV)
  2. MLP-Small  (1 hidden layer, 32 units)
  3. MLP-Medium (2 hidden layers, 64-32 units)
  4. Heuristic baselines (alpha=0.5, score_ratio, per-dataset best)

Returns trained models, scalers, and per-query predictions for the test set.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.features.query_features import FEATURE_NAMES

RESULTS_DIR = Path("results")


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_arrays(
    df: pd.DataFrame, ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Slice df to the given query IDs and return (X, y, sub_df).
    X has shape (n, 17); y is the oracle_alpha column.
    """
    sub = df[df["query_id"].isin(ids)].copy()
    X = sub[FEATURE_NAMES].values.astype(float)
    y = sub["oracle_alpha"].values.astype(float)
    return X, y, sub


# ── Heuristics ────────────────────────────────────────────────────────────────

def heuristic_constant(n: int, alpha: float = 0.5) -> np.ndarray:
    return np.full(n, alpha)


def heuristic_score_ratio(X: np.ndarray) -> np.ndarray:
    """alpha = score_ratio / (1 + score_ratio), clamped [0, 1]."""
    sr_idx = FEATURE_NAMES.index("score_ratio")
    sr = X[:, sr_idx]
    return np.clip(sr / (1.0 + np.abs(sr)), 0.0, 1.0)


def heuristic_per_dataset_best(
    sub_df: pd.DataFrame,
    best_static_by_dataset: Dict[str, float],
) -> np.ndarray:
    """Assign the best static alpha of the originating dataset to each query."""
    return sub_df["dataset"].map(best_static_by_dataset).fillna(0.5).values


# ── Training ──────────────────────────────────────────────────────────────────

def train_ridge(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[RidgeCV, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(X_scaled, y_train)
    print(f"  Ridge best regularization: {model.alpha_:.4f}")
    return model, scaler


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    hidden_layer_sizes: Tuple,
    label: str = "MLP",
) -> MLPRegressor:
    X_scaled = scaler.transform(X_train)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        learning_rate_init=0.001,
        alpha=0.01,
    )
    model.fit(X_scaled, y_train)
    print(f"  {label}: converged={model.n_iter_ < model.max_iter}, "
          f"n_iter={model.n_iter_}, best_val_loss={model.best_validation_score_:.6f}")
    return model


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(
    ridge: RidgeCV,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    coefs = ridge.coef_
    fi_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "coef_signed": coefs,
            "coef_abs": np.abs(coefs),
        }
    ).sort_values("coef_abs", ascending=False)

    print("\nFeature Importance (Ridge |coefficient|, features are scaled):")
    for _, row in fi_df.iterrows():
        print(
            f"  {row['feature']:<25}  coef={row['coef_signed']:+.4f}  "
            f"|coef|={row['coef_abs']:.4f}"
        )

    # Permutation importance on test set
    perm = permutation_importance(
        ridge, X_test_scaled, y_test, n_repeats=30, random_state=42, scoring="r2"
    )
    perm_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_mean", ascending=False)

    print("\nPermutation Importance (Ridge, test set, 30 repeats):")
    for _, row in perm_df.iterrows():
        print(f"  {row['feature']:<25}  {row['perm_mean']:.4f} +/- {row['perm_std']:.4f}")

    return fi_df


# ── Main train function ───────────────────────────────────────────────────────

def train_all_models(
    df: pd.DataFrame,
    split_a: Dict,
    best_static_by_dataset: Dict[str, float],
) -> Dict:
    """
    Train Ridge + MLP models using Strategy A splits.

    Returns a results dict with:
      - trained model objects
      - per-query predictions for val and test sets
      - alpha-prediction metrics (MSE, MAE, Pearson r)
    """
    train_ids = split_a["train_ids"]
    val_ids = split_a["val_ids"]
    test_ids = split_a["test_ids"]

    X_train, y_train, df_train = build_arrays(df, train_ids)
    X_val, y_val, df_val = build_arrays(df, val_ids)
    X_test, y_test, df_test = build_arrays(df, test_ids)

    print(f"\nSplit A: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print("\n[Ridge] Training...")
    ridge, scaler = train_ridge(X_train, y_train)

    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    ridge_val_pred = np.clip(ridge.predict(X_val_sc), 0.0, 1.0)
    ridge_test_pred = np.clip(ridge.predict(X_test_sc), 0.0, 1.0)

    # ── MLP-Small ─────────────────────────────────────────────────────────────
    print("\n[MLP-Small] Training...")
    mlp_s = train_mlp(X_train, y_train, scaler, (32,), label="MLP-Small")
    mlp_s_val_pred = np.clip(mlp_s.predict(X_val_sc), 0.0, 1.0)
    mlp_s_test_pred = np.clip(mlp_s.predict(X_test_sc), 0.0, 1.0)

    # ── MLP-Medium ────────────────────────────────────────────────────────────
    print("\n[MLP-Medium] Training...")
    mlp_m = train_mlp(X_train, y_train, scaler, (64, 32), label="MLP-Medium")
    mlp_m_val_pred = np.clip(mlp_m.predict(X_val_sc), 0.0, 1.0)
    mlp_m_test_pred = np.clip(mlp_m.predict(X_test_sc), 0.0, 1.0)

    # ── Heuristics ────────────────────────────────────────────────────────────
    h_const_pred = heuristic_constant(len(X_test))
    h_ratio_pred = heuristic_score_ratio(X_test)
    h_ds_pred = heuristic_per_dataset_best(df_test, best_static_by_dataset)

    # ── Alpha-prediction metrics ──────────────────────────────────────────────
    def alpha_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict:
        mse = float(np.mean((pred - truth) ** 2))
        mae = float(np.mean(np.abs(pred - truth)))
        corr = float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 1 else 0.0
        return {"mse": mse, "mae": mae, "pearson_r": corr}

    print("\n--- Alpha prediction metrics (test set) ---")
    model_preds = {
        "ridge": ridge_test_pred,
        "mlp_small": mlp_s_test_pred,
        "mlp_medium": mlp_m_test_pred,
        "heuristic_const": h_const_pred,
        "heuristic_ratio": h_ratio_pred,
        "heuristic_per_dataset": h_ds_pred,
    }

    alpha_metrics_dict = {}
    for name, pred in model_preds.items():
        m = alpha_metrics(pred, y_test)
        alpha_metrics_dict[name] = m
        print(
            f"  {name:<25}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  "
            f"Pearson r={m['pearson_r']:.4f}"
        )

    # ── Feature importance (Ridge only) ──────────────────────────────────────
    fi_df = print_feature_importance(ridge, X_test_sc, y_test)

    return {
        "models": {
            "ridge": ridge,
            "mlp_small": mlp_s,
            "mlp_medium": mlp_m,
        },
        "scaler": scaler,
        "predictions": {
            "test": model_preds,
            "val_ridge": ridge_val_pred,
            "val_mlp_small": mlp_s_val_pred,
            "val_mlp_medium": mlp_m_val_pred,
        },
        "df_test": df_test,
        "df_val": df_val,
        "y_test": y_test,
        "y_val": y_val,
        "X_test": X_test,
        "X_test_scaled": X_test_sc,
        "alpha_metrics": alpha_metrics_dict,
        "feature_importance": fi_df,
    }


# ── Cross-dataset training ────────────────────────────────────────────────────

def train_cross_dataset(
    df: pd.DataFrame,
    split_b: Dict,
) -> Dict:
    """
    For each held-out dataset in split_b, train Ridge on the other datasets
    and return predictions for the held-out test set.
    """
    results = {}

    for held_out, ids in split_b.items():
        train_ids = ids["train_ids"]
        test_ids = ids["test_ids"]

        X_train, y_train, _ = build_arrays(df, train_ids)
        X_test, y_test, df_test = build_arrays(df, test_ids)

        if len(X_train) < 5:
            print(f"  [SKIP] {held_out}: not enough training data ({len(X_train)} rows)")
            continue

        # Ridge only for cross-dataset (faster, interpretable)
        ridge, scaler = train_ridge(X_train, y_train)
        X_test_sc = scaler.transform(X_test)
        preds = np.clip(ridge.predict(X_test_sc), 0.0, 1.0)

        results[held_out] = {
            "model": ridge,
            "scaler": scaler,
            "predictions": preds,
            "y_test": y_test,
            "df_test": df_test,
        }
        print(f"  [{held_out}] trained on {len(X_train)}, test on {len(X_test)}")

    return results


if __name__ == "__main__":
    from src.training.prepare_data import prepare_data
    from src.training.split_data import create_splits

    _, df_inf = prepare_data()
    split_a, split_b = create_splits(df_inf)

    # Dummy best_static_by_dataset for standalone testing
    best_static = {ds: 0.5 for ds in df_inf["dataset"].unique()}
    results = train_all_models(df_inf, split_a, best_static)
    print("\nTraining complete.")
