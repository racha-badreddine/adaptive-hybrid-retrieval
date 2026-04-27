"""
evaluate_adaptive.py

Evaluates adaptive alpha-prediction models on retrieval metrics (Steps 7-11):
  Step 7:  Retrieval metrics for all models vs baselines
  Step 8:  Statistical significance testing
  Step 9:  Feature importance (summarized from training)
  Step 10: Cross-dataset transfer (Strategy B)
  Step 11: Efficiency benchmarking

This module exposes run_full_evaluation() which is called from
run_adaptive_pipeline.py with pre-loaded retrieval data.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats

from beir.retrieval.evaluation import EvaluateRetrieval

from src.retrieval.hybrid_static import fuse_single_query, fuse_static
from src.retrieval.rrf import reciprocal_rank_fusion
from src.features.query_features import FEATURE_NAMES, extract_query_features, compute_idf_dict
from src.training.prepare_data import prepare_data
from src.training.split_data import create_splits
from src.training.train_adaptive import (
    train_all_models,
    train_cross_dataset,
    heuristic_constant,
    heuristic_score_ratio,
    heuristic_per_dataset_best,
    build_arrays,
)

RESULTS_DIR = Path("results")


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def fuse_for_queries(
    qids: List[str],
    bm25_results: Dict,
    dense_results: Dict,
    alpha_by_qid: Dict[str, float],
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Apply per-query alpha fusion."""
    return {
        qid: fuse_single_query(
            bm25_results_for_query=bm25_results[qid],
            dense_results_for_query=dense_results.get(qid, {}),
            alpha=float(alpha_by_qid.get(qid, 0.5)),
            top_k=top_k,
        )
        for qid in qids
        if qid in bm25_results
    }


def evaluate_results(qrels: Dict, results: Dict) -> Dict[str, float]:
    """Return NDCG@10, Recall@10, MAP@10, P@10, MRR@10."""
    if not results:
        return {"NDCG@10": 0.0, "Recall@10": 0.0, "MAP@10": 0.0, "P@10": 0.0, "MRR@10": 0.0}

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10])

    try:
        mrr_result = EvaluateRetrieval.evaluate_custom(qrels, results, [10], metric="mrr")
        # evaluate_custom returns a tuple; first element is the metric dict
        mrr_d = mrr_result[0] if isinstance(mrr_result, tuple) else mrr_result
        mrr = next((v for k, v in mrr_d.items() if "10" in str(k)), 0.0)
    except Exception:
        mrr = 0.0

    return {
        "NDCG@10":    ndcg.get("NDCG@10", 0.0),
        "Recall@10":  recall.get("Recall@10", 0.0),
        "MAP@10":     _map.get("MAP@10", 0.0),
        "P@10":       precision.get("P@10", 0.0),
        "MRR@10":     mrr,
    }


def per_query_ndcg(qrels: Dict, results: Dict) -> Dict[str, float]:
    """NDCG@10 for each query individually."""
    out = {}
    for qid, res in results.items():
        if qid not in qrels:
            continue
        ndcg_d, _, _, _ = EvaluateRetrieval.evaluate({qid: qrels[qid]}, {qid: res}, [10])
        out[qid] = ndcg_d.get("NDCG@10", 0.0)
    return out


def find_best_val_alpha(
    val_qids: List[str],
    bm25_results: Dict,
    dense_results: Dict,
    qrels: Dict,
    alpha_grid: List[float],
    top_k: int = 100,
) -> float:
    """Choose alpha that maximises NDCG@10 on the validation split."""
    val_qrels = {q: qrels[q] for q in val_qids if q in qrels}
    if not val_qrels:
        return 0.5

    bm25_sub = {q: bm25_results[q] for q in val_qrels if q in bm25_results}
    dense_sub = {q: dense_results.get(q, {}) for q in val_qrels}

    best_ndcg, best_alpha = -1.0, 0.5
    for alpha in alpha_grid:
        fused = fuse_static(bm25_sub, dense_sub, alpha=alpha, top_k=top_k)
        fused_eval = {q: fused[q] for q in val_qrels if q in fused}
        if not fused_eval:
            continue
        ndcg_d, _, _, _ = EvaluateRetrieval.evaluate(val_qrels, fused_eval, [10])
        ndcg = ndcg_d.get("NDCG@10", 0.0)
        if ndcg > best_ndcg:
            best_ndcg, best_alpha = ndcg, alpha

    return best_alpha


def load_oracle_data(
    dataset_name: str,
    oracle_filename: str = "oracle_alpha.json",
) -> Dict:
    path = RESULTS_DIR / dataset_name / oracle_filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(rows: List[Dict]) -> None:
    header = (
        f"{'Method':<32} | {'NDCG@10':>7} | {'Rec@10':>7} | "
        f"{'MAP@10':>7} | {'P@10':>6} | {'MRR@10':>7} | {'a_MSE':>7} | {'a_MAE':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        mse_s = f"{r['alpha_mse']:.4f}" if r.get("alpha_mse") is not None else "   --  "
        mae_s = f"{r['alpha_mae']:.4f}" if r.get("alpha_mae") is not None else "   --  "
        print(
            f"{r['method']:<32} | {r.get('NDCG@10',0):>7.4f} | {r.get('Recall@10',0):>7.4f} | "
            f"{r.get('MAP@10',0):>7.4f} | {r.get('P@10',0):>6.4f} | {r.get('MRR@10',0):>7.4f} | "
            f"{mse_s:>7} | {mae_s:>7}"
        )
    print(sep)


# ── Statistical significance ──────────────────────────────────────────────────

def significance_test(
    adaptive_ndcg: np.ndarray,
    static_ndcg: np.ndarray,
    model_label: str,
    n_bootstrap: int = 1000,
) -> Dict:
    diff = adaptive_ndcg - static_ndcg
    t_stat, p_value = scipy_stats.ttest_rel(adaptive_ndcg, static_ndcg)

    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    cohens_d = float(diff.mean() / (diff.std() + 1e-12))
    effect = "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"

    result = {
        "mean_diff": float(diff.mean()),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "cohens_d": cohens_d,
        "effect_size": effect,
    }

    sig_str = "YES" if result["significant_at_05"] else "NO"
    print(f"\n{model_label} vs Best Static Hybrid:")
    print(f"  Mean NDCG@10 diff: {diff.mean():+.4f}")
    print(f"  p-value: {float(p_value):.4f}  (significant at alpha=0.05: {sig_str})")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Cohen's d: {cohens_d:.3f} ({effect})")
    return result


# ── Efficiency benchmark ──────────────────────────────────────────────────────

def benchmark_efficiency(
    trained: Dict,
    sample_qids: List[str],
    query_texts: Dict[str, str],
    bm25_results: Dict,
    dense_results: Dict,
    idf_dict: Dict,
    n: int = 100,
) -> Dict:
    sample = [q for q in sample_qids if q in query_texts and q in bm25_results][:n]
    if not sample:
        return {}

    ridge = trained["models"]["ridge"]
    mlp_s = trained["models"]["mlp_small"]
    scaler = trained["scaler"]

    # Feature extraction
    feat_times = []
    features_list = []
    for qid in sample:
        t0 = time.perf_counter()
        f = extract_query_features(
            query_text=query_texts[qid],
            bm25_results_for_query=bm25_results[qid],
            dense_results_for_query=dense_results.get(qid, {}),
            idf_dict=idf_dict,
        )
        feat_times.append((time.perf_counter() - t0) * 1000)
        features_list.append([f[fn] for fn in FEATURE_NAMES])

    feat_matrix = np.array(features_list)
    feat_sc = scaler.transform(feat_matrix)

    ridge_times = []
    for row in feat_sc:
        t0 = time.perf_counter()
        ridge.predict(row.reshape(1, -1))
        ridge_times.append((time.perf_counter() - t0) * 1000)

    mlp_times = []
    for row in feat_sc:
        t0 = time.perf_counter()
        mlp_s.predict(row.reshape(1, -1))
        mlp_times.append((time.perf_counter() - t0) * 1000)

    fusion_times = []
    for qid in sample:
        t0 = time.perf_counter()
        fuse_single_query(bm25_results[qid], dense_results.get(qid, {}), 0.5, 100)
        fusion_times.append((time.perf_counter() - t0) * 1000)

    feat_ms   = float(np.mean(feat_times))
    ridge_ms  = float(np.mean(ridge_times))
    mlp_ms    = float(np.mean(mlp_times))
    fusion_ms = float(np.mean(fusion_times))
    total_r   = feat_ms + ridge_ms + fusion_ms
    total_m   = feat_ms + mlp_ms + fusion_ms

    print(f"\nEfficiency Benchmark (avg over {len(sample)} queries):")
    print(f"  Feature extraction:  {feat_ms:.3f} ms")
    print(f"  Ridge alpha pred:    {ridge_ms:.3f} ms")
    print(f"  MLP-S alpha pred:    {mlp_ms:.3f} ms")
    print(f"  Score fusion:        {fusion_ms:.3f} ms")
    print(f"  Total (Ridge path):  {total_r:.3f} ms")
    print(f"  Total (MLP-S path):  {total_m:.3f} ms")
    llm_est = 200.0
    print(f"  Estimated LLM-based DAT: ~{llm_est:.0f} ms/query")
    print(f"  Our method speedup vs LLM: ~{llm_est/max(total_r,0.001):.0f}x")

    return {
        "feature_extraction_ms": feat_ms,
        "ridge_predict_ms": ridge_ms,
        "mlp_small_predict_ms": mlp_ms,
        "fusion_ms": fusion_ms,
        "total_ridge_ms": total_r,
        "total_mlp_small_ms": total_m,
        "n_samples": len(sample),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_full_evaluation(
    all_retrieval_data: Dict[str, Dict],
    best_static_by_dataset: Dict[str, float],
    top_k: int = 100,
    oracle_filename: str = "oracle_alpha.json",
    features_filename: str = "features.json",
    output_filename: str = "adaptive_model_results.json",
    split_suffix: str = "",
) -> Dict:
    """
    all_retrieval_data keys per dataset:
        bm25_results, dense_results, qrels_eval, query_texts, corpus

    Args:
        oracle_filename: filename for per-query oracle alpha data
            (default "oracle_alpha.json"; use "oracle_alpha_bge.json" for BGE)
        features_filename: features file to load for training
            (default "features.json"; use "features_bge.json" for BGE)
        output_filename: filename for the saved results JSON under results/
        split_suffix: appended to split filenames to avoid overwriting
            MiniLM splits when running BGE (e.g. "_bge")
    """
    dataset_names = sorted(all_retrieval_data.keys())

    # ── Prepare training data & splits ───────────────────────────────────────
    csv_name = (
        "combined_training_data_bge.csv"
        if "bge" in features_filename
        else "combined_training_data.csv"
    )
    df_all, df_inf = prepare_data(
        dataset_names,
        features_filename=features_filename,
        output_csv_name=csv_name,
    )
    if df_inf.empty:
        print("ERROR: No informative queries. Aborting.")
        return {}

    split_a, split_b = create_splits(df_inf, suffix=split_suffix)
    trained = train_all_models(df_inf, split_a, best_static_by_dataset)

    df_test = trained["df_test"]
    df_val  = trained["df_val"]

    # Build lookup: query_id -> predicted alpha for each model
    def _alpha_lookup(preds_arr: np.ndarray) -> Dict[str, float]:
        return dict(zip(df_test["query_id"].tolist(), preds_arr.tolist()))

    preds = trained["predictions"]["test"]
    alpha_by_model: Dict[str, Dict[str, float]] = {
        "adaptive_ridge":      _alpha_lookup(preds["ridge"]),
        "adaptive_mlp_small":  _alpha_lookup(preds["mlp_small"]),
        "adaptive_mlp_medium": _alpha_lookup(preds["mlp_medium"]),
        "heuristic_const":     _alpha_lookup(preds["heuristic_const"]),
        "heuristic_ratio":     _alpha_lookup(preds["heuristic_ratio"]),
        "heuristic_per_dataset": _alpha_lookup(preds["heuristic_per_dataset"]),
    }

    alpha_grid_coarse = [round(i / 10.0, 1) for i in range(1, 10)]

    # ── Per-dataset evaluation ────────────────────────────────────────────────
    dataset_metrics: Dict[str, Dict] = {}
    per_query_ndcg_all: Dict[str, Dict[str, float]] = {m: {} for m in [
        "bm25", "dense", "rrf", "best_static_val", "oracle",
        *alpha_by_model.keys(),
    ]}

    for ds in dataset_names:
        rd = all_retrieval_data[ds]
        bm25_r    = rd["bm25_results"]
        dense_r   = rd["dense_results"]
        qrels_ev  = rd["qrels_eval"]

        # Test queries for this dataset
        ds_test_qids = df_test[df_test["dataset"] == ds]["query_id"].tolist()
        if not ds_test_qids:
            print(f"  [SKIP] {ds}: no test queries in split A")
            continue

        # Filter qrels to test queries that have relevance judgements
        q_qrels = {q: qrels_ev[q] for q in ds_test_qids if q in qrels_ev}
        if not q_qrels:
            print(f"  [SKIP] {ds}: no qrels for test queries")
            continue

        test_qids = list(q_qrels.keys())

        # BM25 / Dense / RRF
        bm25_sub  = {q: bm25_r[q] for q in test_qids if q in bm25_r}
        dense_sub = {q: dense_r[q] for q in test_qids if q in dense_r}
        rrf_sub   = reciprocal_rank_fusion(bm25_sub, dense_sub, k=60, top_k=top_k)

        # Best static alpha on validation queries from this dataset
        val_qids_ds = df_val[df_val["dataset"] == ds]["query_id"].tolist()
        best_val_alpha = find_best_val_alpha(
            val_qids_ds, bm25_r, dense_r, qrels_ev, alpha_grid_coarse, top_k
        )
        _static_all = fuse_static(bm25_sub, dense_sub, best_val_alpha, top_k)
        static_sub = {q: _static_all[q] for q in test_qids if q in _static_all}

        # Oracle: per-query best alpha from oracle data file
        oracle_data = load_oracle_data(ds, oracle_filename=oracle_filename)
        oracle_alpha_map = {qid: d.get("best_alpha", 0.5) for qid, d in oracle_data.items()}
        oracle_sub = fuse_for_queries(test_qids, bm25_r, dense_r, oracle_alpha_map, top_k)

        # Adaptive models
        model_subs = {
            name: fuse_for_queries(test_qids, bm25_r, dense_r, alpha_map, top_k)
            for name, alpha_map in alpha_by_model.items()
        }

        all_method_results = {
            "bm25":             bm25_sub,
            "dense":            dense_sub,
            "rrf":              rrf_sub,
            "best_static_val":  static_sub,
            "oracle":           oracle_sub,
            **model_subs,
        }

        am = trained["alpha_metrics"]
        alpha_mse_map = {
            "adaptive_ridge":     am.get("ridge", {}).get("mse"),
            "adaptive_mlp_small": am.get("mlp_small", {}).get("mse"),
            "adaptive_mlp_medium":am.get("mlp_medium", {}).get("mse"),
        }
        alpha_mae_map = {
            "adaptive_ridge":     am.get("ridge", {}).get("mae"),
            "adaptive_mlp_small": am.get("mlp_small", {}).get("mae"),
            "adaptive_mlp_medium":am.get("mlp_medium", {}).get("mae"),
        }

        # Evaluate
        ds_metrics = {}
        rows = []
        for method, res in all_method_results.items():
            res_eval = {q: res[q] for q in q_qrels if q in res}
            if not res_eval:
                continue
            m = evaluate_results(q_qrels, res_eval)
            ds_metrics[method] = m
            pq = per_query_ndcg(q_qrels, res_eval)
            per_query_ndcg_all[method].update(pq)
            rows.append({
                "method": method, **m,
                "alpha_mse": alpha_mse_map.get(method),
                "alpha_mae": alpha_mae_map.get(method),
            })

        dataset_metrics[ds] = {"metrics": ds_metrics, "best_val_alpha": best_val_alpha}
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds}  (test={len(q_qrels)} queries, best_val_alpha={best_val_alpha:.1f})")
        print_results_table(rows)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("AGGREGATED (mean NDCG@10 across all test queries, all datasets):")
    agg_rows = []
    for method in per_query_ndcg_all:
        pq = per_query_ndcg_all[method]
        if not pq:
            continue
        am = trained["alpha_metrics"]
        mse = am.get({"adaptive_ridge":"ridge","adaptive_mlp_small":"mlp_small",
                       "adaptive_mlp_medium":"mlp_medium"}.get(method,""), {}).get("mse")
        mae = am.get({"adaptive_ridge":"ridge","adaptive_mlp_small":"mlp_small",
                       "adaptive_mlp_medium":"mlp_medium"}.get(method,""), {}).get("mae")
        agg_rows.append({
            "method": method,
            "NDCG@10":   float(np.mean(list(pq.values()))),
            "Recall@10": 0.0, "MAP@10": 0.0, "P@10": 0.0, "MRR@10": 0.0,
            "alpha_mse": mse, "alpha_mae": mae,
        })
    print_results_table(agg_rows)

    # ── Statistical significance ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    test_ids = df_test["query_id"].tolist()
    static_pq_arr = np.array([per_query_ndcg_all["best_static_val"].get(q, 0.0) for q in test_ids])
    sig_results = {}
    for model_label, key in [
        ("Adaptive Ridge", "adaptive_ridge"),
        ("Adaptive MLP-Small", "adaptive_mlp_small"),
        ("Adaptive MLP-Medium", "adaptive_mlp_medium"),
    ]:
        adap_arr = np.array([per_query_ndcg_all[key].get(q, 0.0) for q in test_ids])
        if adap_arr.sum() > 0:
            sig_results[key] = significance_test(adap_arr, static_pq_arr, model_label)

    # ── Cross-dataset transfer ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-DATASET TRANSFER (Strategy B, Ridge):")
    cd_trained = train_cross_dataset(df_inf, split_b)

    xfer_header = (
        f"{'Held-out':<22} | {'Adaptive':>10} | {'Best Static':>12} | {'Delta':>7} | {'Oracle':>8}"
    )
    print(f"\n{'-'*len(xfer_header)}")
    print(xfer_header)
    print(f"{'-'*len(xfer_header)}")

    cross_results = {}
    for held_out, cd in cd_trained.items():
        rd = all_retrieval_data.get(held_out)
        if rd is None:
            continue
        bm25_r   = rd["bm25_results"]
        dense_r  = rd["dense_results"]
        q_qrels  = rd["qrels_eval"]

        cd_test_df = cd["df_test"]
        xfer_alpha = dict(zip(cd_test_df["query_id"].tolist(), cd["predictions"].tolist()))
        xfer_qids  = list(q_qrels.keys())

        adapt_fused  = fuse_for_queries(xfer_qids, bm25_r, dense_r, xfer_alpha, top_k)
        adapt_m      = evaluate_results(q_qrels, adapt_fused)

        best_a = best_static_by_dataset.get(held_out, 0.5)
        static_fused = fuse_static(
            {q: bm25_r[q] for q in xfer_qids if q in bm25_r},
            {q: dense_r.get(q, {}) for q in xfer_qids},
            alpha=best_a, top_k=top_k,
        )
        static_m = evaluate_results(q_qrels, static_fused)

        oracle_data = load_oracle_data(held_out, oracle_filename=oracle_filename)
        oracle_alpha_map = {q: d.get("best_alpha", 0.5) for q, d in oracle_data.items()}
        oracle_fused = fuse_for_queries(xfer_qids, bm25_r, dense_r, oracle_alpha_map, top_k)
        oracle_m = evaluate_results(q_qrels, oracle_fused)

        delta = adapt_m["NDCG@10"] - static_m["NDCG@10"]
        print(
            f"{held_out:<22} | {adapt_m['NDCG@10']:>10.4f} | "
            f"{static_m['NDCG@10']:>12.4f} | {delta:>+7.4f} | {oracle_m['NDCG@10']:>8.4f}"
        )
        cross_results[held_out] = {
            "adaptive_ridge": adapt_m,
            "best_static":    static_m,
            "oracle":         oracle_m,
            "delta_ndcg10":   delta,
        }

    # ── Efficiency benchmark ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    bench_ds = dataset_names[0]
    bench_rd = all_retrieval_data[bench_ds]
    idf_dict = compute_idf_dict(bench_rd["corpus"])
    efficiency = benchmark_efficiency(
        trained=trained,
        sample_qids=list(bench_rd["bm25_results"].keys()),
        query_texts=bench_rd["query_texts"],
        bm25_results=bench_rd["bm25_results"],
        dense_results=bench_rd["dense_results"],
        idf_dict=idf_dict,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    def _jsonable(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: _jsonable(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_jsonable(x) for x in v]
        return v

    results_out = {
        "per_dataset_evaluation": _jsonable({
            ds: {
                "best_val_alpha": info["best_val_alpha"],
                **{m: info["metrics"].get(m, {}) for m in info["metrics"]},
            }
            for ds, info in dataset_metrics.items()
        }),
        "cross_dataset_transfer": _jsonable(cross_results),
        "significance_tests": _jsonable(sig_results),
        "feature_importance": trained["feature_importance"][
            ["feature", "coef_signed", "coef_abs"]
        ].to_dict(orient="records"),
        "alpha_prediction_metrics": _jsonable(trained["alpha_metrics"]),
        "efficiency": _jsonable(efficiency),
    }

    out_path = RESULTS_DIR / output_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nAll results saved -> {out_path}")

    return results_out
