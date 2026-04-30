"""
run_e5_pipeline.py

Full pipeline for intfloat/e5-base-v2 as the third dense encoder.

IMPORTANT: E5 requires prefixes on BOTH queries and documents:
    query  text -> "query: <text>"
    document text -> "passage: <text>"
Without these prefixes, E5 retrieval quality drops significantly (confirmed
by intfloat documentation and MTEB benchmarks).

Steps:
  1.  Verify E5 model loads and produces 768-dim normalized embeddings
  2.  E5 dense retrieval for all 5 datasets (cached to results/cache/dense_e5/)
  3.  Update run_summary.json with e5 keys
  4.  Generate oracle_alpha_e5_v2.json per dataset (21-point 0.05-step grid, all queries)
  5.  Extract features_e5.json per dataset
  6.  Sanity check: SciFact dense NDCG@10 should be ~0.74-0.78
  7.  Train adaptive models on E5 features + oracle targets
  8.  Full evaluation -> adaptive_model_results_e5.json
  9.  Print encoder saturation comparison (MiniLM < BGE < E5 adaptive gain hypothesis)

Usage:
    python run_e5_pipeline.py [--datasets scifact arguana ...] [options]

    --data-dir DIR        BEIR data directory (default: data)
    --split SPLIT         Dataset split to use (default: test)
    --top-k K             Documents to retrieve per query (default: 100)
    --skip-retrieval      Skip encoding; load cached E5 results
"""

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import DATASETS, ALPHA_GRID
from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import (
    run_dense_retrieval,
    E5_QUERY_PREFIX,
    E5_DOC_PREFIX,
)
from src.retrieval.hybrid_static import fuse_static
from src.retrieval.rrf import reciprocal_rank_fusion
from src.training.generate_oracle_alpha import generate_oracle_alphas, save_oracle_data
from src.features.query_features import (
    extract_query_features,
    compute_idf_dict,
    FEATURE_NAMES,
)
from src.training.prepare_data import prepare_data
from src.training.split_data import strategy_a
from src.training.train_adaptive import train_all_models, train_cross_dataset
from src.evaluation.evaluate_adaptive import (
    run_full_evaluation,
    evaluate_results,
    fuse_for_queries,
    load_oracle_data,
    find_best_val_alpha,
    per_query_ndcg,
    print_results_table,
    significance_test,
)

from beir.retrieval.evaluation import EvaluateRetrieval

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
E5_MODEL = "intfloat/e5-base-v2"
E5_CACHE_DIR = "results/cache/dense_e5"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full E5-base-v2 encoder pipeline for adaptive hybrid retrieval."
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--skip-retrieval", action="store_true",
        help="Skip E5 encoding; assume cached embeddings already exist.",
    )
    return parser.parse_args()


# ── Step 1: Verify E5 model ───────────────────────────────────────────────────

def verify_e5_model():
    print("\n" + "=" * 70)
    print("STEP 1: Verifying E5 model")
    print("=" * 70)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(E5_MODEL)
    sample_query = f"{E5_QUERY_PREFIX}What is hybrid retrieval?"
    sample_doc   = f"{E5_DOC_PREFIX}Hybrid retrieval combines sparse and dense methods."
    q_emb = model.encode([sample_query], normalize_embeddings=True)
    d_emb = model.encode([sample_doc],   normalize_embeddings=True)
    assert q_emb.shape == (1, 768), f"Expected (1, 768) but got {q_emb.shape}"
    norm = float(np.linalg.norm(q_emb[0]))
    assert abs(norm - 1.0) < 1e-5, f"Embedding not normalized: norm={norm:.6f}"
    score = float((q_emb @ d_emb.T)[0, 0])
    print(f"  E5 model loaded: embedding dim=768, normalized=True")
    print(f"  Sample query-doc cosine score: {score:.4f}")
    print(f"  Query prefix:    \"{E5_QUERY_PREFIX}\"")
    print(f"  Document prefix: \"{E5_DOC_PREFIX}\"")
    print("  [OK] E5 model verified")


# ── Step 2: E5 dense retrieval ────────────────────────────────────────────────

def run_e5_retrieval(corpus, queries, dataset_name, split, top_k):
    """Run E5 dense retrieval with caching. Returns {qid: {doc_id: score}}."""
    cache_key = f"{dataset_name}::{split}::{E5_MODEL}"
    # doc_prefix and query_prefix are auto-detected from model_name ("e5" in name)
    return run_dense_retrieval(
        corpus, queries,
        model_name=E5_MODEL,
        top_k=top_k,
        cache_dir=E5_CACHE_DIR,
        cache_key=cache_key,
    )


# ── Step 3: Update run_summary.json ──────────────────────────────────────────

def update_run_summary_e5(
    dataset_name: str,
    bm25_results: dict,
    e5_results: dict,
    qrels: dict,
    corpus_size: int,
    top_k: int,
):
    """Add dense_e5, hybrid_e5_alpha_*, rrf_e5_k60, best_static_hybrid_e5 to run_summary.json."""
    summary_path = RESULTS_DIR / dataset_name / "run_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {"metadata": {}, "bm25": {}}

    evaluated_qids = set(bm25_results.keys()) & set(e5_results.keys())
    qrels_ev = {q: qrels[q] for q in evaluated_qids if q in qrels}

    # Dense E5 standalone
    e5_sub = {q: e5_results[q] for q in qrels_ev}
    e5_m = evaluate_results(qrels_ev, e5_sub)
    summary["dense_e5"] = e5_m
    print(f"  dense_e5 NDCG@10={e5_m['NDCG@10']:.4f}")

    # Sanity check: SciFact should be ~0.74-0.78
    if dataset_name == "scifact":
        ndcg = e5_m["NDCG@10"]
        if ndcg < 0.55:
            print(f"  [WARN] SciFact E5 NDCG@10={ndcg:.4f} is suspiciously low.")
            print(f"         Expected ~0.74-0.78. Check that E5 prefixes are applied correctly.")
        elif 0.74 <= ndcg <= 0.82:
            print(f"  [OK] SciFact E5 NDCG@10={ndcg:.4f} is in expected range (0.74-0.78).")
        else:
            print(f"  [NOTE] SciFact E5 NDCG@10={ndcg:.4f} -- outside expected range, verify.")

    # Static hybrid with E5 (21-point 0.05-step grid)
    bm25_sub = {q: bm25_results[q] for q in qrels_ev}
    best_e5_ndcg, best_e5_alpha = -1.0, 0.5
    for alpha in ALPHA_GRID:
        fused = fuse_static(bm25_sub, {q: e5_results.get(q, {}) for q in qrels_ev},
                            alpha=alpha, top_k=top_k)
        m = evaluate_results(qrels_ev, fused)
        key = f"hybrid_e5_alpha_{alpha:.2f}"
        summary[key] = m
        if m["NDCG@10"] > best_e5_ndcg:
            best_e5_ndcg = m["NDCG@10"]
            best_e5_alpha = alpha

    summary["best_static_hybrid_e5"] = {
        "alpha": best_e5_alpha,
        **{k: v for k, v in summary[f"hybrid_e5_alpha_{best_e5_alpha:.2f}"].items()},
    }
    print(f"  best_static_hybrid_e5: alpha={best_e5_alpha}  NDCG@10={best_e5_ndcg:.4f}")

    # RRF with E5
    rrf_fused = reciprocal_rank_fusion(
        bm25_sub, {q: e5_results.get(q, {}) for q in qrels_ev}, k=60, top_k=top_k
    )
    rrf_m = evaluate_results(qrels_ev, rrf_fused)
    summary["rrf_e5_k60"] = rrf_m
    print(f"  rrf_e5_k60 NDCG@10={rrf_m['NDCG@10']:.4f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return best_e5_alpha, qrels_ev


# ── Step 4: Oracle alpha with E5 ─────────────────────────────────────────────

def generate_e5_oracle(dataset_name, bm25_results, e5_results, qrels_ev, top_k):
    """Generate oracle_alpha_e5_v2.json. Uses ALPHA_GRID over ALL queries."""
    print(f"  Running oracle grid search ({len(ALPHA_GRID)} alpha values, "
          f"{len(qrels_ev)} queries)...")
    t0 = time.perf_counter()
    oracle_data = generate_oracle_alphas(
        qrels=qrels_ev,
        bm25_results=bm25_results,
        dense_results=e5_results,
        alpha_grid=ALPHA_GRID,
        top_k=top_k,
    )
    elapsed = time.perf_counter() - t0

    out_path = RESULTS_DIR / dataset_name / "oracle_alpha_e5_v2.json"
    save_oracle_data(oracle_data, out_path)

    # Stats over ALL queries
    all_entries = list(oracle_data.values())
    all_alphas = [d["best_alpha"] for d in all_entries]
    all_ndcgs  = [d["best_ndcg"]  for d in all_entries]
    n_inf = sum(1 for d in all_entries if d.get("is_informative"))
    mean_alpha = float(np.mean(all_alphas)) if all_alphas else 0.0
    std_alpha  = float(np.std(all_alphas))  if all_alphas else 0.0
    mean_ndcg  = float(np.mean(all_ndcgs))  if all_ndcgs  else 0.0

    print(f"  Oracle (E5): mean_alpha={mean_alpha:.3f} std={std_alpha:.3f}  "
          f"NDCG@10={mean_ndcg:.4f}  informative={n_inf}/{len(all_entries)}  "
          f"({elapsed:.1f}s)")
    return oracle_data


# ── Step 5: Feature extraction with E5 ───────────────────────────────────────

def extract_e5_features(
    dataset_name: str,
    corpus: dict,
    queries: dict,
    bm25_results: dict,
    e5_results: dict,
    oracle_e5: dict,
) -> dict:
    """Build features_e5.json with E5 dense signals and v2 oracle targets."""
    idf_dict = compute_idf_dict(corpus)
    query_data = {}
    for qid, query_text in queries.items():
        bm25_q = bm25_results.get(qid, {})
        e5_q   = e5_results.get(qid, {})
        feats = extract_query_features(
            query_text=query_text,
            bm25_results_for_query=bm25_q,
            dense_results_for_query=e5_q,
            idf_dict=idf_dict,
        )
        oracle_entry = oracle_e5.get(qid, {})
        query_data[qid] = {
            "features": feats,
            "oracle_alpha": oracle_entry.get("best_alpha"),
            "oracle_ndcg":  oracle_entry.get("best_ndcg"),
            "is_informative": oracle_entry.get("is_informative", False),
        }

    n_inf = sum(1 for d in query_data.values() if d["is_informative"])
    features_e5 = {
        "metadata": {
            "dataset": dataset_name,
            "num_queries": len(queries),
            "num_informative": n_inf,
            "num_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "dense_model": E5_MODEL,
            "alpha_grid": "0.05-step (21 points)",
            "oracle_version": "v2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "queries": query_data,
    }

    out_path = RESULTS_DIR / dataset_name / "features_e5.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features_e5, f, indent=2)
    print(f"  features_e5.json saved -> {out_path}  "
          f"(queries={len(queries)}, informative={n_inf})")
    return features_e5


# ── Save models ───────────────────────────────────────────────────────────────

def save_models_e5(trained: dict):
    """Persist E5 adaptive models as pkl files."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    to_save = {
        "ridge_e5.pkl":      trained["models"]["ridge"],
        "mlp_small_e5.pkl":  trained["models"]["mlp_small"],
        "mlp_medium_e5.pkl": trained["models"]["mlp_medium"],
        "scaler_e5.pkl":     trained["scaler"],
    }
    for fname, obj in to_save.items():
        path = MODELS_DIR / fname
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {path}")


# ── Encoder saturation summary ────────────────────────────────────────────────

def print_saturation_summary(datasets: list):
    """
    Print the headline encoder-saturation finding:
      - Adaptive vs static gain should decrease: MiniLM > BGE > E5
      - Pearson r (alpha prediction quality) progression
      - Oracle ceiling gap per encoder
    """
    result_files = {
        "MiniLM": RESULTS_DIR / "adaptive_model_results.json",
        "BGE":    RESULTS_DIR / "adaptive_model_results_bge.json",
        "E5":     RESULTS_DIR / "adaptive_model_results_e5.json",
    }

    print("\n" + "=" * 70)
    print("ENCODER SATURATION SUMMARY  (headline finding for paper)")
    print("=" * 70)
    print("Hypothesis: adaptive gain over static decreases as encoder quality increases.")
    print("Expected order: MiniLM > BGE > E5\n")

    header = f"{'Encoder':<10} | {'Adaptive-Static':>16} | {'Pearson r':>10} | {'Oracle gap':>11}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for enc, path in result_files.items():
        if not path.exists():
            print(f"{enc:<10} | {'(not found)':>16} | {'':>10} | {'':>11}")
            continue
        with open(path, encoding="utf-8") as f:
            res = json.load(f)

        # Adaptive ridge mean NDCG vs best static val mean NDCG
        def _mean(method):
            vals = []
            for ds_data in res.get("per_dataset_evaluation", {}).values():
                v = ds_data.get(method, {}).get("NDCG@10")
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")

        adap = _mean("adaptive_ridge")
        stat = _mean("best_static_val")
        orac = _mean("oracle")
        gain = adap - stat if (adap == adap and stat == stat) else float("nan")
        gap  = orac - adap if (orac == orac and adap == adap) else float("nan")

        pr = res.get("alpha_prediction_metrics", {}).get("ridge", {}).get("pearson_r",
             float("nan"))

        def _f(v): return f"{v:+.4f}" if v == v else "  N/A  "
        def _fp(v): return f"{v:.4f}" if v == v else " N/A  "
        print(f"{enc:<10} | {_f(gain):>16} | {_fp(pr):>10} | {_f(gap):>11}")
    print(sep)

    print("\nInterpretation:")
    print("  Adaptive-Static: mean(Adaptive Ridge NDCG) - mean(Best Static Val NDCG)")
    print("  Pearson r:       correlation between predicted and oracle alpha (test set)")
    print("  Oracle gap:      mean(Oracle NDCG) - mean(Adaptive Ridge NDCG)")
    print("  If gain decreases monotonically MiniLM > BGE > E5: saturation hypothesis HOLDS.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\nE5 ENCODER PIPELINE")
    print(f"Model: {E5_MODEL}")
    print(f"Query prefix:    \"{E5_QUERY_PREFIX}\"")
    print(f"Document prefix: \"{E5_DOC_PREFIX}\"")
    print(f"Datasets: {args.datasets}")
    print(f"Top-k: {args.top_k}")
    print(f"Alpha grid: {len(ALPHA_GRID)} points (0.05 step)")

    # Step 1: Verify E5 model
    verify_e5_model()

    all_bm25    = {}
    all_e5      = {}
    all_qrels   = {}
    all_qrels_ev = {}
    all_corpus  = {}
    all_queries = {}
    best_static_e5 = {}
    encoding_times = {}

    # Steps 2-5: Per-dataset retrieval, oracle, features
    for ds in args.datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds}")
        print("=" * 70)

        download_beir_dataset(data_dir=args.data_dir, dataset_name=ds)
        corpus, queries, qrels = load_beir_dataset(
            data_dir=args.data_dir, dataset_name=ds, split=args.split
        )
        print(f"  corpus={len(corpus):,}  queries={len(queries):,}")

        # BM25
        t0 = time.perf_counter()
        bm25_results = run_bm25(corpus, queries, top_k=args.top_k)
        print(f"  BM25 done ({time.perf_counter() - t0:.1f}s)")

        # E5 dense retrieval (auto-applies "query: " / "passage: " prefixes)
        if args.skip_retrieval:
            print(f"  [skip-retrieval] Loading E5 from cache...")
        t0 = time.perf_counter()
        e5_results = run_e5_retrieval(corpus, queries, ds, args.split, args.top_k)
        t_e5 = time.perf_counter() - t0
        encoding_times[ds] = t_e5
        print(f"  E5 retrieval done ({t_e5:.1f}s)")

        all_bm25[ds]     = bm25_results
        all_e5[ds]       = e5_results
        all_qrels[ds]    = qrels
        all_corpus[ds]   = corpus
        all_queries[ds]  = queries

        # Step 3: Update run_summary.json
        print(f"\n[Step 3] Updating run_summary.json for {ds}...")
        best_alpha_e5, qrels_ev = update_run_summary_e5(
            ds, bm25_results, e5_results, qrels, len(corpus), args.top_k
        )
        best_static_e5[ds] = best_alpha_e5
        all_qrels_ev[ds] = qrels_ev

        # Step 4: Oracle alpha with E5
        print(f"\n[Step 4] Generating oracle_alpha_e5_v2.json for {ds}...")
        oracle_e5 = generate_e5_oracle(ds, bm25_results, e5_results, qrels_ev, args.top_k)

        # Step 5: Feature extraction with E5
        print(f"\n[Step 5] Extracting features_e5.json for {ds}...")
        extract_e5_features(ds, corpus, queries, bm25_results, e5_results, oracle_e5)

    # Steps 6-7: Training + full evaluation
    print(f"\n{'=' * 70}")
    print("STEPS 6-7: Training adaptive models with E5 features + full evaluation")
    print("=" * 70)

    all_retrieval_data_e5 = {
        ds: {
            "bm25_results":  all_bm25[ds],
            "dense_results": all_e5[ds],
            "qrels":         all_qrels[ds],
            "qrels_eval":    all_qrels_ev[ds],
            "query_texts":   all_queries[ds],
            "corpus":        all_corpus[ds],
        }
        for ds in args.datasets
    }

    e5_eval_results = run_full_evaluation(
        all_retrieval_data=all_retrieval_data_e5,
        best_static_by_dataset=best_static_e5,
        top_k=args.top_k,
        oracle_filename="oracle_alpha_e5_v2.json",
        features_filename="features_e5.json",
        output_filename="adaptive_model_results_e5.json",
        split_suffix="_e5",
    )

    # Save adaptive models to disk
    print(f"\n[Step 7b] Saving E5 adaptive models...")
    df_all_e5, df_inf_e5 = prepare_data(
        args.datasets,
        features_filename="features_e5.json",
        output_csv_name="combined_training_data_e5.csv",
    )
    if not df_inf_e5.empty:
        split_a_e5 = strategy_a(df_inf_e5)
        trained_e5 = train_all_models(df_inf_e5, split_a_e5, best_static_e5)
        save_models_e5(trained_e5)

    # Step 8: Encoder saturation summary
    print_saturation_summary(args.datasets)

    print(f"\n{'=' * 70}")
    print("E5 PIPELINE COMPLETE")
    print("=" * 70)
    print("Files generated:")
    for ds in args.datasets:
        print(f"  results/{ds}/oracle_alpha_e5_v2.json")
        print(f"  results/{ds}/features_e5.json")
        print(f"  results/{ds}/run_summary.json (updated with E5 keys)")
    print(f"  results/adaptive_model_results_e5.json")
    print(f"  results/combined_training_data_e5.csv")
    print(f"  models/ridge_e5.pkl")
    print(f"  models/mlp_small_e5.pkl")
    print(f"  models/mlp_medium_e5.pkl")
    print()
    print("Recommended next step -- run the oracle v2 recomputation for MiniLM+BGE")
    print("to get consistent baselines on the 0.05 grid before generating tables:")
    print("  python run_oracle_v2.py")


if __name__ == "__main__":
    main()
