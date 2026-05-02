# Adaptive Hybrid Retrieval — Project Summary

**Target venue:** CIKM 2026 (short / resource paper)
**Submission deadline:** June 6, 2026
**Last updated:** 2026-04-28

---

## Research Question

> When does per-query adaptive alpha selection in hybrid BM25+dense retrieval
> actually pay off — and how does encoder quality affect that benefit?

The paper's central claim (the *encoder saturation hypothesis*):
adaptive fusion gains are largest when the dense encoder is weak (MiniLM),
shrink as encoder quality improves (BGE), and approach zero with a strong
encoder (E5). The oracle analysis provides a principled ceiling to quantify
how much headroom exists.

---

## Three Contributions

| # | Contribution | Status |
|---|-------------|--------|
| 1 | **Encoder saturation finding** — adaptive gain decreases monotonically MiniLM > BGE > E5 | Results in `adaptive_model_results_*_v2.json` |
| 2 | **Oracle ceiling analysis** — corrected oracle NDCG over ALL queries, not just "informative" subset | Bug fixed; v2 oracle files regenerated |
| 3 | **Latency-quality Pareto frontier vs DAT** | Week 3 |

---

## Datasets

Five BEIR benchmarks (trec-covid permanently dropped — see `DROPPED.md`):

| Dataset | Domain | Corpus | Queries |
|---------|--------|--------|---------|
| SciFact | Scientific claims | 5,183 | 300 |
| ArguAna | Argument retrieval | 8,674 | 1,406 |
| NFCorpus | Nutrition/fitness | 3,633 | 323 |
| FiQA | Financial QA | 57,638 | 648 |
| SciDocs | Scientific docs | 25,657 | 1,000 |

Canonical dataset list: `src/config.py → DATASETS`

---

## Encoders

Three dense encoders, increasing quality:

| Encoder | Model ID | Dim | Query prefix | Doc prefix |
|---------|----------|-----|-------------|-----------|
| MiniLM | `all-MiniLM-L6-v2` | 384 | none | none |
| BGE | `BAAI/bge-base-en-v1.5` | 768 | `"Represent this sentence..."` | none |
| E5 | `intfloat/e5-base-v2` | 768 | `"query: "` | `"passage: "` |

Corpus embeddings cached to `.npz` under `results/cache/dense{,_bge,_e5}/`.

---

## Alpha Grid

Canonical grid defined in `src/config.py`:

```python
ALPHA_GRID = [0.00, 0.05, 0.10, ..., 1.00]   # 21 points, step 0.05
```

Used for:
- Oracle grid search (per-query best alpha)
- Static-best alpha selection on the validation split

**Not used for:** feature-based regressors (they predict a continuous alpha).

---

## Pipeline Architecture

```
BEIR corpus/queries/qrels
        │
        ├─── BM25 retrieval ─────────────────────────────┐
        │    (rank-bm25, top-100 per query)               │
        │                                                  │
        └─── Dense retrieval ────────────────────────────┤
             (SentenceTransformer, cached .npz)           │
                                                          ▼
                                             Hybrid fusion (alpha sweep)
                                             score = α·BM25 + (1-α)·Dense
                                             (min-max normalized per query)
                                                          │
                              ┌───────────────────────────┤
                              │                           │
                              ▼                           ▼
                    Oracle (ALPHA_GRID,          Static-best (ALPHA_GRID,
                    per-query best α,            best single α on val split)
                    ALL test queries)
                              │
                              ▼
                    Query feature extraction (17 features)
                    [query_length, avg_idf, bm25_top1_score,
                     dense_top1_score, rank_disagreement, ...]
                              │
                              ▼
                    Train/val/test split (70/15/15, stratified)
                    Informative queries only (α varies across grid)
                              │
                              ▼
                    Adaptive model training
                    ├── Ridge (RidgeCV)
                    ├── MLP-Small (hidden=32)
                    └── MLP-Medium (hidden=64→32)
                              │
                              ▼
                    Evaluation on test split
                    + Oracle evaluated on ALL queries (ceiling)
                    + Significance tests (t-test, bootstrap CI, Cohen's d)
                    + Cross-dataset transfer (leave-one-out)
```

---

## File Structure

```
adaptive-hybrid-retrieval/
├── src/
│   ├── config.py                   ← DATASETS, ALPHA_GRID (canonical)
│   ├── data/
│   │   ├── load_beir.py
│   │   └── download_beir.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py      ← E5 doc prefix support added
│   │   ├── hybrid_static.py
│   │   └── rrf.py
│   ├── features/
│   │   └── query_features.py       ← 17 query features
│   ├── training/
│   │   ├── generate_oracle_alpha.py ← 0.05 grid, closest-to-0.5 tie-break
│   │   ├── prepare_data.py
│   │   ├── split_data.py
│   │   └── train_adaptive.py
│   └── evaluation/
│       └── evaluate_adaptive.py    ← oracle over ALL queries (bug fixed)
│
├── run_oracle_v2.py                ← recompute MiniLM+BGE oracle (new grid)
├── run_e5_pipeline.py              ← full E5 encode+oracle+train pipeline
├── run_bge_pipeline.py             ← full BGE pipeline (updated to new grid)
├── run_adaptive_pipeline.py        ← train+evaluate any encoder via flags
├── main.py                         ← original MiniLM baseline pipeline
│
├── results/
│   ├── {dataset}/
│   │   ├── oracle_alpha_{enc}_v2.json    ← per-query best alpha (v2)
│   │   ├── features_{enc}_v2.json        ← 17 features + oracle targets
│   │   └── run_summary.json              ← BM25/Dense/RRF/Static baselines
│   ├── adaptive_model_results_{enc}_v2.json  ← full evaluation output
│   ├── combined_training_data_{enc}.csv
│   └── splits/
│
├── models/                         ← pickled Ridge/MLP/Scaler per encoder
├── DROPPED.md                      ← trec-covid removal rationale
└── PROJECT_SUMMARY.md              ← this file
```

---

## Key Bug Fixes (Week 1)

### 1. Oracle over ALL queries
**Before:** Oracle NDCG was computed only on the "informative" training test split
(queries where alpha varies). This artificially deflated oracle NDCG for strong
encoders like BGE (fewer queries are "informative" when dense is already good),
causing the FiQA anomaly.

**After:** Oracle is evaluated over every query in qrels, giving the true
global ceiling. Adaptive/static are still evaluated on the informative test split.

### 2. Alpha grid unified at 0.05 step
**Before:** Oracle used a 51-point 0.02-step grid; static-best used a 9-point
[0.1..0.9] grid. Inconsistent.

**After:** Both use `ALPHA_GRID` from `src/config.py` — 21 points, step 0.05.

### 3. Oracle tie-breaking: parsimony
**Before:** Middle alpha among ties (arbitrary).
**After:** Alpha closest to 0.5 — the most neutral blend, reducing noise in targets.

### 4. E5 document prefix
E5 requires `"passage: "` prepended to document text at encoding time.
Without it, retrieval quality drops significantly. Added to `dense_retriever.py`
via `get_doc_prefix()` with auto-detection from model name.

---

## How to Run

### Full E5 pipeline (encoding ~3-6h on GPU)
```powershell
python run_e5_pipeline.py
```

### Recompute MiniLM + BGE oracle on new grid (minutes, uses cached embeddings)
```powershell
python run_oracle_v2.py
```

### Retrain adaptive models with v2 oracle targets
```powershell
python run_adaptive_pipeline.py --oracle-filename oracle_alpha_minilm_v2.json --features-filename features_minilm_v2.json --dense-label minilm_v2

python run_adaptive_pipeline.py --dense-model BAAI/bge-base-en-v1.5 --dense-cache-dir results/cache/dense_bge --oracle-filename oracle_alpha_bge_v2.json --features-filename features_bge_v2.json --dense-label bge_v2
```

---

## Week Plan

| Week | Dates | Goal | Status |
|------|-------|------|--------|
| 1 | Apr 27 – May 3 | Foundation refactor: 5-dataset set, 0.05 grid, oracle bug fix, E5 encoder | **Done** |
| 2 | May 4 – May 10 | MIRAGE dataset, DAT latency baseline | Pending |
| 3 | May 11 – May 17 | Pareto frontier analysis, statistical tables | Pending |
| 4 | May 18 – May 24 | Paper draft: intro, method, results | Pending |
| 5 | May 25 – May 31 | Related work, conclusion, revision | Pending |
| 6 | Jun 1 – Jun 6 | Final polish, submission | Pending |

---

## Output Files per Encoder (v2 pipeline)

| File | Contents |
|------|---------|
| `results/{ds}/oracle_alpha_{enc}_v2.json` | Per-query best alpha + NDCG@10 grid scores |
| `results/{ds}/features_{enc}_v2.json` | 17 features + oracle alpha + is_informative flag |
| `results/{ds}/run_summary.json` | BM25, Dense, RRF, Static hybrid baselines |
| `results/adaptive_model_results_{enc}_v2.json` | Ridge/MLP metrics, significance tests, oracle gap |
| `results/combined_training_data_{enc}.csv` | Flat training DataFrame (informative queries) |
| `models/{ridge,mlp_small,mlp_medium,scaler}_{enc}.pkl` | Trained model checkpoints |
