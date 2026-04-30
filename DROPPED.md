# Dropped Dataset: trec-covid

**Status:** Permanently dropped as of Week 1 pivot (2026-04-27).

## Why

trec-covid has 171,333 documents and only 50 queries. Corpus encoding with
any transformer encoder (MiniLM, BGE, E5) takes several hours and produces
embeddings that cannot be loaded into memory on the dev machine during
the oracle grid search (100 alpha values x 50 queries x 171K docs).

Additionally, 50 queries is too small a sample for reliable statistical
testing (paired t-tests, bootstrap CIs, Cohen's d).

## Replacement

The medical/biomedical signal that trec-covid was meant to provide will be
covered by **MIRAGE** (Week 2). MIRAGE is a BEIR-compatible medical QA
benchmark with ~1,000 queries, making it statistically tractable.

## Final Dataset Set

| Dataset  | Domain        | Corpus   | Queries |
|----------|---------------|----------|---------|
| SciFact  | Science/claims| 5,183    | 300     |
| ArguAna  | Argument      | 8,674    | 1,406   |
| NFCorpus | Nutrition     | 3,633    | 323     |
| FiQA     | Finance QA    | 57,638   | 648     |
| SciDocs  | Scientific    | 25,657   | 1,000   |

## Data Files

The `data/trec-covid/` directory and `results/trec-covid/` files are kept
on disk but excluded from all new pipeline runs. Do not delete them --
they may be useful for checking encoding feasibility with smaller samples.
