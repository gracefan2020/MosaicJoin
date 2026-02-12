# PEXESO Baseline (Simplified Integration)

This folder provides a clean, repository-local PEXESO baseline runner that:
- follows the core PEXESO flow (pivot mapping + hierarchical grid + block-and-verify),
- avoids hardcoded paths from the original code,
- writes output in SemSketch evaluator format.

## Install

From repo root:

```bash
python3 -m pip install -r baselines/pexeso/requirements.txt
```

## One-command run (AutoFJ)

From repo root:

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj_join_benchmark \
  --out_csv autofj-experiments/pexeso-autofj-full-ranked.csv \
  --index_cache autofj-experiments/pexeso-autofj.index.pkl
```

## One-command run (AutoFJ-GDC)

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj-gdc \
  --out_csv autofj-gdc-experiments/pexeso-autofj-gdc-full-ranked.csv \
  --index_cache autofj-gdc-experiments/pexeso-autofj-gdc.index.pkl
```

## Optional: use external token embeddings

If you have a token->vector pickle (similar to the author code path), run:

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj_join_benchmark \
  --embedding_mode pickle \
  --embedding_pickle /path/to/token_vectors.pkl \
  --out_csv autofj-experiments/pexeso-autofj-full-ranked.csv
```

## Notes

- Default mode is `--embedding_mode hash`, which is deterministic and requires no external model files.
- The script auto-detects query source in this order:
  1. `autofj_query_columns.csv`
  2. `queries/`
  3. `groundtruth-joinable.csv` (uses `left_table` + default query column)
- Output schema matches `evaluate_retrieval.py` expectations:
  `query_table,query_column,candidate_table,candidate_column,similarity_score`
