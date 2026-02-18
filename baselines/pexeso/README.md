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

## One-command run (AutoFJ, default MPNet)

From repo root:

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj_join_benchmark \
  --embedding_mode mpnet \
  --out_csv autofj-experiments/pexeso-autofj-full-ranked.csv \
  --index_cache autofj-experiments/pexeso-autofj.index.pkl
```

## Paper-faithful (GloVe)

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj_join_benchmark \
  --embedding_mode glove \
  --embedding_pickle /path/to/model/glove.pikle \
  --out_csv autofj-experiments/pexeso-autofj-full-ranked.csv \
  --index_cache autofj-experiments/pexeso-autofj.index.pkl
```

If you have the PEXESO repo locally, you can pass:

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj_join_benchmark \
  --embedding_mode glove \
  --pexeso_repo /Users/yifanwu/Desktop/VIDA/tmp/LakeBench/join/Pexeso \
  --out_csv autofj-experiments/pexeso-autofj-full-ranked.csv
```

Default mode is `--embedding_mode mpnet` (not paper-faithful) and requires
`sentence-transformers` and `torch`. Paper-faithful requires the GloVe pickle
from the author’s setup. If you have the PEXESO repo locally, the expected path
is usually `model/glove.pikle` under that repo.

All downloaded model artifacts are stored in `baselines/pexeso/model`.

## One-command run (AutoFJ-GDC)

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj-gdc \
  --out_csv autofj-gdc-experiments/pexeso-autofj-gdc-full-ranked.csv \
  --index_cache autofj-gdc-experiments/pexeso-autofj-gdc.index.pkl
```

## Optional: fastText (paper alternative)

```bash
python3 baselines/pexeso/run_pexeso_baseline.py \
  --dataset_dir datasets/autofj_join_benchmark \
  --embedding_mode fasttext \
  --out_csv autofj-experiments/pexeso-autofj-full-ranked.csv
```

FastText vectors are downloaded from the official fastText site
(default: Common Crawl `crawl-300d-2M-subword.zip`) and cached under
`baselines/pexeso/model` by default. This file is large.

If `import fasttext` fails on macOS due to a wheel built for a newer OS,
reinstall from source or use conda:
- `pip uninstall -y fasttext && pip install --no-binary :all: fasttext`
- or `conda install -c conda-forge fasttext`

## Notes

- Default mode is `--embedding_mode mpnet` and requires `sentence-transformers` and `torch`.
- Paper-faithful mode is `--embedding_mode glove`, which requires GloVe pickle.
- Pivot selection default is `--pivot_method lof`, matching the author code and requires `scikit-learn`.
- The script auto-detects query source in this order:
  1. `autofj_query_columns.csv`
  2. `queries/`
  3. `groundtruth-joinable.csv` (uses `left_table` + default query column)
- Output schema matches `evaluate_retrieval.py` expectations:
  `query_table,query_column,candidate_table,candidate_column,similarity_score`

## References

- GitHub: https://github.com/mutong184/LakeBench/tree/main/join/Pexeso
- Paper: https://ieeexplore.ieee.org/document/9458717
