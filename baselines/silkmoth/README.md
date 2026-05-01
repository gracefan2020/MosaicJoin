# SILKMOTH Baseline (Minimal Search Runner)

This folder provides a small, repository-local SILKMOTH baseline runner for
SemSketch benchmarks.

The implementation follows the paper's token-based search pipeline closely in
the `alpha = 0` Jaccard setting:

- weighted signature generation,
- signature-based candidate selection,
- check filtering,
- nearest-neighbor refinement,
- exact maximum-weight bipartite matching verification.

It is intentionally minimal and dependency-light: pure Python, no external
model downloads, and evaluator-compatible CSV output.

## Install

From repo root:

```bash
python3 -m pip install -r baselines/silkmoth/requirements.txt
```

## One-command run

From repo root:

```bash
python3 baselines/silkmoth/run_silkmoth_baseline.py \
  --dataset_dir datasets/autofj \
  --metric containment \
  --relatedness_threshold 0.7 \
  --out_csv autofj-experiments/silkmoth-autofj.csv
```

## Smaller smoke test

```bash
python3 baselines/silkmoth/run_silkmoth_baseline.py \
  --dataset_dir datasets/autofj \
  --metric containment \
  --relatedness_threshold 0.7 \
  --max_datalake_tables 25 \
  --max_queries 5 \
  --out_csv /tmp/silkmoth-smoke.csv
```

## SLURM runner

```bash
cd baselines/silkmoth
sbatch run_eval.SBATCH all
```

Supported benchmark names:

- `autofj`
- `autofj-wdc`
- `freyja`
- `freyja-wdc`
- `wt`
- `wt-wdc`
- `all`

Useful overrides:

```bash
SILKMOTH_METRIC=containment \
SILKMOTH_THRESHOLD=0.7 \
SILKMOTH_TOP_K=50 \
sbatch run_eval.SBATCH wt-wdc
```

## Output schema

The runner writes:

- `query_table`
- `query_column`
- `candidate_table`
- `candidate_column`
- `similarity_score`

This is directly compatible with `evaluate_retrieval.py`.

## Notes

- Default mode is `--metric containment`, which matches the paper's approximate
  inclusion-dependency search setting most closely.
- The current minimal runner implements the paper's Jaccard search path with
  `alpha = 0`. It does not include the paper's `alpha > 0` skyline/dichotomy
  extensions or the edit-similarity/q-gram variant.
- Query sources are auto-discovered in this order:
  `query_columns.csv` -> `autofj_query_columns.csv` ->
  `gdc_breakdown_query_columns.csv` -> `groundtruth-joinable.csv` ->
  `queries/` -> `datalake/`.

## References

- Paper: https://www.vldb.org/pvldb/vol10/p1082-deng.pdf
- DOI: https://doi.org/10.14778/3115404.3115413
