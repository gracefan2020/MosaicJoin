# WDC Top-k Annotation Workflow

This folder contains a 2-step pipeline:

1. `prune_wdc_top_k_for_annotation.py` prepares candidate pairs to annotate.
2. `top_k_annotation_llm.py` batches those pairs by `query_table` and asks an LLM to label joinability.

## 1) Prepare annotation candidates

Script: `scripts/prune_wdc_top_k_for_annotation.py`

What it does:
- Reads method result CSVs from `scripts/wdc_top_k/`.
- Keeps candidates whose `candidate_table` starts with `target_` (configurable).
- Keeps top-k per `(query_table, query_column)` within each method file.
- Unions unique pairs across methods.
- Optionally applies lexical filtering.
- Writes output CSV(s) to `scripts/wdc_top_k_annotation/`.

Input filename format:
- `METHOD_ft_BENCHMARK.csv`
- Example: `deepjoin_ft_wt-wdc.csv`

Required input columns in each top-k CSV:
- `query_table`
- `query_column`
- `candidate_table`
- `candidate_column`
- Optional but used for ranking: `similarity_score`

Default run:

```bash
python3 scripts/prune_wdc_top_k_for_annotation.py
```

Useful options:

```bash
# Process selected benchmarks only
python3 scripts/prune_wdc_top_k_for_annotation.py \
  --benchmarks autofj-wdc wt-wdc

# Disable top-k cap (keep all WDC candidates)
python3 scripts/prune_wdc_top_k_for_annotation.py \
  --top-k 0

# Enable lexical filtering
python3 scripts/prune_wdc_top_k_for_annotation.py \
  --lexical-filter \
  --lexical-threshold 0.2
```

Output file pattern:
- `scripts/wdc_top_k_annotation/<benchmark>_annotation_candidates.csv`

Output columns:
- `benchmark`
- `query_table`
- `query_column`
- `candidate_table`
- `candidate_column`
- `methods`
- `method_scores`
- `lexical_score`
- `query_unique_values`
- `candidate_unique_values`

## 2) Batch LLM annotation

Script: `scripts/top_k_annotation_llm.py`

What it does:
- Reads one CSV file or scans all `.csv` files under an input directory recursively.
- Keeps only files that contain required headers:
  - `query_table`, `query_column`, `candidate_table`, `candidate_column`,
  - `query_unique_values`, `candidate_unique_values`.
- Groups rows by `query_table`.
- Sends one LLM request per `query_table` batch.
- Parses model CSV output and appends labels.
- Writes checkpoint CSVs for resume/restart.

Dependencies:
- `pandas`
- `portkey-ai`
- `PORTKEY_API_KEY` environment variable must be set.

Default run:

```bash
python3 scripts/top_k_annotation_llm.py
```

Common runs:

```bash
# Annotate a specific candidates file
python3 scripts/top_k_annotation_llm.py \
  --input scripts/wdc_top_k_annotation/wt-wdc_annotation_candidates.csv

# Restrict to selected query tables
python3 scripts/top_k_annotation_llm.py \
  --query-tables query_1.csv query_2.csv

# Restart from scratch for existing outputs
python3 scripts/top_k_annotation_llm.py \
  --overwrite-checkpoint
```

Output directory default:
- `scripts/wdc_top_k_annotation_llm/`

Output naming:
- For input `<benchmark>_annotation_candidates.csv`, output is `<benchmark>_annotation_llm.csv`.

Checkpoint naming:
- `<output_stem>.checkpoint.csv` in the same output directory.

Notes:
- Model/provider knobs `MAX_TOKENS`, `THINKING_BUDGET`, `MAX_RETRIES`, `SLEEP_SECONDS`, and `OUTPUT_SUFFIX` are constants in `scripts/top_k_annotation_llm.py` (not CLI flags).
- Provider is currently `portkey` only.

## End-to-end example

```bash
# Step 1: generate candidate pairs
python3 scripts/prune_wdc_top_k_for_annotation.py --benchmarks wt-wdc

# Step 2: annotate with LLM
python3 scripts/top_k_annotation_llm.py \
  --input scripts/wdc_top_k_annotation \
  --output-dir scripts/wdc_top_k_annotation_llm
```

