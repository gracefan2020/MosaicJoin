# KOIOS Baseline

This folder contains a small MosaicJoin-native KOIOS baseline.

- `build_koios_embedding_db.py` exports MosaicJoin CSV columns as KOIOS sets and
  builds the KOIOS-style `ft.sqlite3` embedding DB.
- `run_koios_baseline.py` reads those set files and computes KOIOS semantic
  overlap with exact maximum bipartite matching.
- Output schema matches the other baselines:
  `query_table,query_column,candidate_table,candidate_column,similarity_score`.

Supported dataset shortcuts:

- `autofj`
- `autofj-wdc`
- `freyja`
- `freyja-wdc`
- `wt`
- `wt-wdc`

## Install

From the KOIOS baseline directory:

```bash
cd /scratch/yfw215/MosaicJoin/baselines/koios
python -m pip install -r requirements.txt
```

`run_koios_baseline.py` uses only the Python standard library in its default
CPU mode. Its optional FAISS candidate mode needs a Python FAISS install;
GPU FAISS should be installed with conda, for example `faiss-gpu`.
The remaining requirements are for building `ft.sqlite3` with `build_koios_embedding_db.py`.
The commands below can also be run from the repo root by prefixing script paths
with `baselines/koios/`; dataset shortcuts still write to
`baselines/koios/work/<dataset>/`.

## Build KOIOS Data

Build set files and embeddings for one dataset:

```bash
python build_koios_embedding_db.py \
  --dataset_name freyja \
  --batch_size 512 \
  --device cuda \
  --overwrite
```

This writes:

- `work/autofj/sets/`
- `work/autofj/ft.sqlite3`
- `work/autofj/sets_manifest.json`

Use another shortcut name for other datasets, for example:

```bash
python build_koios_embedding_db.py --dataset_name wt-wdc --device cuda --overwrite
```

On Slurm, request an L40S GPU for the embedding build:

```bash
sbatch build_embeddings.SBATCH wt-wdc
```

## Run Baseline

Run one dataset:

```bash
python run_koios_baseline.py \
  --dataset_name autofj \
  --top_k 50 \
  --alpha 0.7 \
  --max_values_per_set 128 \
  --out_csv koios_ft_autofj.csv
```

`--max_values_per_set 128` keeps the exact matching stage practical on large
columns. Use `--max_values_per_set 0` for the full uncapped sets.

Use the FAISS candidate generator to emulate KOIOS's GPU vector-index stage:

```bash
python run_koios_baseline.py \
  --dataset_name wt-wdc \
  --candidate_mode faiss \
  --faiss_device gpu \
  --faiss_k 2048
```

Smoke test:

```bash
python run_koios_baseline.py \
  --dataset_name autofj \
  --max_datalake_sets 100 \
  --max_values_per_set 2 \
  --max_queries 1 \
  --top_k 1 \
  --out_csv /tmp/koios_smoke.csv
```

## SLURM Runner

`run_eval.SBATCH` requests one L40S and uses FAISS candidate search by default.
Build `ft.sqlite3` first with `build_koios_embedding_db.py` or
`build_embeddings.SBATCH`.
If FAISS is installed in a separate conda env, set `KOIOS_ENV_PREFIX` to that
env path when submitting.

Run a prepared dataset:

```bash
sbatch run_eval.SBATCH autofj
```

Run all supported datasets:

```bash
sbatch run_eval.SBATCH all
```

Useful overrides:

```bash
KOIOS_TOP_K=50 \
KOIOS_ALPHA=0.7 \
KOIOS_MAX_VALUES_PER_SET=128 \
sbatch run_eval.SBATCH wt-wdc
```

Set `KOIOS_FAISS_K` to tune the number of nearest tokens per query value.
Use `KOIOS_CANDIDATE_MODE=all` to disable the FAISS candidate generator.
Use `KOIOS_MAX_VALUES_PER_SET=0` for full uncapped matching.

## Direct Paths

You can also bypass shortcuts:

```bash
python build_koios_embedding_db.py \
  --dataset_dir ../../datasets/freyja \
  --sets_dir work/freyja/sets \
  --out_db work/freyja/ft.sqlite3 \
  --manifest_json work/freyja/sets_manifest.json \
  --device cuda \
  --overwrite

python run_koios_baseline.py \
  --sets_dir work/freyja/sets \
  --db work/freyja/ft.sqlite3 \
  --manifest_json work/freyja/sets_manifest.json \
  --dataset_dir ../../datasets/freyja \
  --out_csv koios_ft_freyja.csv
```

## Notes

The original C++ KOIOS code expects a set directory plus `ft.sqlite3` with
`wv(word TEXT PRIMARY KEY, vec BLOB)`. The build script creates that shape from
MosaicJoin CSV datasets using SentenceTransformer embeddings projected/padded to
KOIOS's fixed 300 dimensions.

The Python runner is intentionally minimal and exact at verification time. Its
FAISS mode recreates KOIOS's vector-search candidate stage, but does not
reimplement KOIOS's full C++ filter pipeline.
