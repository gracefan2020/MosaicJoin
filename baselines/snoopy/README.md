# Snoopy Baseline (Pretrained Inference Only)

This baseline integrates Snoopy's pretrained `Scorpion` model into SemSketch baseline workflow.
It runs **inference only** (no training) and writes evaluator-compatible CSV output.

## Install

From `SemSketch/` root:

```bash
python -m pip install -r baselines/snoopy/requirements.txt
```


## Single run

```bash
python baselines/snoopy/run_snoopy_baseline.py \
  --dataset_dir datasets/autofj \
  --query_source datasets/autofj/query_columns.csv \
  --checkpoint_path baselines/snoopy/checkpoints/opendata/t=0.2_gendata_mat_leveltrained.pth \
  --out_csv baselines/snoopy/snoopy_ft_autofj.csv \
  --index_cache baselines/snoopy/cache/snoopy-autofj.pkl \
  --device auto
```

## SLURM runner

```bash
cd baselines/snoopy
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

Default checkpoint mapping in `run_eval.SBATCH`:

- `autofj -> baselines/snoopy/checkpoints/opendata/t=0.2_gendata_mat_leveltrained.pth`
- `wt -> baselines/snoopy/checkpoints/WikiTable/t=0.2_gendata_mat_leveltrained.pth`
- `autofj-wdc/freyja/freyja-wdc/wt-wdc -> baselines/snoopy/checkpoints/WDC/t=0.2_gendata_mat_leveltrained.pth`

Override checkpoint for a run:

```bash
SNOOPY_CHECKPOINT_PATH=/path/to/model.pth sbatch run_eval.SBATCH autofj
```

or

```bash
sbatch run_eval.SBATCH autofj /path/to/model.pth
```

## Minimal standalone Snoopy script

`run_snoopy_minimal.py` is a bare-bones local script that runs checkpoint inference
directly on Snoopy-style `query.npy` / `target.npy` (no fastText/CSV indexing step):

```bash
python baselines/snoopy/run_snoopy_minimal.py \
  --checkpoint_path baselines/snoopy/checkpoints/WDC/t=0.2_gendata_mat_leveltrained.pth \
  --target_npy baselines/snoopy/precomputed/WDC/target.npy \
  --query_npy baselines/snoopy/precomputed/WDC/query.npy \
  --top_k 25 \
  --out_csv baselines/snoopy/minimal_topk.csv
```

## Output schema

The script always writes:

- `query_table`
- `query_column`
- `candidate_table`
- `candidate_column`
- `similarity_score`

This is directly compatible with `evaluate_retrieval.py`.

## Notes

- Cache payload stores `items`, `embeddings`, and metadata.
- Cache is reused only when metadata matches (dataset, checkpoint signature, precomputed signatures, etc.).
- For merged `*-wdc` datasets, `target_<id>.csv` / `query_<id>.csv` can reuse Snoopy precomputed `.npy` vectors when available.
