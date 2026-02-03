# DeepJoin Baseline

## Installation
From the repo root:

```bash
python -m pip install -r baselines/deepjoin/requirements.txt
```

## Pull model weights (Git LFS)
From the repo root:

```bash
git lfs pull
```

## Run inference
From the repo root:

```bash
python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/autofj-gdc/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv output.csv
```

## Run inference with a query columns CSV (e.g. On GDC Breakdown dataset)
From the repo root:

```bash
python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/gdc-breakdown/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv deepjoin_ft_gdc.csv --with_header --query_dir datasets/gdc-breakdown/gdc_breakdown_query_columns.csv
```
