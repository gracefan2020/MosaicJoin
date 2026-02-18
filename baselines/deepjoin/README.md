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

## Run inference with a query columns CSV
From the repo root:

**GDC Breakdown Dataset**
```bash
python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/gdc-breakdown/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv deepjoin_ft_gdc.csv --with_header --query_dir datasets/gdc-breakdown/gdc_breakdown_query_columns.csv
```

**AutoFJ GDC Dataset**
```bash
python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/autofj-gdc-breakdown/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv deepjoin_ft_autofj_gdc.csv --with_header --query_dir datasets/autofj-gdc-breakdown/autofj_gdc_query_columns.csv
```

**AutoFJ Dataset**
```bash
python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/autofj_join_benchmark/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv deepjoin_ft_autofj.csv --with_header --query_dir datasets/autofj_join_benchmark/groundtruth-joinable.csv
```

## References

- GitHub: https://github.com/mutong184/deepjoin
- Paper: https://dl.acm.org/doi/10.14778/3603581.3603587
