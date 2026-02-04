## To run our method:
1. ```python run_offline_embeddings_parallel.py```

** Make sure to change paths in main function!

2. ```python run_offline_sketch_parallel.py```

** Make sure to change paths in main function!

3. ```python run_queries_slurm.py```

** Make sure to change paths in main function!

4. Merge query results: ```run monitor_and_combine.sh```

-- make sure to change the directory path, and # jobs: NUM_JOBS=${2:-10}

5. Evaluate search results: ```python evaluate_search_results.py```


# Evaluation:
## Run all benchmarks
```
./run_evaluation.sh all
```

## Run specific benchmark
```
./run_evaluation.sh gdc      # GDC only
./run_evaluation.sh autofj   # AutoFJ with DeepJoin comparison
./run_evaluation.sh autofj-gdc   # AutoFJ+GDC benchmark
```

## Run DeepJoin Baseline:
For example, on gdc-breakdown (singleton tables):
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/gdc-breakdown/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv gdc-experiments/deepjoin_ft_gdc.csv --with_header --query_dir datasets/gdc-breakdown/gdc_breakdown_query_columns.csv
```
On AutoFJ tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/autofj_join_benchmark/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv autofj-experiments/deepjoin_ft_autofj_grace.csv --with_header --query_dir datasets/autofj_join_benchmark/autofj_query_columns.csv
```

On AutoFJ+GDC tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/autofj-gdc/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv autofj-gdc-experiments/deepjoin_ft_autofj-gdc.csv --with_header --query_dir datasets/autofj-gdc/autofj_query_columns.csv
```

On GDC+AutoFJ tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/gdc-autofj/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv gdc-autofj-experiments/deepjoin_ft_gdc-autofj.csv --with_header --query_dir datasets/gdc-autofj/gdc_breakdown_query_columns.csv
```

On GDC+Freyja tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/gdc-freyja/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv gdc-freyja-experiments/deepjoin_ft_gdc-freyja.csv --with_header --query_dir datasets/gdc-freyja/gdc_breakdown_query_columns.csv --column_name '*'
```

On AutoFJ+SANTOS tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/autofj-santos-small/datalake/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv autofj-santos-experiments/deepjoin_ft_autofj-santos.csv --with_header --query_dir datasets/autofj-santos-small/autofj_query_columns.csv
```


On Freyja tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/freyja-semantic-join/datalake/singletons/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv freyja-experiments/deepjoin_ft_freyja.csv --with_header --query_dir datasets/freyja-semantic-join/freyja_single_query_columns_no_column_names.csv
```


On WT tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/wt/datalake_no_column_names/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv wt-experiments/deepjoin_ft_wt_no_col_names.csv --with_header --query_dir datasets/wt/wt_query_columns_no_column_names.csv
```



On WT+AutoFJ tables:
```
time python baselines/deepjoin/infer_full_dataset.py --datalake_dir datasets/wt-autofj/datalake_no_column_names/ --model_name baselines/deepjoin/all-mpnet-base-v2 --out_csv wt-autofj-experiments/deepjoin_ft_wt-autofj_no_col_names.csv --with_header --query_dir datasets/wt-autofj/wt_query_columns_no_column_names.csv
```




## To run our LLM variation (deprecated):
1. python run_llm_processing.py
2. python regenerate_pruned_results.py