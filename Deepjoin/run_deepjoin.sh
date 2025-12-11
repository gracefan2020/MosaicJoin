#!/usr/bin/env bash
# Make sure to run: conda activate deepjoin38


python ./run_deepjoin.py \
  --dataset_root ../datasets/freyja-semantic-join \
  --out_dir ./output \
  --model_dir sentence-transformers/all-mpnet-base-v2 \
  --lake_subdir datalake \
  --queries_subdir datalake \
  --split_lake 8 \
  --split_queries 4 \
  --K 50 \
  --N 30 \
  --threshold 0.1 \
  --ground_truth ../datasets/freyja-semantic-join/freyja_ground_truth.csv