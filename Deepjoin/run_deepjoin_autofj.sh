#!/usr/bin/env bash
# Make sure to run: conda activate deepjoin38

# Set CUDA_VISIBLE_DEVICES to use the allocated GPU
export CUDA_VISIBLE_DEVICES=0

python ./run_deepjoin.py \
  --dataset_root ../datasets/autofj_join_benchmark \
  --out_dir ./output-autofj \
  --model_dir sentence-transformers/all-mpnet-base-v2 \
  --lake_subdir datalake \
  --queries_subdir datalake \
  --split_lake 8 \
  --split_queries 8 \
  --K 10 \
  --N 10 \
  --threshold 0.1 \
  --ground_truth ../datasets/autofj_join_benchmark/groundtruth-joinable.csv