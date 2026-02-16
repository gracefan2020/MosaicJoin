#!/usr/bin/env python3
"""
Slurm-based Query Processing Script
Submits query processing jobs using Slurm with chunking support.
"""

import os
import subprocess
import math
from pathlib import Path
import pandas as pd
import torch
import argparse

def main(experiment: str, embedding_model: str = "embeddinggemma", d_sketch_size: int = 128, sketch_size: int = 128, similarity_method: str = "chamfer", top_k_return: int = 50, embedding_dim: int = 128, debug_matches: bool = False, debug_top_n: int = 10):
    print(f"Running queries for {experiment} with {embedding_model} embeddings, d_sketch_size {d_sketch_size}, sketch_size {sketch_size}, similarity_method {similarity_method}, top_k_return {top_k_return}")
    exp_dir = f"{experiment}-experiments"
    datalake_dir = f"datasets/{experiment}/datalake"
    query_file = f"datasets/{experiment}/query_columns.csv"

    sketches_dir = f"{exp_dir}/{experiment}_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    embeddings_dir = f"{exp_dir}/{experiment}_offline_data_{embedding_model}/embeddings"
    output_dir = f"{exp_dir}/{experiment}_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}_slurm"
    queries_per_job = 10
    
    # Validate query file
    if not Path(query_file).exists():
        print(f"❌ Error: Query file not found: {query_file}")
        return 1
    
    # Count queries and calculate jobs
    num_queries = len(pd.read_csv(query_file))
    num_jobs = math.ceil(num_queries / queries_per_job)
    
    print(f"Found {num_queries} queries, creating {num_jobs} jobs")
    print(f"Similarity method: {similarity_method}")
    device_setting = "cuda" if torch.cuda.is_available() else "auto"
    
    # Create output directory and script
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slurm_script = Path(output_dir) / "run_slurm_jobs.sh"
    
    # Limit concurrent jobs if using GPU to avoid QOSGrpGRES errors
    array_spec = f"0-{num_jobs-1}"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=query
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output={output_dir}/slurm_%a.out
#SBATCH --error={output_dir}/slurm_%a.err
#SBATCH --array={array_spec}


START_INDEX=$((SLURM_ARRAY_TASK_ID * {queries_per_job}))
END_INDEX=$((START_INDEX + {queries_per_job} - 1))
if [ $END_INDEX -ge {num_queries} ]; then
    END_INDEX=$(({num_queries} - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}/job_$SLURM_ARRAY_TASK_ID" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold 0.1 \\
    --similarity-method "{similarity_method}" \\
    --sketch-size {sketch_size} \\
    --device {device_setting} \\
    --embeddings-dir "{embeddings_dir}" \\
    --embedding-model "{embedding_model}" \\
    --embedding-dim {embedding_dim} \\
    --query-indices ${{QUERY_INDICES}}{"" if not debug_matches else f" --debug-matches --debug-top-n {debug_top_n}"}
"""
    
    with open(slurm_script, 'w') as f:
        f.write(script_content)
    os.chmod(slurm_script, 0o755)
    
    print(f"📝 SLURM script created: {slurm_script}")
    print(f"\nTo submit: sbatch {slurm_script}")
    print(f"To check status: squeue -u $USER")
    
    if input("\nSubmit jobs now? (y/n): ").lower() == 'y':
        result = subprocess.run(f"sbatch --account torch_pr_66_general {slurm_script}", shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Jobs submitted successfully!")
        else:
            print(f"❌ Error: {result.stderr}")
    
    return 0

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", choices=["autofj", "wt", "freyja", "gdc", "autofj-wdc", "wt-wdc", "freyja-wdc"], type=str, required=True)
    argparser.add_argument("--embedding_model", choices=["embeddinggemma", "mpnet"], default="embeddinggemma", type=str, required=False)
    argparser.add_argument("--d_sketch_size", type=int, default=128, required=False)
    argparser.add_argument("--sketch_size", type=int, default=0, required=False)
    argparser.add_argument("--similarity_method", choices=["chamfer", "mean", "greedy_match", "top_k_mean", "max", "inverse_chamfer", "symmetric_chamfer", "harmonic_chamfer"], default="chamfer", type=str, required=False)
    argparser.add_argument("--top_k_return", type=int, default=50, required=False)
    argparser.add_argument("--embedding_dim", type=int, default=128, required=False)
    argparser.add_argument("--debug_matches", action="store_true", help="Print detailed match info for debugging")
    argparser.add_argument("--debug_top_n", type=int, default=10, help="Number of top/worst matches to print")
    args = argparser.parse_args()
    main(args.experiment, args.embedding_model, args.d_sketch_size, args.sketch_size, args.similarity_method, args.top_k_return, args.embedding_dim, args.debug_matches, args.debug_top_n)


