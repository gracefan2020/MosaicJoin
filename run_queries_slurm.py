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

def main():
    # Configuration
    top_k_return = 50
    similarity_threshold = 0.1  # For column-level matching
    sketch_size = 128
    queries_per_job = 10
    similarity_method = "chamfer"  # "mean", "greedy_match", "top_k_mean", "max", "chamfer"
    embedding_model = "embeddinggemma"
    embedding_dim = 128  # Output dimension for embeddinggemma model
    d_sketch_size = 128
    
    # # For Freyja experiments
    # datalake_dir = "datasets/freyja-semantic-join/datalake/singletons"
    # sketches_dir = f"freyja-experiments/freyja_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    # query_file = "datasets/freyja-semantic-join/freyja_single_query_columns_no_column_names.csv"
    # embeddings_dir = f"freyja-experiments/freyja_offline_data_{embedding_model}/embeddings"
    # output_dir = f"freyja-experiments/freyja_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}_slurm"
    # if similarity_method == "inverse_chamfer":
    #     output_dir = f"freyja-experiments/freyja_query_results_{embedding_model}_D{d_sketch_size}_{similarity_method}_top{top_k_return}_slurm"

    # # For AutoFJ experiments
    # datalake_dir = "datasets/autofj_join_benchmark/datalake"
    # sketches_dir = f"autofj-experiments/autofj_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    # query_file = "datasets/autofj_join_benchmark/autofj_query_columns.csv"
    # embeddings_dir = f"autofj-experiments/autofj_offline_data_{embedding_model}/embeddings"
    # output_dir = f"autofj-experiments/autofj_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}_slurm"

    # For AutoFJ-WDC experiments
    datalake_dir = "datasets/autofj-wdc/datalake"
    sketches_dir = f"autofj-wdc-experiments/autofj-wdc_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    query_file = "datasets/autofj-wdc/autofj_query_columns.csv"
    embeddings_dir = f"autofj-wdc-experiments/autofj-wdc_offline_data_{embedding_model}/embeddings"
    output_dir = f"autofj-wdc-experiments/autofj-wdc_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}_slurm"

    # # For GDC experiments
    # datalake_dir = "datasets/gdc-breakdown/datalake"
    # sketches_dir = f"gdc-experiments/gdc_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    # query_file = "datasets/gdc-breakdown/gdc_breakdown_query_columns.csv"
    # embeddings_dir = f"gdc-experiments/gdc_offline_data_{embedding_model}/embeddings"
    # output_dir = f"gdc-experiments/gdc_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}_slurm"
    
    # # For AutoFJ+GDC experiments
    # datalake_dir = "datasets/autofj-gdc/datalake"
    # sketches_dir = f"autofj-gdc-experiments/autofj-gdc_offline_data/sketches_k{d_sketch_size}"
    # query_file = "datasets/autofj-gdc/autofj_query_columns.csv"
    # embeddings_dir = "autofj-gdc-experiments/autofj-gdc_offline_data/embeddings"
    # output_dir = f"autofj-gdc-experiments/autofj-gdc_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    
    # # For GDC+AutoFJ (with GDC breakdown / GDC GT)
    # datalake_dir = "datasets/gdc-autofj/datalake"
    # sketches_dir = f"gdc-autofj-experiments/gdc-autofj_offline_data/sketches_k{d_sketch_size}"
    # query_file = "datasets/gdc-autofj/gdc_breakdown_query_columns.csv"
    # embeddings_dir = "gdc-autofj-experiments/gdc-autofj_offline_data/embeddings"
    # output_dir = f"gdc-autofj-experiments/gdc-autofj_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    

    # # For GDC+Freyja (with GDC breakdown / GDC GT)
    # datalake_dir = "datasets/gdc-freyja/datalake"
    # sketches_dir = f"gdc-freyja-experiments/gdc-freyja_offline_data/sketches_k{d_sketch_size}"
    # query_file = "datasets/gdc-freyja/gdc_breakdown_query_columns.csv"
    # embeddings_dir = "gdc-freyja-experiments/gdc-freyja_offline_data/embeddings"
    # output_dir = f"gdc-freyja-experiments/gdc-freyja_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    
    # # For WT
    # datalake_dir = "datasets/wt/datalake_no_column_names"
    # sketches_dir = f"wt-experiments/wt_offline_data_{embedding_model}_no_column_names/sketches_k{d_sketch_size}"
    # query_file = "datasets/wt/wt_query_columns_no_column_names.csv"
    # embeddings_dir = f"wt-experiments/wt_offline_data_{embedding_model}_no_column_names/embeddings"
    # output_dir = f"wt-experiments/wt_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}"
    

    # # For WT+AutoFJ
    # datalake_dir = "datasets/wt-autofj/datalake_no_column_names"
    # sketches_dir = f"wt-autofj-experiments/wt-autofj_offline_data_no_column_names/sketches_k{d_sketch_size}"
    # query_file = "datasets/wt-autofj/wt_query_columns_no_column_names.csv"
    # embeddings_dir = "wt-autofj-experiments/wt-autofj_offline_data_no_column_names/embeddings"
    # output_dir = f"wt-autofj-experiments/wt-autofj_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm_no_column_names"
    

    """
    Snoopy datasets
    """
    # # For WikiTable
    
    # exp_dir = "wikitable-experiments"
    # datalake_dir = "datasets/WikiTable/datalake"
    # sketches_dir = exp_dir + f"/wikitable_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    # query_file = "datasets/WikiTable/autofj_query_columns.csv"
    # embeddings_dir = exp_dir + f"/wikitable_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/wikitable_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}"
    
    # # For WDC
    # embedding_model = "embeddinggemma"
    # d_sketch_size = 128
    # exp_dir = "wdc-experiments"
    # datalake_dir = "datasets/WDC/datalake"
    # sketches_dir = exp_dir + f"/wdc_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    # query_file = "datasets/WDC/autofj_query_columns.csv"
    # embeddings_dir = exp_dir + f"/wdc_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/wdc_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}"

    # # For opendata
    # embedding_model = "embeddinggemma"
    # embedding_dim = 128
    # d_sketch_size = 128
    # exp_dir = "opendata-experiments"
    # datalake_dir = "datasets/opendata/datalake"
    # sketches_dir = exp_dir + f"/opendata_offline_data_{embedding_model}/sketches_k{d_sketch_size}"
    # query_file = "datasets/opendata/autofj_query_columns.csv"
    # embeddings_dir = exp_dir + f"/opendata_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/opendata_query_results_{embedding_model}_D{d_sketch_size}_Q{sketch_size}_{similarity_method}_top{top_k_return}"


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
#SBATCH --job-name=semantic_query
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
    --similarity-threshold {similarity_threshold} \\
    --similarity-method "{similarity_method}" \\
    --sketch-size {sketch_size} \\
    --device {device_setting} \\
    --embeddings-dir "{embeddings_dir}" \\
    --embedding-model "{embedding_model}" \\
    --embedding-dim {embedding_dim} \\
    --query-indices ${{QUERY_INDICES}}
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
    exit(main())

