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

def main():
    # Configuration
    top_k_return = 50
    similarity_threshold = 0.1  # For column-level matching
    sketch_size = 1024
    queries_per_job = 10
    
    # For Freyja experiments
    datalake_dir = "datasets/freyja-semantic-join/datalake/singletons"
    sketches_dir = f"freyja-experiments/freyja_offline_data/sketches_k{sketch_size}"
    query_file = "datasets/freyja-semantic-join/freyja_single_query_columns_no_column_names.csv"
    embeddings_dir = "freyja-experiments/freyja_offline_data/embeddings"
    output_dir = f"freyja-experiments/freyja_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"


    # datalake_dir = "datasets/freyja-semantic-join/datalake"
    # sketches_dir = f"offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    # embeddings_dir = "offline_data/embeddings"
    # output_dir = f"query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"

    # # For AutoFJ experiments
    # datalake_dir = "datasets/autofj_join_benchmark/datalake"
    # sketches_dir = f"autofj-experiments/autofj_offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/autofj_join_benchmark/autofj_query_columns.csv"
    # embeddings_dir = "autofj-experiments/autofj_offline_data/embeddings"
    # output_dir = f"autofj-experiments/autofj_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"

    # # For GDC experiments
    # datalake_dir = "datasets/gdc-breakdown/datalake"
    # sketches_dir = f"gdc-experiments/gdc_offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/gdc-breakdown/gdc_breakdown_query_columns.csv"
    # embeddings_dir = "gdc-experiments/gdc_offline_data/embeddings"
    # output_dir = f"gdc-experiments/gdc_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    
    # # For AutoFJ+GDC experiments
    # datalake_dir = "datasets/autofj-gdc/datalake"
    # sketches_dir = f"autofj-gdc-experiments/autofj-gdc_offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/autofj-gdc/autofj_query_columns.csv"
    # embeddings_dir = "autofj-gdc-experiments/autofj-gdc_offline_data/embeddings"
    # output_dir = f"autofj-gdc-experiments/autofj-gdc_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    
    # # For GDC+AutoFJ (with GDC breakdown / GDC GT)
    # datalake_dir = "datasets/gdc-autofj/datalake"
    # sketches_dir = f"gdc-autofj-experiments/gdc-autofj_offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/gdc-autofj/gdc_breakdown_query_columns.csv"
    # embeddings_dir = "gdc-autofj-experiments/gdc-autofj_offline_data/embeddings"
    # output_dir = f"gdc-autofj-experiments/gdc-autofj_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    

    # # For GDC+Freyja (with GDC breakdown / GDC GT)
    # datalake_dir = "datasets/gdc-freyja/datalake"
    # sketches_dir = f"gdc-freyja-experiments/gdc-freyja_offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/gdc-freyja/gdc_breakdown_query_columns.csv"
    # embeddings_dir = "gdc-freyja-experiments/gdc-freyja_offline_data/embeddings"
    # output_dir = f"gdc-freyja-experiments/gdc-freyja_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    
    # # For WT
    # datalake_dir = "datasets/wt/datalake_no_column_names"
    # sketches_dir = f"wt-experiments/wt_offline_data_no_column_names/sketches_k{sketch_size}"
    # query_file = "datasets/wt/wt_query_columns_no_column_names.csv"
    # embeddings_dir = "wt-experiments/wt_offline_data_no_column_names/embeddings"
    # output_dir = f"wt-experiments/wt_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm_no_column_names"
    

    # # For WT+AutoFJ
    # datalake_dir = "datasets/wt-autofj/datalake_no_column_names"
    # sketches_dir = f"wt-autofj-experiments/wt-autofj_offline_data_no_column_names/sketches_k{sketch_size}"
    # query_file = "datasets/wt-autofj/wt_query_columns_no_column_names.csv"
    # embeddings_dir = "wt-autofj-experiments/wt-autofj_offline_data_no_column_names/embeddings"
    # output_dir = f"wt-autofj-experiments/wt-autofj_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm_no_column_names"
    


    # # For AutoFJ+SANTOS Small experiments
    # datalake_dir = "datasets/autofj-santos-small/datalake"
    # sketches_dir = f"autofj-santos-experiments/autofj-santos_offline_data/sketches_k{sketch_size}"
    # query_file = "datasets/autofj-santos-small/autofj_query_columns.csv"
    # embeddings_dir = "autofj-santos-experiments/autofj-santos_offline_data/embeddings"
    # output_dir = f"autofj-santos-experiments/autofj-santos_query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_slurm"
    

    # Validate query file
    if not Path(query_file).exists():
        print(f"❌ Error: Query file not found: {query_file}")
        return 1
    
    # Count queries and calculate jobs
    num_queries = len(pd.read_csv(query_file))
    num_jobs = math.ceil(num_queries / queries_per_job)
    
    print(f"Found {num_queries} queries, creating {num_jobs} jobs")
    
    # Create output directory and script
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slurm_script = Path(output_dir) / "run_slurm_jobs.sh"
    
    # Generate Slurm batch script
    script_content = f"""#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output={output_dir}/slurm_%a.out
#SBATCH --error={output_dir}/slurm_%a.err
#SBATCH --array=0-{num_jobs-1}
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

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
    --similarity-method "greedy_match" \\
    --sketch-size {sketch_size} \\
    --device auto \\
    --embeddings-dir "{embeddings_dir}" \\
    --query-indices ${{QUERY_INDICES}}
"""
    
    with open(slurm_script, 'w') as f:
        f.write(script_content)
    os.chmod(slurm_script, 0o755)
    
    print(f"📝 SLURM script created: {slurm_script}")
    print(f"\nTo submit: sbatch {slurm_script}")
    print(f"To check status: squeue -u $USER")
    
    if input("\nSubmit jobs now? (y/n): ").lower() == 'y':
        result = subprocess.run(f"sbatch {slurm_script}", shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Jobs submitted successfully!")
        else:
            print(f"❌ Error: {result.stderr}")
    
    return 0

if __name__ == "__main__":
    exit(main())

