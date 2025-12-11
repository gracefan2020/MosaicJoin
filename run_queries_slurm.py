#!/usr/bin/env python3
"""
Slurm-based Query Processing Script
Submits query processing jobs using Slurm with chunking support.

This script divides queries into chunks and submits them as separate Slurm jobs.
Each job processes a subset of queries, enabling parallel processing.
"""

import os
import sys
import subprocess
import math
from pathlib import Path
import pandas as pd

def count_queries(query_file: str) -> int:
    """Count the number of queries in the file."""
    df = pd.read_csv(query_file)
    return len(df)

def submit_slurm_jobs(
    num_queries: int,
    queries_per_job: int,
    query_file: str,
    datalake_dir: str,
    sketches_dir: str,
    output_dir: str,
    top_k_return: int,
    similarity_threshold: float,
    sketch_size: int,
    device: str,
    embeddings_dir: str,
    use_deepjoin: bool,
    deepjoin_params: dict
):
    """Submit Slurm jobs for query processing."""
    
    num_jobs = math.ceil(num_queries / queries_per_job)
    
    print(f"🔧 Configuration:")
    print(f"  Total queries: {num_queries}")
    print(f"  Queries per job: {queries_per_job}")
    print(f"  Number of jobs: {num_jobs}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create SLURM command
    slurm_script = Path(output_dir) / "run_slurm_jobs.sh"
    
    # Build the base command without query indices
    base_cmd = f"""python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold {similarity_threshold} \\
    --sketch-size {sketch_size} \\
    --device "{device}" \\
    --embeddings-dir "{embeddings_dir}" \\
    --sort-by "similarity_score"
    """
    
    # Add DeepJoin parameters if enabled
    if use_deepjoin:
        base_cmd += " \\\n    --use-deepjoin-index"
        if deepjoin_params['embeddings_path']:
            base_cmd += f" \\\n    --deepjoin-embeddings-path \"{deepjoin_params['embeddings_path']}\""
        if deepjoin_params['query_embeddings_path']:
            base_cmd += f" \\\n    --deepjoin-query-embeddings-path \"{deepjoin_params['query_embeddings_path']}\""
        if deepjoin_params['index_path']:
            base_cmd += f" \\\n    --deepjoin-index-path \"{deepjoin_params['index_path']}\""
        base_cmd += f" \\\n    --deepjoin-scale {deepjoin_params['scale']}"
        base_cmd += f" \\\n    --deepjoin-encoder {deepjoin_params['encoder']}"
        base_cmd += f" \\\n    --deepjoin-candidate-limit {deepjoin_params['candidate_limit']}"
        base_cmd += f" \\\n    --deepjoin-top-k {deepjoin_params['top_k']}"
        base_cmd += f" \\\n    --deepjoin-threshold {deepjoin_params['threshold']}"
    
    # Create bash script for array job
    # Build the Python command with appropriate arguments
    deepjoin_args = ""
    if use_deepjoin:
        deepjoin_args = f""" \\
    --use-deepjoin-index \\
    --deepjoin-embeddings-path "{deepjoin_params['embeddings_path']}" \\
    --deepjoin-query-embeddings-path "{deepjoin_params['query_embeddings_path']}" \\
    --deepjoin-scale {deepjoin_params['scale']} \\
    --deepjoin-encoder {deepjoin_params['encoder']} \\
    --deepjoin-candidate-limit {deepjoin_params['candidate_limit']} \\
    --deepjoin-top-k {deepjoin_params['top_k']} \\
    --deepjoin-threshold {deepjoin_params['threshold']}"""
    
    array_script = f"""#!/bin/bash
#SBATCH --job-name=semantic_query
#SBATCH --output={output_dir}/slurm_%a.out
#SBATCH --error={output_dir}/slurm_%a.err
#SBATCH --array=0-{num_jobs-1}
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Calculate query range for this job
START_INDEX=$((SLURM_ARRAY_TASK_ID * {queries_per_job}))
END_INDEX=$((START_INDEX + {queries_per_job} - 1))

# Adjust END_INDEX if it exceeds the number of queries
if [ $END_INDEX -ge {num_queries} ]; then
    END_INDEX=$(({num_queries} - 1))
fi

echo "Job $SLURM_ARRAY_TASK_ID: Processing queries $START_INDEX to $END_INDEX"
echo "Total queries in job: $((END_INDEX - START_INDEX + 1))"

# Build query indices list
QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

# Run query processing with specific query indices
python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}/job_$SLURM_ARRAY_TASK_ID" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold {similarity_threshold} \\
    --sketch-size {sketch_size} \\
    --device "{device}" \\
    --embeddings-dir "{embeddings_dir}" \\
    --sort-by "similarity_score" \\
    --query-indices ${{QUERY_INDICES}}{deepjoin_args}

echo "Job $SLURM_ARRAY_TASK_ID completed successfully"
"""
    
    # Write SLURM script
    with open(slurm_script, 'w') as f:
        f.write(array_script)
    
    os.chmod(slurm_script, 0o755)
    
    print(f"📝 SLURM script created: {slurm_script}")
    print()
    print("To submit the jobs, run:")
    print(f"  sbatch {slurm_script}")
    print()
    print("To check job status:")
    print("  squeue -u $USER")
    print()
    print("To view job output:")
    print(f"  tail -f {output_dir}/slurm_*.out")
    print()
    print("To view job errors:")
    print(f"  tail -f {output_dir}/slurm_*.err")
    print()
    
    # Ask if user wants to submit now
    response = input("Submit jobs now? (y/n): ")
    if response.lower() == 'y':
        result = subprocess.run(f"sbatch {slurm_script}", shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Jobs submitted successfully!")
            print("\nTo monitor progress:")
            print("  squeue -u $USER")
            print(f"  watch -n 5 'tail -n 20 {output_dir}/slurm_*.out'")
        else:
            print("❌ Error submitting jobs:")
            print(result.stderr)
    else:
        print("Jobs not submitted. Run the command above when ready.")

def main():
    """Main function."""
    # Query parameters
    top_k_return = 50
    similarity_threshold = 0.1
    sketch_size = 1024
    device = "auto"
    queries_per_job = 10  # Process 10 queries per SLURM job
    
    # DeepJoin integration options
    use_deepjoin_index = False  # Disabled as requested
    deepjoin_embeddings_path = "Deepjoin/output/freyja_lake_embeddings_frequent.pkl"
    deepjoin_query_embeddings_path = "Deepjoin/output/freyja_queries_embeddings_frequent.pkl"
    deepjoin_index_path = None
    deepjoin_scale = 1.0
    deepjoin_encoder = "sherlock"
    deepjoin_candidate_limit = 100
    deepjoin_top_k = 500
    deepjoin_threshold = 0.6
    
    # Paths
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    sketches_dir = f"offline_data/sketches_k{sketch_size}"
    query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    embeddings_dir = "offline_data/embeddings"
    
    # Generate output directory name
    deepjoin_suffix = f"_deepjoin_N{deepjoin_candidate_limit}_K{deepjoin_top_k}_T{deepjoin_threshold}" if use_deepjoin_index else ""
    output_dir = f"query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}_similarity_score{deepjoin_suffix}_slurm"
    
    # Check if required files exist
    if not Path(query_file).exists():
        print(f"❌ Error: Query file not found: {query_file}")
        return 1
    
    # Count queries
    num_queries = count_queries(query_file)
    print(f"Found {num_queries} queries in {query_file}")
    
    # Prepare DeepJoin parameters
    deepjoin_params = {
        'embeddings_path': deepjoin_embeddings_path,
        'query_embeddings_path': deepjoin_query_embeddings_path,
        'index_path': deepjoin_index_path,
        'scale': deepjoin_scale,
        'encoder': deepjoin_encoder,
        'candidate_limit': deepjoin_candidate_limit,
        'top_k': deepjoin_top_k,
        'threshold': deepjoin_threshold
    }
    
    # Submit jobs
    submit_slurm_jobs(
        num_queries=num_queries,
        queries_per_job=queries_per_job,
        query_file=query_file,
        datalake_dir=datalake_dir,
        sketches_dir=sketches_dir,
        output_dir=output_dir,
        top_k_return=top_k_return,
        similarity_threshold=similarity_threshold,
        sketch_size=sketch_size,
        device=device,
        embeddings_dir=embeddings_dir,
        use_deepjoin=use_deepjoin_index,
        deepjoin_params=deepjoin_params
    )
    
    return 0

if __name__ == "__main__":
    exit(main())

