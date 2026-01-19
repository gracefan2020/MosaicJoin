import os
import subprocess
import math
from pathlib import Path
import pandas as pd

def main():
    # # Freyja Benchmark:
    # query_results_dir = "query_results_k1024_t0.1_top100_slurm"
    # datalake_dir = "datasets/freyja-semantic-join/datalake"
    # Autofj Benchmark:
    query_results_dir = "entity-linking-experiments/autofj_query_results_k1024_t0.1_top10_slurm"
    datalake_dir = "datasets/autofj_join_benchmark/datalake"
    queries_per_job = 10
    
    # Count unique queries from results
    results_file = Path(query_results_dir) / "all_query_results.csv"
    if not results_file.exists():
        print(f"❌ Error: Results file not found: {results_file}")
        return 1
    
    df = pd.read_csv(results_file)
    num_queries = df[['query_table', 'query_column']].drop_duplicates().shape[0]
    num_jobs = math.ceil(num_queries / queries_per_job)
    
    print(f"Found {num_queries} queries, creating {num_jobs} jobs")
    
    # Create Slurm script
    slurm_script = Path(query_results_dir) / "run_llm_jobs.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=llm_postprocessing
#SBATCH --output={query_results_dir}/llm_%a.out
#SBATCH --error={query_results_dir}/llm_%a.err
#SBATCH --array=0-{num_jobs-1}
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

START_INDEX=$((SLURM_ARRAY_TASK_ID * {queries_per_job}))
END_INDEX=$((START_INDEX + {queries_per_job} - 1))
if [ $END_INDEX -ge {num_queries} ]; then
    END_INDEX=$(({num_queries} - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

python llm_postprocessing.py "{query_results_dir}" --datalake-dir "{datalake_dir}" --query-indices ${{QUERY_INDICES}}
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