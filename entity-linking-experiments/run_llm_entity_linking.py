"""
Slurm runner for LLM-based entity linking.

This script creates and optionally submits Slurm jobs to run the LLM entity linking
pipeline in parallel across multiple column pairs.
"""

import os
import subprocess
import math
from pathlib import Path
import pandas as pd


def main():
    # Get project root (parent of entity-linking-experiments/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Configuration - Choose benchmark
    # Autofj Benchmark:
    query_results_dir = script_dir / "autofj_query_results_k1024_t0.1_top10_slurm"
    # Freyja Benchmark:
    # query_results_dir = project_root / "query_results_k1024_t0.1_top100_slurm"
    
    # Path relative to project root (for use in Slurm script)
    query_results_dir_relative = query_results_dir.relative_to(project_root)
    
    pairs_per_job = 100  # Number of column pairs per Slurm job
    
    # Count unique column pairs from contributing entities files
    contributing_files = list(query_results_dir.glob("**/contributing_entities/query_*_contributing_entities.csv"))
    num_pairs = len(contributing_files)
    
    if num_pairs == 0:
        print(f"❌ Error: No contributing_entities.csv files found in {query_results_dir}")
        print("   Make sure the query processing has been run first.")
        return 1
    
    num_jobs = math.ceil(num_pairs / pairs_per_job)
    
    print(f"Found {num_pairs} column pairs with contributing entities, creating {num_jobs} jobs")
    
    # Create Slurm script
    slurm_script = query_results_dir / "run_llm_entity_linking.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=llm_entity_linking
#SBATCH --output={query_results_dir_relative}/llm_entity_linking_%a.out
#SBATCH --error={query_results_dir_relative}/llm_entity_linking_%a.err
#SBATCH --array=0-{num_jobs-1}
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

START_INDEX=$((SLURM_ARRAY_TASK_ID * {pairs_per_job}))
END_INDEX=$((START_INDEX + {pairs_per_job} - 1))
if [ $END_INDEX -ge {num_pairs} ]; then
    END_INDEX=$(({num_pairs} - 1))
fi

QUERY_INDICES=""
for i in $(seq $START_INDEX $END_INDEX); do
    QUERY_INDICES="$QUERY_INDICES $i"
done

cd {project_root}
python entity-linking-experiments/llm_entity_linking.py "{query_results_dir_relative}" --query-indices ${{QUERY_INDICES}}
"""
    
    with open(slurm_script, 'w') as f:
        f.write(script_content)
    os.chmod(slurm_script, 0o755)
    
    print(f"📝 SLURM script created: {slurm_script}")
    print(f"\nTo submit: sbatch {slurm_script}")
    print(f"To check status: squeue -u $USER")
    
    # Also print how to run locally for testing
    print(f"\nTo run locally (first 5 pairs):")
    print(f"  cd {project_root}")
    print(f"  python entity-linking-experiments/llm_entity_linking.py {query_results_dir_relative} --query-indices 0 1 2 3 4")
    
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
