#!/usr/bin/env python3
"""AutoFJ DeepJoin Baseline Pipeline"""

import os
import subprocess
import math
import time
import argparse
from pathlib import Path
import pandas as pd
import shutil

DEFAULT_DEEPJOIN_RESULTS = "../Deepjoin/output-autofj/deepjoin_results_K10_N10_T0.1.csv"
DEFAULT_DATALAKE_DIR = "../datasets/autofj_join_benchmark/datalake"
DEFAULT_OUTPUT_DIR = "autofj_deepjoin_baseline_k10_n10_t0.1"
DEFAULT_TARGET_PRECISION = 0.9
DEFAULT_PAIRS_PER_JOB = 50
DEFAULT_QUERY_COLUMNS = "../datasets/autofj_join_benchmark/autofj_query_columns.csv"


def run_sequential(deepjoin_results, datalake_dir, output_dir, target_precision, query_columns):
    """Run AutoFJ baseline generation sequentially."""
    script_dir = Path(__file__).parent
    baseline_script = script_dir / "autofj_deepjoin_baseline.py"
    
    if not baseline_script.exists():
        print(f"ERROR: Baseline script not found: {baseline_script}")
        return 1
    
    cmd = ["python", str(baseline_script), str(deepjoin_results), str(datalake_dir), 
           str(output_dir), str(target_precision)]
    if query_columns:
        cmd.extend(["--query-columns", str(query_columns)])
    return subprocess.run(cmd, cwd=str(script_dir)).returncode


def generate_and_submit_slurm(deepjoin_results, datalake_dir, output_dir, target_precision, pairs_per_job, query_columns):
    """Generate and submit SLURM jobs."""
    script_dir = Path(__file__).parent
    
    deepjoin_df = pd.read_csv(deepjoin_results)
    
    # Filter to valid queries if query_columns is provided
    if query_columns and Path(query_columns).exists():
        query_cols_df = pd.read_csv(query_columns)
        valid_queries = set()
        for _, row in query_cols_df.iterrows():
            query_table = str(row['target_ds']).strip()
            query_col = str(row['target_attr']).strip()
            valid_queries.add((query_table, query_col))
        
        original_count = len(deepjoin_df)
        deepjoin_df = deepjoin_df[
            deepjoin_df.apply(
                lambda row: (str(row['query_table']).strip(), str(row['query_col']).strip()) in valid_queries,
                axis=1
            )
        ]
        filtered_count = len(deepjoin_df)
        print(f"Filtered deepjoin results: {original_count} -> {filtered_count} rows (kept {filtered_count/original_count*100:.1f}%)")
    
    unique_pairs = deepjoin_df[['query_table', 'query_col', 'candidate_table', 'candidate_col']].drop_duplicates()
    num_pairs = len(unique_pairs)
    num_jobs = math.ceil(num_pairs / pairs_per_job)
    
    print(f"Found {num_pairs} unique pairs, creating {num_jobs} SLURM jobs")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pairs_file = output_path / "unique_pairs.csv"
    unique_pairs.to_csv(pairs_file, index=False)
    
    script_dir_abs = script_dir.resolve()
    deepjoin_results_abs = Path(deepjoin_results).resolve()
    datalake_dir_abs = Path(datalake_dir).resolve()
    output_dir_abs = output_path.resolve()
    pairs_file_abs = pairs_file.resolve()
    baseline_script_abs = (script_dir_abs / "autofj_deepjoin_baseline.py").resolve()
    
    # Build query columns argument if provided
    query_columns_abs = None
    if query_columns and Path(query_columns).exists():
        query_columns_abs = Path(query_columns).resolve()
    
    slurm_script = output_path / "run_slurm_jobs.sh"
    
    # Build the command with optional query-columns argument
    cmd_lines = [
        f'python "{baseline_script_abs}"',
        f'    "{deepjoin_results_abs}"',
        f'    "{datalake_dir_abs}"',
        f'    "{output_dir_abs}/job_$SLURM_ARRAY_TASK_ID"',
        f'    "{target_precision}"',
        f'    --start-index $START_INDEX',
        f'    --end-index $END_INDEX',
        f'    --pairs-file "{pairs_file_abs}"'
    ]
    
    if query_columns_abs:
        cmd_lines.append(f'    --query-columns "{query_columns_abs}"')
    
    # Join with backslash for line continuation (except last line)
    cmd_section = ' \\\n'.join(cmd_lines)
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=autofj_baseline
#SBATCH --array=0-{num_jobs-1}
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

START_INDEX=$((SLURM_ARRAY_TASK_ID * {pairs_per_job}))
END_INDEX=$((START_INDEX + {pairs_per_job} - 1))
if [ $END_INDEX -ge {num_pairs} ]; then
    END_INDEX=$(({num_pairs} - 1))
fi

cd "{script_dir_abs}" || exit 1

{cmd_section}
"""
    
    with open(slurm_script, 'w') as f:
        f.write(script_content)
    os.chmod(slurm_script, 0o755)
    
    result = subprocess.run(f"sbatch {slurm_script}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Jobs submitted: {result.stdout.strip()}")
        return True
    else:
        print(f"✗ Error: {result.stderr}")
        return False


def wait_for_jobs(job_name="autofj_baseline"):
    """Wait for SLURM jobs to complete."""
    print("Waiting for jobs to complete...")
    try:
        while True:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-n", job_name, "-h"],
                capture_output=True, text=True
            )
            if not result.stdout.strip():
                print("✓ All jobs completed")
                return True
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted. Combine results manually when jobs complete.")
        return False


def combine_results(output_dir, deepjoin_results=None, query_columns=None):
    """Combine match files from SLURM job directories."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"ERROR: Output directory not found: {output_path}")
        return 1
    
    job_dirs = sorted(output_path.glob("job_*"))
    match_files_moved = 0
    all_query_results_found = False
    
    if job_dirs:
        for job_dir in job_dirs:
            if not job_dir.is_dir():
                continue
            for match_file in job_dir.glob("*_matches.csv"):
                dest = output_path / match_file.name
                if not dest.exists():
                    shutil.move(str(match_file), str(dest))
                    match_files_moved += 1
            
            # Copy all_query_results.csv from first job directory found
            if not all_query_results_found:
                job_results_file = job_dir / "all_query_results.csv"
                if job_results_file.exists():
                    dest_results = output_path / "all_query_results.csv"
                    if not dest_results.exists():
                        shutil.copy2(str(job_results_file), str(dest_results))
                        all_query_results_found = True
            
            try:
                job_dir.rmdir()
            except OSError:
                pass
    
    # Create all_query_results.csv if it doesn't exist (from job dirs or from deepjoin_results)
    results_file = output_path / "all_query_results.csv"
    if not results_file.exists() and deepjoin_results and Path(deepjoin_results).exists():
        deepjoin_df = pd.read_csv(deepjoin_results)
        
        # Filter to valid queries if query_columns is provided
        if query_columns and Path(query_columns).exists():
            query_cols_df = pd.read_csv(query_columns)
            valid_queries = set()
            for _, row in query_cols_df.iterrows():
                query_table = str(row['target_ds']).strip()
                query_col = str(row['target_attr']).strip()
                valid_queries.add((query_table, query_col))
            
            deepjoin_df = deepjoin_df[
                deepjoin_df.apply(
                    lambda row: (str(row['query_table']).strip(), str(row['query_col']).strip()) in valid_queries,
                    axis=1
                )
            ]
        
        semantic_df = deepjoin_df.rename(columns={
            'query_col': 'query_column',
            'candidate_col': 'candidate_column',
            'score': 'similarity_score'
        })
        semantic_df.to_csv(results_file, index=False)
        all_query_results_found = True
    
    match_file_count = len(list(output_path.glob("*_matches.csv")))
    print(f"✓ Combined {match_file_count} match files")
    if results_file.exists():
        print(f"✓ Created all_query_results.csv")
    return 0


def main():
    parser = argparse.ArgumentParser(description="AutoFJ DeepJoin Baseline Pipeline")
    parser.add_argument('mode', nargs='?', default='sequential', choices=['sequential', 'slurm', 'combine', 'pipeline'],
                       help='Execution mode: sequential (default), slurm (submit only), combine, or pipeline (submit+wait+combine)')
    parser.add_argument('--deepjoin-results', default=DEFAULT_DEEPJOIN_RESULTS)
    parser.add_argument('--datalake-dir', default=DEFAULT_DATALAKE_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--target-precision', type=float, default=DEFAULT_TARGET_PRECISION)
    parser.add_argument('--pairs-per-job', type=int, default=DEFAULT_PAIRS_PER_JOB)
    parser.add_argument('--query-columns', default=DEFAULT_QUERY_COLUMNS,
                       help='Path to autofj_query_columns.csv to filter valid queries')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.mode == 'sequential':
        return run_sequential(args.deepjoin_results, args.datalake_dir, args.output_dir, 
                              args.target_precision, args.query_columns)
    
    elif args.mode == 'slurm':
        if generate_and_submit_slurm(args.deepjoin_results, args.datalake_dir, args.output_dir, 
                                     args.target_precision, args.pairs_per_job, args.query_columns):
            print(f"\nAfter jobs complete: python {Path(__file__).name} combine --output-dir {args.output_dir} --query-columns {args.query_columns}")
            return 0
        return 1
    
    elif args.mode == 'combine':
        return combine_results(args.output_dir, args.deepjoin_results, args.query_columns)
    
    elif args.mode == 'pipeline':
        if not generate_and_submit_slurm(args.deepjoin_results, args.datalake_dir, args.output_dir,
                                         args.target_precision, args.pairs_per_job, args.query_columns):
            return 1
        if wait_for_jobs():
            return combine_results(args.output_dir, args.deepjoin_results, args.query_columns)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
