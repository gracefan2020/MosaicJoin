#!/usr/bin/env python3
"""
Parallel Sketch Building Script
Generates and executes commands for building sketches in parallel chunks.
Automatically consolidates sketches after all chunks complete.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
import argparse

def discover_embedding_tables(embeddings_dir: Path):
    """Discover all tables that have embeddings."""
    table_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
    return sorted([d.name for d in table_dirs])


def submit_slurm_job(slurm_cmd: str) -> str:
    """Submit a SLURM job and return the job ID."""
    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: SLURM submission failed: {result.stderr}")
        return None
    
    # Parse job ID from output like "Submitted batch job 12345"
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if match:
        return match.group(1)
    return None

def split_into_chunks(items, num_chunks):
    """Split a list of items into roughly equal chunks."""
    chunk_size = len(items) // num_chunks
    remainder = len(items) % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(items[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks


def main(experiment: str, embedding_model: str = "embeddinggemma", sketch_size: int = 128):
    exp_dir = f"{experiment}-experiments"
    embeddings_dir = f"{exp_dir}/{experiment}_offline_data_{embedding_model}/embeddings"
    output_dir = f"{exp_dir}/{experiment}_offline_data_{embedding_model}"

    num_chunks = 10
    
    # Discover tables with embeddings
    tables = discover_embedding_tables(Path(embeddings_dir))
    print(f"Found {len(tables)} tables with embeddings")
    
    # Split into chunks
    chunks = split_into_chunks(tables, num_chunks)
    
    # Generate and execute commands
    commands = []
    for i, chunk_tables in enumerate(chunks, 1):
        cmd = f"""python offline_sketch.py build "{embeddings_dir}" \\
    --output-dir "{output_dir}/sketches_k{sketch_size}" \\
    --sketch-size {sketch_size} \\
    --selection-method "farthest_point" \\
    --tables {' '.join(f'"{table}"' for table in chunk_tables)}"""
        
        commands.append(cmd)
        print(f"\n# Chunk {i} ({len(chunk_tables)} tables)")
        # if i > 1:
        #     break
    
    # Execute commands in parallel
    print(f"\nExecuting {len(commands)} chunks in parallel...")

    job_ids = []
    sketches_output_dir = f"{output_dir}/sketches_k{sketch_size}"
    
    for i, cmd in enumerate(commands, 1):
        # Create bash script for this chunk
        script_filename = f"{exp_dir}/sketch_chunk_sketch_size_{embedding_model}_{sketch_size}_{i}.sh"
        
        # Write the bash script
        with open(script_filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Chunk {i} sketch script\n")
            f.write(f"{cmd}\n")
        
        # Make the script executable
        os.chmod(script_filename, 0o755)
        
        # Submit the script to SLURM
        slurm_cmd = f'sbatch --account torch_pr_66_general --gres=gpu:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=4 --mem=32GB --time=24:00:00 --output={exp_dir}/sketch_chunk_sketch_size_{embedding_model}_{sketch_size}_{i}.log {script_filename}'
        
        print(f"Created script: {script_filename}")
        print(f"Running slurm command: {slurm_cmd}")
            
        job_id = submit_slurm_job(slurm_cmd)
        if job_id:
            job_ids.append(job_id)
            print(f"Successfully submitted chunk {i} (job {job_id})")
        else:
            print(f"ERROR: SLURM submission failed for chunk {i}")
    
    # Submit consolidation job that depends on all sketch building jobs
    if job_ids:
        print(f"\n{'='*60}")
        print("Submitting consolidation job (will run after all chunks complete)...")
        
        # Create consolidation script
        consolidate_script = f"{exp_dir}/consolidate_sketches_{embedding_model}_{sketch_size}.sh"
        with open(consolidate_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Consolidate sketches for faster loading at query time\n")
            f.write(f'echo "Starting sketch consolidation..."\n')
            f.write(f'python offline_sketch.py consolidate "{sketches_output_dir}" --remove-originals\n')
            f.write(f'echo "Consolidation complete! Original folders removed to save disk space."\n')
        
        os.chmod(consolidate_script, 0o755)
        
        # Create dependency string (afterok means only run if all jobs succeed)
        dependency_str = ":".join(job_ids)
        consolidate_slurm_cmd = (
            f'sbatch --account torch_pr_66_general --dependency=afterok:{dependency_str} '
            f'--nodes=1 --tasks-per-node=1 --cpus-per-task=4 --mem=64GB --time=2:00:00 '
            f'--output={exp_dir}/consolidate_sketches_{embedding_model}_{sketch_size}.log '
            f'{consolidate_script}'
        )
        
        print(f"Created consolidation script: {consolidate_script}")
        print(f"Dependency: will run after jobs {', '.join(job_ids)} complete")
        
        consolidate_job_id = submit_slurm_job(consolidate_slurm_cmd)
        if consolidate_job_id:
            print(f"Successfully submitted consolidation job (job {consolidate_job_id})")
            print(f"\nAll jobs submitted! Monitor with: squeue -u $USER")
            print(f"After completion, query times will use the consolidated store automatically.")
        else:
            print("WARNING: Failed to submit consolidation job. Run manually after sketches complete:")
            print(f'  python offline_sketch.py consolidate "{sketches_output_dir}"')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", choices=["autofj", "wt", "freyja", "gdc", "autofj-wdc", "wt-wdc", "freyja-wdc"], type=str, required=True)
    argparser.add_argument("--embedding_model", choices=["embeddinggemma", "mpnet"], default="embeddinggemma", type=str, required=False)
    argparser.add_argument("--sketch_size", type=int, default=128, required=False)
    args = argparser.parse_args()
    main(args.experiment, args.embedding_model, args.sketch_size)

