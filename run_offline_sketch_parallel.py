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

def cleanup_sketch_data(output_dir: str, sketch_size: int):
    """Clean up previous sketch data."""
    
    # Clean sketch directories
    sketches_dir = Path(output_dir) / f"sketches_k{sketch_size}"
    if sketches_dir.exists():
        print(f"Removing previous sketches: {sketches_dir}")
        shutil.rmtree(sketches_dir)

def main():
    embedding_model = "embeddinggemma"
    # Configuration
    # # For Freyja experiments
    # exp_dir = "freyja-experiments"
    # embeddings_dir = exp_dir + f"/freyja_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/freyja_offline_data_{embedding_model}"


    # # For AutoFJ experiments
    # exp_dir = "autofj-experiments"
    # embeddings_dir = f"{exp_dir}/autofj_offline_data_{embedding_model}/embeddings"
    # output_dir = f"{exp_dir}/autofj_offline_data_{embedding_model}"

    # For AutoFJ-WDC experiments
    exp_dir = "autofj-wdc-experiments"
    embeddings_dir = f"{exp_dir}/autofj-wdc_offline_data_{embedding_model}/embeddings"
    output_dir = f"{exp_dir}/autofj-wdc_offline_data_{embedding_model}"

    # # For GDC experiments
    # exp_dir = "gdc-experiments"
    # embeddings_dir = f"{exp_dir}/gdc_offline_data_{embedding_model}/embeddings"
    # output_dir = f"{exp_dir}/gdc_offline_data_{embedding_model}"

    # # For AutoFJ+GDC experiments
    # embeddings_dir = "autofj-gdc-experiments/autofj-gdc_offline_data/embeddings"
    # exp_dir = "autofj-gdc-experiments"
    # output_dir = "autofj-gdc-experiments/autofj-gdc_offline_data"
    # num_chunks = 4

    # # For GDC+AutoFJ (with GDC breakdown / GDC GT)
    # exp_dir = "gdc-autofj-experiments"
    # embeddings_dir = exp_dir + "/gdc-autofj_offline_data/embeddings"
    # output_dir = exp_dir + "/gdc-autofj_offline_data"
    # num_chunks = 4

    # # For GDC+Freyja (with GDC breakdown / GDC GT)
    # exp_dir = "gdc-freyja-experiments"
    # embeddings_dir = exp_dir + "/gdc-freyja_offline_data/embeddings"
    # output_dir = exp_dir + "/gdc-freyja_offline_data"
    # num_chunks = 4

    # # For WT
    # exp_dir = "wt-experiments"
    # embeddings_dir = exp_dir + f"/wt_offline_data_{embedding_model}_no_column_names/embeddings"
    # output_dir = exp_dir + f"/wt_offline_data_{embedding_model}_no_column_names"
    # num_chunks = 4

    # # For WT+AutoFJ
    # exp_dir = "wt-autofj-experiments"
    # embeddings_dir = exp_dir + "/wt-autofj_offline_data_no_column_names/embeddings"
    # output_dir = exp_dir + "/wt-autofj_offline_data_no_column_names"
    # num_chunks = 4

    # # For AutoFJ+SANTOS Small experiments
    # exp_dir = "autofj-santos-experiments"
    # embeddings_dir = "autofj-santos-experiments/autofj-santos_offline_data/embeddings"
    # output_dir = "autofj-santos-experiments/autofj-santos_offline_data"
    # num_chunks = 10

    """
    Snoopy datasets
    """
    # # For WikiTable
    # exp_dir = "wikitable-experiments"
    # embedding_model = "embeddinggemma"
    # embeddings_dir = exp_dir + f"/wikitable_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/wikitable_offline_data_{embedding_model}"

    # # For WDC
    # exp_dir = "wdc-experiments"
    # embedding_model = "embeddinggemma"
    # embeddings_dir = exp_dir + f"/wdc_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/wdc_offline_data_{embedding_model}"

    # # For opendata
    # exp_dir = "opendata-experiments"
    # embedding_model = "embeddinggemma"
    # embeddings_dir = exp_dir + f"/opendata_offline_data_{embedding_model}/embeddings"
    # output_dir = exp_dir + f"/opendata_offline_data_{embedding_model}"

    sketch_size = 128
    num_chunks = 10
    # Clean up previous sketch data
    # cleanup_sketch_data(output_dir, sketch_size)
    
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
            f.write(f'python offline_sketch.py consolidate "{sketches_output_dir}"\n')
            f.write(f'echo "Consolidation complete!"\n')
        
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
    main()
