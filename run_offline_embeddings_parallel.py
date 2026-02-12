#!/usr/bin/env python3
"""
Parallel Embedding Building Script
Generates and executes commands for building embeddings in parallel chunks.
"""

import os
import shutil
from pathlib import Path

def discover_tables(datalake_dir: Path):
    """Discover all CSV tables in the datalake directory."""
    csv_files = list(datalake_dir.glob("*.csv"))
    return sorted([f.stem for f in csv_files])

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

def cleanup_previous_runs(output_dir: str):
    """Clean up all previous run data."""
    
    # Clean main offline data directory
    offline_data_dir = Path(output_dir)
    if offline_data_dir.exists():
        print(f"Removing previous offline data: {offline_data_dir}")
        shutil.rmtree(offline_data_dir)
    
    # Clean any chunk directories
    chunk_dirs = list(Path(".").glob(f"{exp_dir}/{output_dir}_chunk_*"))
    for chunk_dir in chunk_dirs:
        print(f"Removing chunk directory: {chunk_dir}")
        shutil.rmtree(chunk_dir)
    
    # Clean bash script files
    script_files = list(Path(".").glob(f"{exp_dir}/embedding_chunk_*.sh"))
    for script_file in script_files:
        print(f"Removing script file: {script_file}")
        script_file.unlink()

def main():
    embedding_model = "embeddinggemma"

    # Configuration
    # # For Freyja
    # datalake_dir = "datasets/freyja-semantic-join/datalake/singletons"
    # exp_dir = "freyja-experiments"
    # output_dir = f"freyja-experiments/freyja_offline_data_{embedding_model}"

    # # For AutoFuzzyJoin
    # datalake_dir = "datasets/autofj_join_benchmark/datalake"
    # exp_dir = "autofj-experiments"
    # output_dir = f"autofj-experiments/autofj_offline_data_{embedding_model}"

    # For AutoFuzzyJoin-WDC
    datalake_dir = "datasets/autofj-wdc/datalake"
    exp_dir = "autofj-wdc-experiments"
    output_dir = f"autofj-wdc-experiments/autofj-wdc_offline_data_{embedding_model}"

    # # For GDC
    # datalake_dir = "datasets/gdc-breakdown/datalake"
    # exp_dir = "gdc-experiments"
    # output_dir = f"gdc-experiments/gdc_offline_data_{embedding_model}"

    # # For AutoFJ+GDC
    # datalake_dir = "datasets/autofj-gdc/datalake"
    # exp_dir = "autofj-gdc-experiments"
    # output_dir = "autofj-gdc-experiments/autofj-gdc_offline_data"

    # # For GDC+AutoFJ (with GDC breakdown / GDC GT)
    # datalake_dir = "datasets/gdc-autofj/datalake"
    # exp_dir = "gdc-autofj-experiments"
    # output_dir = "gdc-autofj-experiments/gdc-autofj_offline_data"

    # # For GDC+Freyja (with GDC breakdown / GDC GT)
    # datalake_dir = "datasets/gdc-freyja/datalake"
    # exp_dir = "gdc-freyja-experiments"
    # output_dir = "gdc-freyja-experiments/gdc-freyja_offline_data"

    # # For AutoFJ+SANTOS Small
    # datalake_dir = "datasets/autofj-santos-small/datalake"
    # exp_dir = "autofj-santos-experiments"
    # output_dir = "autofj-santos-experiments/autofj-santos_offline_data"

    # # For WT
    # datalake_dir = "datasets/wt/datalake_no_column_names"
    # exp_dir = "wt-experiments"
    # embedding_model = "embeddinggemma"
    # output_dir = f"wt-experiments/wt_offline_data_{embedding_model}_no_column_names"

    # # For WT+AutoFJ
    # datalake_dir = "datasets/wt-autofj/datalake_no_column_names"
    # exp_dir = "wt-autofj-experiments"
    # output_dir = "wt-autofj-experiments/wt-autofj_offline_data_no_column_names"

    """
    Snoopy datasets
    """
    # # For WikiTable
    # datalake_dir = "datasets/WikiTable/datalake"
    # exp_dir = "wikitable-experiments"
    # output_dir = f"{exp_dir}/wikitable_offline_data_{embedding_model}"

    # # For WDC
    # datalake_dir = "datasets/WDC/datalake"
    # exp_dir = "wdc-experiments"
    # output_dir = f"{exp_dir}/wdc_offline_data_{embedding_model}"

    # # For opendata
    # datalake_dir = "datasets/opendata/datalake"
    # exp_dir = "opendata-experiments"
    # output_dir = f"{exp_dir}/opendata_offline_data_{embedding_model}"

    num_chunks = 10
    device = "auto"
    
    # Clean up all previous runs
    # cleanup_previous_runs(output_dir)
    
    # Discover tables
    tables = discover_tables(Path(datalake_dir))
    print(f"Found {len(tables)} tables")
    
    # Split into chunks
    chunks = split_into_chunks(tables, num_chunks)
    
    # Generate and execute commands
    commands = []
    for i, chunk_tables in enumerate(chunks, 1):
        tables_with_ext = [f"{table}.csv" for table in chunk_tables]
        
        cmd = f"""python offline_embedding.py "{datalake_dir}" \\
    --output-dir "{output_dir}/embeddings" \\
    --embedding-model "{embedding_model}" \\
    --embedding-dim 128 \\
    --device "{device}" \\
    --tables {' '.join(f'"{table}"' for table in tables_with_ext)}"""
        
        commands.append(cmd)
        print(f"\n# Chunk {i} ({len(chunk_tables)} tables)")
        # if i > 1:
        #     break
    
    # Execute commands in parallel
    print(f"\nExecuting {len(commands)} chunks in parallel...")
    
    for i, cmd in enumerate(commands, 1):
        # Create bash script for this chunk
        script_filename = f"{exp_dir}/embedding_chunk_{embedding_model}_{i}.sh"
        
        # Write the bash script
        with open(script_filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Chunk {i} embedding script\n")
            f.write(f"{cmd}\n")
        
        # Make the script executable
        os.chmod(script_filename, 0o755)
        
        # Submit the script to SLURM
        slurm_cmd = f'sbatch --account torch_pr_66_general --gres=gpu:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=20GB --time=10:00:00 --output={exp_dir}/embedding_chunk_{embedding_model}_{i}.log {script_filename}'
        
        print(f"Created script: {script_filename}")
        print(f"Running slurm command: {slurm_cmd}")
            
        result = os.system(slurm_cmd)
        if result != 0:
            print(f"ERROR: SLURM submission failed for chunk {i}")
        else:
            print(f"Successfully submitted chunk {i}")

if __name__ == "__main__":
    main()
