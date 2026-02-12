#!/usr/bin/env python3
"""
Parallel Embedding Building Script
Generates and executes commands for building embeddings in parallel chunks.
"""

import os
import shutil
from pathlib import Path
import argparse

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


def main(experiment: str, embedding_model: str = "embeddinggemma"):

    exp_dir = f"{experiment}-experiments"
    datalake_dir = f"datasets/{experiment}/datalake"
    output_dir = f"{exp_dir}/{experiment}_offline_data_{embedding_model}"

    num_chunks = 10
    device = "auto"
    
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", choices=["autofj", "wt", "freyja", "gdc", "autofj-wdc", "wt-wdc", "freyja-wdc"], type=str, required=True)
    argparser.add_argument("--embedding_model", choices=["embeddinggemma", "mpnet"], default="embeddinggemma", type=str, required=False)
    args = argparser.parse_args()
    main(args.experiment, args.embedding_model)
