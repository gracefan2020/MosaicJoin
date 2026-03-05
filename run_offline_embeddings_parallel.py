#!/usr/bin/env python3
"""Run offline embedding building in parallel via SLURM."""

import os
from pathlib import Path
import argparse


def discover_tables(datalake_dir: Path):
    """Discover all CSV tables in the datalake directory."""
    csv_files = list(datalake_dir.glob("*.csv"))
    return sorted([f.stem for f in csv_files])

def split_into_chunks(items, num_chunks):
    """Split into n roughly equal chunks for parallel processing."""
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


def main(experiment: str, embedding_model: str = "embeddinggemma", embedding_dim: int = 128):
    """Submit 10 SLURM jobs, each processing a chunk of tables from datasets/{experiment}/datalake."""
    exp_dir = f"{experiment}-experiments"
    datalake_dir = Path(f"datasets/{experiment}/datalake")
    Path(exp_dir).mkdir(exist_ok=True)

    tables = discover_tables(Path(datalake_dir))
    chunks = split_into_chunks(tables, 10)
    commands = []
    for i, chunk_tables in enumerate(chunks, 1):
        tables_with_ext = [f"{table}.csv" for table in chunk_tables]
        
        cmd = f"""python offline_embedding.py "{datalake_dir}" \\
    --output-dir "{exp_dir}/{experiment}_offline_data_{embedding_model}/embeddings_{embedding_model}_{embedding_dim}" \\
    --embedding-model "{embedding_model}" --embedding-dim {embedding_dim} \\
    --device auto --tables {' '.join(f'"{table}"' for table in tables_with_ext)}"""
        
        commands.append(cmd)

    # Execute commands in parallel
    
    for i, cmd in enumerate(commands, 1):
        script_filename = f"{exp_dir}/embedding_chunk_{embedding_model}_{embedding_dim}_{i}.sh"
        with open(script_filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Chunk {i} embedding script\n")
            f.write(f"{cmd}\n")
        
        # Make the script executable
        os.chmod(script_filename, 0o755)
        
        # Submit the script to SLURM
        slurm_cmd = f'sbatch --account torch_pr_66_general --gres=gpu:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=20GB --time=10:00:00 --output={exp_dir}/embedding_chunk_{embedding_model}_{embedding_dim}_{i}.log {script_filename}'
        result = os.system(slurm_cmd)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", choices=["autofj", "wt", "freyja", "autofj-wdc", "wt-wdc", "freyja-wdc"], type=str, required=True)
    argparser.add_argument("--embedding_model", choices=["embeddinggemma", "mpnet", "bge"], default="embeddinggemma", type=str, required=False)
    argparser.add_argument("--embedding_dim", type=int, default=128, required=False)
    args = argparser.parse_args()
    main(args.experiment, args.embedding_model, args.embedding_dim)
