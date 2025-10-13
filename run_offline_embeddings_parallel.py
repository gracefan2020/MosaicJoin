#!/usr/bin/env python3
"""
Parallel Embedding Building Script
Generates and executes commands for building embeddings in parallel chunks.
"""

import os
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

def main():
    # Configuration
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    output_dir = "offline_data"
    num_chunks = 4
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
        
        cmd = f"""python run_offline_processing.py "{datalake_dir}" \\
    --output-dir "{output_dir}_chunk_{i}" \\
    --device "{device}" \\
    --tables {' '.join(f'"{table}"' for table in tables_with_ext)} \\
    --embeddings-only"""
        
        commands.append(cmd)
        print(f"\n# Chunk {i} ({len(chunk_tables)} tables)")
        print(cmd)
    
    # Execute commands in parallel
    print(f"\nExecuting {len(commands)} chunks in parallel...")
    
    for i, cmd in enumerate(commands, 1):
        print(cmd)
        slurm_cmd = 'sbatch -c 1 -G 1 -J embedding-chunk-%i --tasks-per-node=1 --output=embedding_chunk_%i.log --wrap="%s"' % (i, i, cmd.replace('"', r'\"'))
        os.system(slurm_cmd)

if __name__ == "__main__":
    main()
