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
    print("🧹 Cleaning up previous runs...")
    
    # Clean main offline data directory
    offline_data_dir = Path(output_dir)
    if offline_data_dir.exists():
        print(f"Removing previous offline data: {offline_data_dir}")
        shutil.rmtree(offline_data_dir)
        print("✅ Cleaned previous offline data")
    
    # Clean any chunk directories
    chunk_dirs = list(Path(".").glob(f"{output_dir}_chunk_*"))
    for chunk_dir in chunk_dirs:
        print(f"Removing chunk directory: {chunk_dir}")
        shutil.rmtree(chunk_dir)
        print(f"✅ Cleaned {chunk_dir}")
    
    
    # Clean log files
    log_files = list(Path(".").glob("*.log")) + list(Path(".").glob("*_chunk_*.log"))
    for log_file in log_files:
        print(f"Removing log file: {log_file}")
        log_file.unlink()
        print(f"✅ Cleaned {log_file}")
    
    print("🎉 Cleanup completed!\n")

def main():
    # Configuration
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    output_dir = "offline_data"
    num_chunks = 32
    device = "auto"
    
    # Clean up all previous runs
    cleanup_previous_runs(output_dir)
    
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
    --device "{device}" \\
    --tables {' '.join(f'"{table}"' for table in tables_with_ext)}"""
        
        commands.append(cmd)
        print(f"\n# Chunk {i} ({len(chunk_tables)} tables)")
        if i > 1:
            break
    
    # Execute commands in parallel
    print(f"\nExecuting {len(commands)} chunks in parallel...")
    
    for i, cmd in enumerate(commands, 1):
        # slurm_cmd = 'sbatch -c 1 -G 1 -J embedding-chunk-%i --tasks-per-node=1 --output=embedding_chunk_%i.log --wrap="%s"' % (i, i, cmd.replace('"', r'\"'))
        # os.system(slurm_cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()
