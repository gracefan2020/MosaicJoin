#!/usr/bin/env python3
"""
Parallel Sketch Building Script
Generates and executes commands for building sketches in parallel chunks.
"""

import os
import shutil
import subprocess
from pathlib import Path

def discover_embedding_tables(embeddings_dir: Path):
    """Discover all tables that have embeddings."""
    table_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
    return sorted([d.name for d in table_dirs])

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
    print("🧹 Cleaning up previous sketch data...")
    
    # Clean sketch directories
    sketches_dir = Path(output_dir) / f"sketches_k{sketch_size}"
    if sketches_dir.exists():
        print(f"Removing previous sketches: {sketches_dir}")
        shutil.rmtree(sketches_dir)
        print("✅ Cleaned previous sketches")
    
    print("🎉 Sketch cleanup completed!\n")

def main():
    # Configuration
    embeddings_dir = "offline_data/embeddings"
    output_dir = "offline_data"
    num_chunks = 2  # Changed from 32 to 2 for testing
    sketch_size = 1024
    similarity_threshold = 0.7
    
    # Clean up previous sketch data
    cleanup_sketch_data(output_dir, sketch_size)
    
    # Discover tables with embeddings
    tables = discover_embedding_tables(Path(embeddings_dir))
    print(f"Found {len(tables)} tables with embeddings")
    
    # Split into chunks
    chunks = split_into_chunks(tables, num_chunks)
    
    # Generate and execute commands
    commands = []
    for i, chunk_tables in enumerate(chunks, 1):
        # Don't add .csv extension since table names are already directory names
        cmd = f"""python offline_sketch.py "{embeddings_dir}" \\
    --output-dir "{output_dir}/sketches_k{sketch_size}" \\
    --sketch-size {sketch_size} \\
    --similarity-threshold {similarity_threshold} \\
    --tables {' '.join(f'"{table}"' for table in chunk_tables)}"""
        
        commands.append(cmd)
        print(f"\n# Chunk {i} ({len(chunk_tables)} tables)")
        if i > 1:
            break
    
    # Execute commands in parallel
    print(f"\nExecuting {len(commands)} chunks in parallel...")
    
    for i, cmd in enumerate(commands, 1):
        # slurm_cmd = 'sbatch -c 1 -G 1 -J sketch-chunk-%i --tasks-per-node=1 --output=sketch_chunk_%i.log --wrap="%s"' % (i, i, cmd.replace('"', r'\"'))
        # os.system(slurm_cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()
