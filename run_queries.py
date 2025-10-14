#!/usr/bin/env python3
"""
Query Processing Runner Script
Simple script to run query processing with common configurations.
"""

import os
import shutil
from pathlib import Path

def cleanup_query_data(output_dir: str):
    """Clean up previous query data."""
    print("🧹 Cleaning up previous query data...")
    
    # Clean query results directory
    query_results_dir = Path(output_dir)
    if query_results_dir.exists():
        print(f"Removing previous query results: {query_results_dir}")
        shutil.rmtree(query_results_dir)
        print("✅ Cleaned previous query results")

def main():
    # Configuration
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    sketches_dir = "offline_data/sketches_k1024"
    query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    output_dir = "query_results"
    
    # Clean up previous query data
    cleanup_query_data(output_dir)
    
    # Query parameters
    top_k_return = 50
    similarity_threshold = 0.7
    sketch_size = 1024
    device = "auto"
    
    # Optional embeddings directory (for query sketch building)
    embeddings_dir = "offline_data/embeddings"
    
    # Check if required files exist
    if not Path(datalake_dir).exists():
        print(f"Error: Datalake directory not found: {datalake_dir}")
        print("Please run offline processing first:")
        print("  python run_offline_embeddings_parallel.py")
        print("  python run_offline_sketch_parallel.py")
        return 1
    
    if not Path(sketches_dir).exists():
        print(f"Error: Sketches directory not found: {sketches_dir}")
        print("Please run sketch building first:")
        print("  python run_offline_sketch_parallel.py")
        return 1
    
    if not Path(query_file).exists():
        print(f"Error: Query file not found: {query_file}")
        return 1
    
    # Build command
    cmd = f"""python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold {similarity_threshold} \\
    --sketch-size {sketch_size} \\
    --device "{device}" \\
    --embeddings-dir "{embeddings_dir}" """
    
    print("Running query processing...")
    print(f"Command: {cmd}")
    print()
    
    # Execute command
    os.system(cmd)

if __name__ == "__main__":
    main()
