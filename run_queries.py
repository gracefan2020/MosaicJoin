#!/usr/bin/env python3
"""
Query Processing Runner Script
Simple script to run query processing with common configurations.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    sketches_dir = "offline_data/sketches_k1024"
    query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    output_dir = "query_results"
    
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
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\nQuery processing completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"\nTo evaluate results, run:")
        print(f"python evaluate_semantic_join.py {output_dir}/all_query_results.csv join/Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv datasets/freyja-semantic-join/freyja_ground_truth.csv --output-dir evaluation_results")
    else:
        print(f"\nQuery processing failed with return code {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
