#!/usr/bin/env python3
"""
Evaluation Runner Script
Simple script to run evaluation with common configurations.
"""

import os
import glob
from pathlib import Path

def find_latest_query_results():
    """Find the most recent query results directory."""
    query_dirs = glob.glob("query_results_k*_t*_top*")
    if not query_dirs:
        return "query_results"  # fallback to default
    # Return the most recent one (assuming they're created in order)
    return sorted(query_dirs)[-1]

def main():
    # Configuration
    query_results_dir = find_latest_query_results()
    semantic_results = f"{query_results_dir}/all_query_results.csv"
    deepjoin_results = "Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv"
    # deepjoin_results = "Deepjoin/output/deepjoin_results_T0.7_exact.csv"

    ground_truth = "datasets/freyja-semantic-join/freyja_ground_truth.csv"
    
    # Create matching evaluation directory name
    if query_results_dir.startswith("query_results_"):
        config_suffix = query_results_dir.replace("query_results_", "")
        output_dir = f"evaluation_results_{config_suffix}"
    else:
        output_dir = "evaluation_results"
    
    print(f"Using query results from: {query_results_dir}")
    print(f"Evaluation results will be saved to: {output_dir}")
    print()
    
    # Check if required files exist
    if not Path(semantic_results).exists():
        print(f"Error: Semantic results file not found: {semantic_results}")
        print("Please run query processing first:")
        print("  python run_queries.py")
        return 1
    
    if not Path(deepjoin_results).exists():
        print(f"Error: DeepJoin results file not found: {deepjoin_results}")
        print("Please run DeepJoin evaluation first")
        return 1
    
    if not Path(ground_truth).exists():
        print(f"Error: Ground truth file not found: {ground_truth}")
        return 1
    
    # Build command
    cmd = f"""python evaluate_semantic_join.py \\
    --semantic-results "{semantic_results}" \\
    --deepjoin-results "{deepjoin_results}" \\
    --ground-truth "{ground_truth}" \\
    --output-dir "{output_dir}" """
    
    print("Running evaluation...")
    print(f"Command: {cmd}")
    print()
    
    # Execute command
    os.system(cmd)

if __name__ == "__main__":
    main()
