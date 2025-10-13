#!/usr/bin/env python3
"""
Evaluation Runner Script
Simple script to run evaluation with common configurations.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    semantic_results = "query_results/all_query_results.csv"
    deepjoin_results = "join/Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv"
    ground_truth = "datasets/freyja-semantic-join/freyja_ground_truth.csv"
    output_dir = "evaluation_results"
    
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
    cmd = f"""python evaluate_semantic_join.py "{semantic_results}" "{deepjoin_results}" "{ground_truth}" \\
    --output-dir "{output_dir}" """
    
    print("Running evaluation...")
    print(f"Command: {cmd}")
    print()
    
    # Execute command
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
    else:
        print(f"\nEvaluation failed with return code {result.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
