#!/usr/bin/env python3
"""
Evaluation Runner Script
Simple script to run evaluation with common configurations.
"""

import os
import glob
import argparse
from pathlib import Path

def find_latest_query_results():
    """Find the most recent query results directory."""
    query_dirs = glob.glob("query_results_k*_t*_top*")
    if not query_dirs:
        return "query_results"  # fallback to default
    # Return the most recent one (assuming they're created in order)
    return sorted(query_dirs)[-1]

def main():
    parser = argparse.ArgumentParser(description='Run evaluation with enhanced sample analysis')
    parser.add_argument('--print-samples', action='store_true', 
                       help='Print sample values and metrics to console')
    parser.add_argument('--sample-count', type=int, default=5,
                       help='Number of sample values per column to include')
    parser.add_argument('--no-samples', action='store_true',
                       help='Disable sample analysis (faster execution)')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='Similarity threshold for filtering semantic results (default: 0.7)')
    parser.add_argument('--quick-metrics', action='store_true',
                       help='Only print quick metrics summary, skip detailed analysis')
    parser.add_argument('--analyze-false-positives', action='store_true',
                       help='Only analyze false positives (semantic method)')
    parser.add_argument('--analyze-disagreements', action='store_true',
                       help='Only analyze method disagreements (one right, one wrong)')
    args = parser.parse_args()
    
    # Configuration
    # query_results_dir = find_latest_query_results()
    # query_results_dir = "query_results_k1024_t0.7_top50_deepjoin_N100_K500_T0.6"
    # query_results_dir = "query_results_k1024_t0.7_top50_slurm"
    query_results_dir = "query_results_k1024_t0.7_top50_0"

    semantic_results = f"{query_results_dir}/all_query_results.csv"
    # semantic_results = f"{query_results_dir}/llm_pruned_query_results.csv"

    deepjoin_results = "Deepjoin/output/deepjoin_results_frequent_K50_N20_T0.7.csv"
    # deepjoin_results = "Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv"
    # deepjoin_results = "Deepjoin/output/deepjoin_results_T0.7_exact.csv"

    ground_truth = "datasets/freyja-semantic-join/freyja_ground_truth.csv"
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    
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
    
    if not Path(datalake_dir).exists():
        print(f"Warning: Datalake directory not found: {datalake_dir}")
        print("Sample values and overlap metrics will not be available")
        datalake_dir = None
    
    # Build command
    cmd_parts = [
        "python evaluate_semantic_join.py",
        f"--semantic-results \"{semantic_results}\"",
        f"--deepjoin-results \"{deepjoin_results}\"",
        f"--ground-truth \"{ground_truth}\"",
        f"--output-dir \"{output_dir}\"",
        f"--similarity-threshold {args.similarity_threshold}",
        f"--quick-metrics"
    ]
    
    # # Add analysis mode options
    # if args.quick_metrics:
    #     cmd_parts.append("--quick-metrics")
    # elif args.analyze_false_positives:
    #     cmd_parts.append("--analyze-false-positives")
    # elif args.analyze_disagreements:
    #     cmd_parts.append("--analyze-disagreements")
    
    # Add sample analysis options if not disabled
    if not args.no_samples:
        cmd_parts.append(f"--sample-count {args.sample_count}")
        if args.print_samples:
            cmd_parts.append("--print-samples")
        if datalake_dir:
            cmd_parts.append(f"--datalake-dir \"{datalake_dir}\"")
    
    cmd = " \\\n    ".join(cmd_parts)
    
    print("Running evaluation...")
    print(f"Command: {cmd}")
    print()
    
    # Execute command
    os.system(cmd)

if __name__ == "__main__":
    main()
