#!/usr/bin/env python3
"""
General Evaluation Script for Semantic Join Retrieval Results

Supports both table-level and column-level evaluation with:
- HITS@K, Precision@K, Recall@K, NDCG@K
- All metrics use MACRO-AVERAGING (per-query score averaged over all queries)
- Optional comparison between two methods (e.g., SemSketch vs DeepJoin)

Usage:
    # Single method evaluation
    python evaluate_retrieval.py --results results.csv --ground-truth gt.csv --level column

    # Compare two methods
    python evaluate_retrieval.py --results semsketch.csv --baseline deepjoin.csv --ground-truth gt.csv --level table
"""

import math
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


# =============================================================================
# Ground Truth Loading
# =============================================================================

def load_ground_truth_column_level(gt_file: str) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
    """Load column-level ground truth (e.g., GDC benchmark)."""
    df = pd.read_csv(gt_file)
    ground_truth = defaultdict(set)
    
    for _, row in df.iterrows():
        src_table = str(row['source_table']).replace('.csv', '').lower()
        src_col = str(row['source_column']).lower()
        tgt_table = str(row['target_table']).replace('.csv', '').lower()
        tgt_col = str(row['target_column']).lower()
        ground_truth[(src_table, src_col)].add((tgt_table, tgt_col))
    
    return dict(ground_truth)


def load_ground_truth_table_level(gt_file: str) -> Dict[str, Set[str]]:
    """Load table-level ground truth (e.g., AutoFJ benchmark)."""
    df = pd.read_csv(gt_file)
    ground_truth = defaultdict(set)
    
    if 'left_table' in df.columns:
        for _, row in df.iterrows():
            left = row['left_table'].replace('.csv', '').lower()
            right = row['right_table'].replace('.csv', '').lower()
            ground_truth[left].add(right)
    elif 'target_ds' in df.columns:
        for _, row in df.iterrows():
            query = row['target_ds'].replace('.csv', '').lower()
            candidate = row['candidate_ds'].replace('.csv', '').lower()
            ground_truth[query].add(candidate)
    else:
        raise ValueError(f"Unknown ground truth format. Columns: {df.columns.tolist()}")
    
    return dict(ground_truth)


# =============================================================================
# Results Loading
# =============================================================================

def load_results_column_level(results_file: str, exclude_self: bool = True) -> Dict[Tuple[str, str], List[Tuple[str, str, float]]]:
    """Load column-level results."""
    df = pd.read_csv(results_file)
    results = defaultdict(list)
    
    for _, row in df.iterrows():
        query_table = str(row['query_table']).replace('.csv', '').lower()
        query_column = str(row['query_column']).lower()
        candidate_table = str(row['candidate_table']).replace('.csv', '').lower()
        candidate_column = str(row['candidate_column']).lower()
        score = float(row['similarity_score'])
        
        if exclude_self and query_table == candidate_table and query_column == candidate_column:
            continue
        
        results[(query_table, query_column)].append((candidate_table, candidate_column, score))
    
    for query in results:
        results[query] = sorted(results[query], key=lambda x: -x[2])
    
    return dict(results)


def load_results_table_level(results_file: str, exclude_self: bool = True) -> Dict[str, List[Tuple[str, float]]]:
    """Load table-level results, deduplicating by keeping max score per candidate."""
    df = pd.read_csv(results_file)
    # Use nested dict to track max score per (query, candidate) pair
    results_dict = defaultdict(dict)
    
    for _, row in df.iterrows():
        query_table = str(row['query_table']).replace('.csv', '').lower()
        candidate_table = str(row['candidate_table']).replace('.csv', '').lower()
        score = float(row['similarity_score'])
        
        if exclude_self and query_table == candidate_table:
            continue
        
        # Keep max score for each candidate
        if candidate_table not in results_dict[query_table] or score > results_dict[query_table][candidate_table]:
            results_dict[query_table][candidate_table] = score
    
    # Convert to sorted list format
    results = {}
    for query, candidates in results_dict.items():
        results[query] = sorted([(c, s) for c, s in candidates.items()], key=lambda x: -x[1])
    
    return results


# =============================================================================
# NDCG Computation
# =============================================================================

def compute_dcg(relevance_list: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevance_list[:k]):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(relevance_list: List[int], k: int, num_relevant: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    dcg = compute_dcg(relevance_list, k)
    ideal_relevance = [1] * min(k, num_relevant) + [0] * max(0, k - num_relevant)
    idcg = compute_dcg(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# Metrics Evaluation (Macro-Averaged)
# =============================================================================

def evaluate_metrics(ground_truth: Dict, results: Dict, k_values: List[int], level: str) -> Tuple[Dict, int, Dict]:
    """Evaluate HITS@K, Precision@K, Recall@K, NDCG@K using MACRO-AVERAGING."""
    hits_sum = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    recall_sum = {k: 0.0 for k in k_values}
    ndcg_sum = {k: 0.0 for k in k_values}
    
    total = 0
    precision_at_gt_sum = 0.0
    recall_at_gt_sum = 0.0
    ndcg_at_gt_sum = 0.0
    hits_at_gt_sum = 0.0
    per_query_metrics = {}
    
    for query, expected_set in ground_truth.items():
        if query not in results:
            continue
        total += 1
        
        if not isinstance(expected_set, set):
            expected_set = {expected_set}
        
        gt_size = len(expected_set)
        
        if level == 'column':
            candidates = [(c[0], c[1]) for c in results[query]]
            relevance = [1 if c in expected_set else 0 for c in candidates]
        else:
            candidates = [c[0] for c in results[query]]
            relevance = [1 if c in expected_set else 0 for c in candidates]
        
        # Metrics at |GT| size
        num_correct_at_gt = sum(relevance[:gt_size])
        precision_at_gt = num_correct_at_gt / gt_size
        recall_at_gt = num_correct_at_gt / gt_size
        ndcg_at_gt = compute_ndcg(relevance, gt_size, gt_size)
        hit_at_gt = 1.0 if num_correct_at_gt > 0 else 0.0
        
        precision_at_gt_sum += precision_at_gt
        recall_at_gt_sum += recall_at_gt
        ndcg_at_gt_sum += ndcg_at_gt
        hits_at_gt_sum += hit_at_gt
        
        per_query_metrics[query] = {
            'gt_size': gt_size,
            'num_correct_at_gt': num_correct_at_gt,
            'precision_at_gt': precision_at_gt,
            'recall_at_gt': recall_at_gt,
            'ndcg_at_gt': ndcg_at_gt
        }
        
        for k in k_values:
            num_correct = sum(relevance[:k])
            hits_sum[k] += 1.0 if num_correct > 0 else 0.0
            precision_sum[k] += num_correct / k
            recall_sum[k] += num_correct / gt_size
            ndcg_sum[k] += compute_ndcg(relevance, k, gt_size)
    
    metrics = {
        'hits': {k: hits_sum[k] / total if total > 0 else 0.0 for k in k_values},
        'precision': {k: precision_sum[k] / total if total > 0 else 0.0 for k in k_values},
        'recall': {k: recall_sum[k] / total if total > 0 else 0.0 for k in k_values},
        'ndcg': {k: ndcg_sum[k] / total if total > 0 else 0.0 for k in k_values}
    }
    
    gt_size_metrics = {
        'precision_at_gt': precision_at_gt_sum / total if total > 0 else 0.0,
        'recall_at_gt': recall_at_gt_sum / total if total > 0 else 0.0,
        'ndcg_at_gt': ndcg_at_gt_sum / total if total > 0 else 0.0,
        'hits_at_gt': hits_at_gt_sum / total if total > 0 else 0.0,
        'hits_at_gt_count': int(hits_at_gt_sum),
        'per_query': per_query_metrics
    }
    
    return metrics, total, gt_size_metrics


# =============================================================================
# Output
# =============================================================================

def print_metrics(metrics: Dict, total: int, name: str, k_values: List[int]):
    """Print formatted metrics table."""
    print(f"\n{'=' * 80}")
    print(f"{name} (n={total} queries, macro-averaged)")
    print(f"{'=' * 80}")
    print(f"{'K':>4} {'HITS@K':>14} {'Precision@K':>14} {'Recall@K':>12} {'NDCG@K':>12}")
    print("-" * 70)
    for k in k_values:
        h = metrics['hits'][k]
        p = metrics['precision'][k]
        r = metrics['recall'][k]
        n = metrics['ndcg'][k]
        hit_count = int(h * total)
        print(f"{k:>4} {h:>8.3f} ({hit_count:>3}) {p:>12.3f} {r:>12.3f} {n:>12.3f}")


def print_gt_size_metrics(gt_size_metrics: Dict, total: int):
    """Print metrics at ground truth size."""
    per_query = gt_size_metrics['per_query']
    gt_sizes = [pq['gt_size'] for pq in per_query.values()]
    avg_gt_size = sum(gt_sizes) / len(gt_sizes) if gt_sizes else 0
    
    print(f"\n{'=' * 60}")
    print(f"METRICS AT GROUND TRUTH SIZE (K = |GT| per query)")
    print(f"{'=' * 60}")
    print(f"Average |GT| size: {avg_gt_size:.2f} (min: {min(gt_sizes)}, max: {max(gt_sizes)})")
    print(f"")
    print(f"  HITS@|GT|:      {gt_size_metrics['hits_at_gt']:>8.3f} ({gt_size_metrics['hits_at_gt_count']:>3}/{total})")
    print(f"  Precision@|GT|: {gt_size_metrics['precision_at_gt']:>8.3f}")
    print(f"  Recall@|GT|:    {gt_size_metrics['recall_at_gt']:>8.3f}")
    print(f"  NDCG@|GT|:      {gt_size_metrics['ndcg_at_gt']:>8.3f}")


def print_comparison(metrics1: Dict, metrics2: Dict, name1: str, name2: str, k_values: List[int]):
    """Print comparison table between two methods."""
    print(f"\n{'=' * 75}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 75}")
    
    for metric_name, metric_key in [("HITS@K", "hits"), ("Precision@K", "precision"), 
                                      ("Recall@K", "recall"), ("NDCG@K", "ndcg")]:
        print(f"\n--- {metric_name} ---")
        print(f"{'K':>4} {name1:>12} {name2:>12} {'Δ':>10} {'Winner':>12}")
        print("-" * 55)
        for k in k_values:
            v1 = metrics1[metric_key][k]
            v2 = metrics2[metric_key][k]
            diff = v1 - v2
            winner = name1 if diff > 0.001 else (name2 if diff < -0.001 else "Tie")
            print(f"{k:>4} {v1:>12.3f} {v2:>12.3f} {diff:>+10.3f} {winner:>12}")


def print_multi_comparison(all_metrics: List[Tuple[str, Dict, int, Dict]], k_values: List[int]):
    """Print condensed comparison table with all methods side-by-side for each metric.
    
    Args:
        all_metrics: List of (name, metrics, total, gt_size_metrics) tuples
        k_values: List of K values to evaluate
    """
    if len(all_metrics) < 2:
        return
    
    names = [m[0] for m in all_metrics]
    metrics_list = [m[1] for m in all_metrics]
    gt_metrics_list = [m[3] for m in all_metrics]
    
    # Calculate column width based on method names
    col_width = max(12, max(len(n) for n in names) + 2)
    
    print(f"\n{'=' * (10 + col_width * len(names))}")
    print("CONDENSED COMPARISON (All Methods)")
    print(f"{'=' * (10 + col_width * len(names))}")
    
    for metric_name, metric_key in [("HITS@K", "hits"), ("Precision@K", "precision"), 
                                      ("Recall@K", "recall"), ("NDCG@K", "ndcg")]:
        print(f"\n--- {metric_name} ---")
        header = f"{'K':>4}"
        for name in names:
            header += f" {name:>{col_width}}"
        print(header)
        print("-" * (6 + col_width * len(names)))
        
        for k in k_values:
            row = f"{k:>4}"
            values = [m[metric_key][k] for m in metrics_list]
            best_val = max(values)
            for i, v in enumerate(values):
                # Mark best with asterisk
                marker = "*" if abs(v - best_val) < 0.001 and len(set(values)) > 1 else " "
                row += f" {v:>{col_width-1}.3f}{marker}"
            print(row)
    
    # Print metrics at ground truth size
    print(f"\n{'=' * (10 + col_width * len(names))}")
    print("METRICS AT GROUND TRUTH SIZE (K = |GT| per query)")
    print(f"{'=' * (10 + col_width * len(names))}")
    
    # Show average |GT| size from first method
    if gt_metrics_list[0] and 'per_query' in gt_metrics_list[0]:
        per_query = gt_metrics_list[0]['per_query']
        gt_sizes = [pq['gt_size'] for pq in per_query.values()]
        if gt_sizes:
            print(f"Average |GT| size: {sum(gt_sizes)/len(gt_sizes):.2f} (min: {min(gt_sizes)}, max: {max(gt_sizes)})")
    
    header = f"{'Metric':>16}"
    for name in names:
        header += f" {name:>{col_width}}"
    print(header)
    print("-" * (18 + col_width * len(names)))
    
    for metric_name, metric_key in [("HITS@|GT|", "hits_at_gt"), ("Precision@|GT|", "precision_at_gt"), 
                                      ("Recall@|GT|", "recall_at_gt"), ("NDCG@|GT|", "ndcg_at_gt")]:
        row = f"{metric_name:>16}"
        values = [m.get(metric_key, 0.0) if m else 0.0 for m in gt_metrics_list]
        best_val = max(values) if values else 0.0
        for v in values:
            marker = "*" if abs(v - best_val) < 0.001 and len(set(values)) > 1 else " "
            row += f" {v:>{col_width-1}.3f}{marker}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate semantic join retrieval results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single method
  python evaluate_retrieval.py --results results.csv --ground-truth gt.csv --level column

  # Compare SemSketch vs DeepJoin
  python evaluate_retrieval.py --results semsketch.csv --baseline deepjoin.csv --ground-truth gt.csv

  # Compare SemSketch vs multiple baselines (condensed output)
  python evaluate_retrieval.py --results semsketch.csv \\
      --baselines deepjoin_base.csv deepjoin_ft.csv \\
      --baseline-names "DeepJoin-Base" "DeepJoin-FT" \\
      --ground-truth gt.csv --level table
        """
    )
    parser.add_argument('--results', type=str, required=True, help='Path to results CSV (SemSketch)')
    parser.add_argument('--baseline', type=str, help='Path to baseline results CSV (single baseline)')
    parser.add_argument('--baselines', type=str, nargs='+', help='Paths to multiple baseline CSVs')
    parser.add_argument('--ground-truth', type=str, required=True, help='Path to ground truth CSV')
    parser.add_argument('--level', type=str, choices=['table', 'column'], default='column',
                       help='Evaluation level (default: column)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 3, 5, 10],
                       help='K values for evaluation (default: 1 3 5 10)')
    parser.add_argument('--name', type=str, default='SemSketch', help='Name for the main results')
    parser.add_argument('--baseline-name', type=str, default='DeepJoin', help='Name for single baseline')
    parser.add_argument('--baseline-names', type=str, nargs='+', help='Names for multiple baselines')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"SEMANTIC JOIN RETRIEVAL EVALUATION ({args.level.upper()}-LEVEL)")
    print("=" * 70)
    
    # Validate paths
    if not Path(args.ground_truth).exists():
        print(f"❌ Ground truth not found: {args.ground_truth}")
        return 1
    
    if not Path(args.results).exists():
        print(f"❌ Results not found: {args.results}")
        return 1
    
    # Load ground truth
    if args.level == 'column':
        gt = load_ground_truth_column_level(args.ground_truth)
        load_results = load_results_column_level
    else:
        gt = load_ground_truth_table_level(args.ground_truth)
        load_results = load_results_table_level
    
    total_pairs = sum(len(v) if isinstance(v, set) else 1 for v in gt.values())
    print(f"\nGround truth: {len(gt)} queries with {total_pairs} total join pairs")
    
    # Evaluate main results
    results = load_results(args.results)
    print(f"{args.name} results: {len(results)} queries")
    metrics, total, gt_size_metrics = evaluate_metrics(gt, results, args.k_values, args.level)
    
    # Collect all metrics for multi-comparison (name, metrics, total, gt_size_metrics)
    all_metrics = [(args.name, metrics, total, gt_size_metrics)]
    
    # Handle multiple baselines (new feature)
    if args.baselines:
        baseline_names = args.baseline_names if args.baseline_names else [f"Baseline-{i+1}" for i in range(len(args.baselines))]
        if len(baseline_names) < len(args.baselines):
            baseline_names.extend([f"Baseline-{i+1}" for i in range(len(baseline_names), len(args.baselines))])
        
        for baseline_path, baseline_name in zip(args.baselines, baseline_names):
            if Path(baseline_path).exists():
                baseline_results = load_results(baseline_path)
                print(f"{baseline_name} results: {len(baseline_results)} queries")
                baseline_metrics, baseline_total, baseline_gt_metrics = evaluate_metrics(
                    gt, baseline_results, args.k_values, args.level
                )
                all_metrics.append((baseline_name, baseline_metrics, baseline_total, baseline_gt_metrics))
            else:
                print(f"⚠️  Baseline not found: {baseline_path}")
        
        # Print condensed multi-comparison table
        print_multi_comparison(all_metrics, args.k_values)
    
    # Handle single baseline (legacy behavior)
    elif args.baseline and Path(args.baseline).exists():
        baseline_results = load_results(args.baseline)
        print(f"\n{args.baseline_name} results: {len(baseline_results)} queries")
        baseline_metrics, baseline_total, baseline_gt_metrics = evaluate_metrics(
            gt, baseline_results, args.k_values, args.level
        )
        print_metrics(metrics, total, args.name, args.k_values)
        print_gt_size_metrics(gt_size_metrics, total)
        print_metrics(baseline_metrics, baseline_total, args.baseline_name, args.k_values)
        print_gt_size_metrics(baseline_gt_metrics, baseline_total)
        
        # Print comparison
        print_comparison(metrics, baseline_metrics, args.name, args.baseline_name, args.k_values)
    elif args.baseline:
        print(f"\n⚠️  Baseline not found: {args.baseline}")
        print_metrics(metrics, total, args.name, args.k_values)
        print_gt_size_metrics(gt_size_metrics, total)
    else:
        # No baselines, just print main results
        print_metrics(metrics, total, args.name, args.k_values)
        print_gt_size_metrics(gt_size_metrics, total)
    
    # Coverage info
    queries_in_gt = set(gt.keys())
    queries_in_results = set(results.keys())
    missing = queries_in_gt - queries_in_results
    if missing:
        print(f"\n⚠️  {len(missing)} ground truth queries not found in results")
    
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
