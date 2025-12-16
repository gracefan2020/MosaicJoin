"""
Evaluation Script for Semantic Join Results

This script evaluates semantic join results against DeepJoin baseline
and ground truth, providing comprehensive comparison metrics.

Usage:
    python evaluate_semantic_join.py \
        --semantic-results query_results/all_query_results.csv \
        --deepjoin-results Deepjoin/output/deepjoin_results.csv \
        --ground-truth datasets/freyja-semantic-join/freyja_ground_truth.csv \
        --output-dir evaluation_results
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np

def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Load ground truth from CSV file (Freyja format)."""
    gt_df = pd.read_csv(gt_path)
    gt_df['target_ds'] = gt_df['target_ds'].str.replace('.csv', '', regex=False).str.lower()
    gt_df['candidate_ds'] = gt_df['candidate_ds'].str.replace('.csv', '', regex=False).str.lower()
    
    by_query: Dict[str, Set[str]] = {}
    for _, r in gt_df.iterrows():
        q = f"{r['target_ds']}.{r['target_attr']}"
        c = f"{r['candidate_ds']}.{r['candidate_attr']}"
        by_query.setdefault(q, set()).add(c)
    
    return by_query

def load_semantic_results(semantic_path: str) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, int]]]:
    """Load semantic join results from CSV file.

    Expected columns (Freyja SemSketch output):
      - query_table, query_column, candidate_table, candidate_column
      - similarity_score (or score)
      - semantic_matches (optional; int)
    """
    df = pd.read_csv(semantic_path)

    # Be tolerant to schema changes
    score_col = 'similarity_score' if 'similarity_score' in df.columns else ('score' if 'score' in df.columns else None)
    if score_col is None:
        raise ValueError(f"Could not find score column in semantic results. Columns: {list(df.columns)}")

    out: Dict[str, List[Tuple[str, float]]] = {}
    semantic_matches: Dict[str, Dict[str, int]] = {}

    for _, r in df.iterrows():
        query_table = str(r['query_table']).replace('.csv', '').lower()
        candidate_table = str(r['candidate_table']).replace('.csv', '').lower()

        q = f"{query_table}.{r['query_column']}"
        c = f"{candidate_table}.{r['candidate_column']}"
        out.setdefault(q, []).append((c, float(r[score_col])))

        if 'semantic_matches' in df.columns:
            try:
                m = int(r['semantic_matches'])
                # Keep max if duplicates exist
                semantic_matches.setdefault(q, {})
                prev = semantic_matches[q].get(c)
                semantic_matches[q][c] = m if prev is None else max(prev, m)
            except Exception:
                # If malformed, just skip
                pass

    # Sort by score descending
    for k in out:
        out[k].sort(key=lambda x: x[1], reverse=True)

    return out, semantic_matches


def remove_self_joins(preds: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    """Remove self-joins from predictions."""
    cleaned: Dict[str, List[Tuple[str, float]]] = {}
    for q, items in preds.items():
        cleaned[q] = [(c, s) for c, s in items if c != q]
    return cleaned


def calculate_metrics(predictions: Dict[str, List[Tuple[str, float]]],
                      ground_truth: Dict[str, Set[str]],
                      k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Dict[str, float]]:
    """Calculate Precision@k, Recall@k, F1@k, and summary metrics."""
    metrics: Dict[str, Dict[str, float]] = {}

    # Average metrics using adaptive k (all predictions per query)
    avg_precision: List[float] = []
    avg_recall: List[float] = []
    avg_f1: List[float] = []
    pred_counts: List[int] = []

    for query, preds in predictions.items():
        if query not in ground_truth:
            continue

        gt_set = ground_truth[query]
        pred_set = set([cand for cand, _ in preds])
        pred_counts.append(len(pred_set))

        if len(pred_set) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            tp = len(pred_set.intersection(gt_set))
            precision = tp / len(pred_set)
            recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_f1.append(f1)

    metrics['Average_Precision'] = {'mean': float(np.mean(avg_precision)) if avg_precision else 0.0,
                                    'std': float(np.std(avg_precision)) if avg_precision else 0.0,
                                    'median': float(np.median(avg_precision)) if avg_precision else 0.0}
    metrics['Average_Recall'] = {'mean': float(np.mean(avg_recall)) if avg_recall else 0.0,
                                 'std': float(np.std(avg_recall)) if avg_recall else 0.0,
                                 'median': float(np.median(avg_recall)) if avg_recall else 0.0}
    metrics['Average_F1'] = {'mean': float(np.mean(avg_f1)) if avg_f1 else 0.0,
                             'std': float(np.std(avg_f1)) if avg_f1 else 0.0,
                             'median': float(np.median(avg_f1)) if avg_f1 else 0.0}
    metrics['Avg_Pred_Count'] = {'mean': float(np.mean(pred_counts)) if pred_counts else 0.0,
                                 'std': float(np.std(pred_counts)) if pred_counts else 0.0,
                                 'median': float(np.median(pred_counts)) if pred_counts else 0.0}

    # @k metrics
    for k in k_values:
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        f1_scores: List[float] = []

        for query, preds in predictions.items():
            if query not in ground_truth:
                continue

            gt_set = ground_truth[query]
            top_k_preds = [cand for cand, _ in preds[:k]]
            pred_set = set(top_k_preds)

            if len(pred_set) == 0:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                tp = len(pred_set.intersection(gt_set))
                precision = tp / len(pred_set)
                recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        if precision_scores:
            metrics[f'P@{k}'] = {'mean': float(np.mean(precision_scores)),
                                 'std': float(np.std(precision_scores)),
                                 'median': float(np.median(precision_scores))}
            metrics[f'R@{k}'] = {'mean': float(np.mean(recall_scores)),
                                 'std': float(np.std(recall_scores)),
                                 'median': float(np.median(recall_scores))}
            metrics[f'F1@{k}'] = {'mean': float(np.mean(f1_scores)),
                                  'std': float(np.std(f1_scores)),
                                  'median': float(np.median(f1_scores))}
        else:
            metrics[f'P@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'R@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'F1@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}

    # Metrics at ground truth size (adaptive k per query)
    precision_at_gt_size: List[float] = []
    recall_at_gt_size: List[float] = []
    f1_at_gt_size: List[float] = []
    gt_sizes: List[int] = []

    for query, preds in predictions.items():
        if query not in ground_truth:
            continue

        gt_set = ground_truth[query]
        gt_size = len(gt_set)
        gt_sizes.append(gt_size)

        top_k_preds = [cand for cand, _ in preds[:gt_size]]
        pred_set = set(top_k_preds)

        if len(pred_set) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            tp = len(pred_set.intersection(gt_set))
            precision = tp / len(pred_set)
            recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_at_gt_size.append(precision)
        recall_at_gt_size.append(recall)
        f1_at_gt_size.append(f1)

    if precision_at_gt_size:
        metrics['P@GT_Size'] = {'mean': float(np.mean(precision_at_gt_size)),
                                'std': float(np.std(precision_at_gt_size)),
                                'median': float(np.median(precision_at_gt_size))}
        metrics['R@GT_Size'] = {'mean': float(np.mean(recall_at_gt_size)),
                                'std': float(np.std(recall_at_gt_size)),
                                'median': float(np.median(recall_at_gt_size))}
        metrics['F1@GT_Size'] = {'mean': float(np.mean(f1_at_gt_size)),
                                 'std': float(np.std(f1_at_gt_size)),
                                 'median': float(np.median(f1_at_gt_size))}
        metrics['Avg_GT_Size'] = {'mean': float(np.mean(gt_sizes)),
                                  'std': float(np.std(gt_sizes)),
                                  'median': float(np.median(gt_sizes))}
    else:
        metrics['P@GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        metrics['R@GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        metrics['F1@GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        metrics['Avg_GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}

    return metrics

def load_deepjoin_results(deepjoin_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load DeepJoin results from CSV file."""
    df = pd.read_csv(deepjoin_path)
    df['query_table'] = df['query_table'].str.replace('datalake-', '', regex=False).str.replace('.csv', '', regex=False).str.lower()
    df['candidate_table'] = df['candidate_table'].str.replace('datalake-', '', regex=False).str.replace('.csv', '', regex=False).str.lower()
    
    out: Dict[str, List[Tuple[str, float]]] = {}
    for _, r in df.iterrows():
        q = f"{r['query_table']}.{r['query_col']}"
        c = f"{r['candidate_table']}.{r['candidate_col']}"
        out.setdefault(q, []).append((c, float(r['score'])))
    
    # Sort by score descending
    for k in out:
        out[k].sort(key=lambda x: x[1], reverse=True)
    
    return out

def print_metrics_summary(semantic_metrics: Dict[str, Dict[str, float]], 
                         deepjoin_metrics: Dict[str, Dict[str, float]], 
                         k_values: List[int] = [1, 5, 10, 20, 50],
                         additional_semantic_metrics: List[tuple] = None):
    """Print a quick summary of metric scores."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Overall metrics (using adaptive k - all predictions per query)
    semantic_avg_preds = semantic_metrics.get('Avg_Pred_Count', {}).get('mean', 0.0)
    deepjoin_avg_preds = deepjoin_metrics.get('Avg_Pred_Count', {}).get('mean', 0.0)
    print(f"\nOVERALL METRICS (using adaptive k - all predictions per query):")
    print(f"  Average predictions per query - SemSketch: {semantic_avg_preds:.1f}, DeepJoin: {deepjoin_avg_preds:.1f}")
    print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'SemSketch (sketch)':<20} {semantic_metrics['Average_Precision']['mean']:<12.3f} {semantic_metrics['Average_Recall']['mean']:<12.3f} {semantic_metrics['Average_F1']['mean']:<12.3f}")
    
    # Print additional semantic results if available
    if additional_semantic_metrics:
        for name, metrics in additional_semantic_metrics:
            print(f"{'SemSketch (w/ LLM)':<20} {metrics['Average_Precision']['mean']:<12.3f} {metrics['Average_Recall']['mean']:<12.3f} {metrics['Average_F1']['mean']:<12.3f}")
    
    print(f"{'DeepJoin':<20} {deepjoin_metrics['Average_Precision']['mean']:<12.3f} {deepjoin_metrics['Average_Recall']['mean']:<12.3f} {deepjoin_metrics['Average_F1']['mean']:<12.3f}")
    
    # Calculate percentage improvements vs DeepJoin
    precision_improvement = ((semantic_metrics['Average_Precision']['mean'] - deepjoin_metrics['Average_Precision']['mean']) / deepjoin_metrics['Average_Precision']['mean']) * 100
    recall_improvement = ((semantic_metrics['Average_Recall']['mean'] - deepjoin_metrics['Average_Recall']['mean']) / deepjoin_metrics['Average_Recall']['mean']) * 100
    f1_improvement = ((semantic_metrics['Average_F1']['mean'] - deepjoin_metrics['Average_F1']['mean']) / deepjoin_metrics['Average_F1']['mean']) * 100
    
    print(f"{'Improvement (sketch)':<20} {precision_improvement:+12.1f}% {recall_improvement:+12.1f}% {f1_improvement:+12.1f}%")
    
    # Calculate improvement for SemSketch with LLM vs DeepJoin if available
    if additional_semantic_metrics:
        llm_metrics = additional_semantic_metrics[0][1] if additional_semantic_metrics else None
        if llm_metrics:
            precision_llm_improvement = ((llm_metrics['Average_Precision']['mean'] - deepjoin_metrics['Average_Precision']['mean']) / deepjoin_metrics['Average_Precision']['mean']) * 100
            recall_llm_improvement = ((llm_metrics['Average_Recall']['mean'] - deepjoin_metrics['Average_Recall']['mean']) / deepjoin_metrics['Average_Recall']['mean']) * 100
            f1_llm_improvement = ((llm_metrics['Average_F1']['mean'] - deepjoin_metrics['Average_F1']['mean']) / deepjoin_metrics['Average_F1']['mean']) * 100
            print(f"{'Improvement (w/ LLM)':<20} {precision_llm_improvement:+12.1f}% {recall_llm_improvement:+12.1f}% {f1_llm_improvement:+12.1f}%")
    
    # @k metrics - Precision
    print(f"\nTOP-K METRICS - PRECISION:")
    if additional_semantic_metrics:
        # Include SemSketch w/ LLM column
        print(f"{'K':<4} {'SemSketch P@k':<15} {'SemSketch+LLM P@k':<18} {'DeepJoin P@k':<15} {'Imp (sketch)':<15} {'Imp (w/ LLM)':<15}")
        print(f"{'-'*4} {'-'*15} {'-'*18} {'-'*15} {'-'*15} {'-'*15}")
        llm_metrics = additional_semantic_metrics[0][1] if additional_semantic_metrics else None
        for k in k_values:
            p_improvement = ((semantic_metrics[f'P@{k}']['mean'] - deepjoin_metrics[f'P@{k}']['mean']) / deepjoin_metrics[f'P@{k}']['mean']) * 100
            if llm_metrics:
                llm_p_at_k = llm_metrics[f'P@{k}']['mean']
                p_llm_improvement = ((llm_p_at_k - deepjoin_metrics[f'P@{k}']['mean']) / deepjoin_metrics[f'P@{k}']['mean']) * 100
                print(f"{k:<4} {semantic_metrics[f'P@{k}']['mean']:<15.3f} {llm_p_at_k:<18.3f} {deepjoin_metrics[f'P@{k}']['mean']:<15.3f} {p_improvement:+15.1f}% {p_llm_improvement:+15.1f}%")
            else:
                print(f"{k:<4} {semantic_metrics[f'P@{k}']['mean']:<15.3f} {'N/A':<18} {deepjoin_metrics[f'P@{k}']['mean']:<15.3f} {p_improvement:+15.1f}% {'N/A':<15}")
    else:
        # Original format without LLM
        print(f"{'K':<4} {'SemSketch P@k':<15} {'DeepJoin P@k':<15} {'Improvement':<12}")
        print(f"{'-'*4} {'-'*15} {'-'*15} {'-'*12}")
        for k in k_values:
            p_improvement = ((semantic_metrics[f'P@{k}']['mean'] - deepjoin_metrics[f'P@{k}']['mean']) / deepjoin_metrics[f'P@{k}']['mean']) * 100
            print(f"{k:<4} {semantic_metrics[f'P@{k}']['mean']:<15.3f} {deepjoin_metrics[f'P@{k}']['mean']:<15.3f} {p_improvement:+12.1f}%")
    
    # @k metrics - Recall
    print(f"\nTOP-K METRICS - RECALL:")
    if additional_semantic_metrics:
        # Include SemSketch w/ LLM column
        print(f"{'K':<4} {'SemSketch R@k':<15} {'SemSketch+LLM R@k':<18} {'DeepJoin R@k':<15} {'Imp (sketch)':<15} {'Imp (w/ LLM)':<15}")
        print(f"{'-'*4} {'-'*15} {'-'*18} {'-'*15} {'-'*15} {'-'*15}")
        llm_metrics = additional_semantic_metrics[0][1] if additional_semantic_metrics else None
        for k in k_values:
            r_improvement = ((semantic_metrics[f'R@{k}']['mean'] - deepjoin_metrics[f'R@{k}']['mean']) / deepjoin_metrics[f'R@{k}']['mean']) * 100 if deepjoin_metrics[f'R@{k}']['mean'] > 0 else 0.0
            if llm_metrics:
                llm_r_at_k = llm_metrics[f'R@{k}']['mean']
                r_llm_improvement = ((llm_r_at_k - deepjoin_metrics[f'R@{k}']['mean']) / deepjoin_metrics[f'R@{k}']['mean']) * 100 if deepjoin_metrics[f'R@{k}']['mean'] > 0 else 0.0
                print(f"{k:<4} {semantic_metrics[f'R@{k}']['mean']:<15.3f} {llm_r_at_k:<18.3f} {deepjoin_metrics[f'R@{k}']['mean']:<15.3f} {r_improvement:+15.1f}% {r_llm_improvement:+15.1f}%")
            else:
                print(f"{k:<4} {semantic_metrics[f'R@{k}']['mean']:<15.3f} {'N/A':<18} {deepjoin_metrics[f'R@{k}']['mean']:<15.3f} {r_improvement:+15.1f}% {'N/A':<15}")
    else:
        # Original format without LLM
        print(f"{'K':<4} {'SemSketch R@k':<15} {'DeepJoin R@k':<15} {'Improvement':<12}")
        print(f"{'-'*4} {'-'*15} {'-'*15} {'-'*12}")
        for k in k_values:
            r_improvement = ((semantic_metrics[f'R@{k}']['mean'] - deepjoin_metrics[f'R@{k}']['mean']) / deepjoin_metrics[f'R@{k}']['mean']) * 100 if deepjoin_metrics[f'R@{k}']['mean'] > 0 else 0.0
            print(f"{k:<4} {semantic_metrics[f'R@{k}']['mean']:<15.3f} {deepjoin_metrics[f'R@{k}']['mean']:<15.3f} {r_improvement:+12.1f}%")
    
    # Metrics at ground truth size (adaptive k per query)
    print(f"\nMETRICS AT GROUND TRUTH SIZE (adaptive k per query):")
    avg_gt_size = semantic_metrics.get('Avg_GT_Size', {}).get('mean', 0.0)
    print(f"Average ground truth size: {avg_gt_size:.1f}")
    print(f"{'Method':<20} {'Precision@GT':<15} {'Recall@GT':<15} {'F1@GT':<15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'SemSketch (sketch)':<20} {semantic_metrics.get('P@GT_Size', {}).get('mean', 0.0):<15.3f} {semantic_metrics.get('R@GT_Size', {}).get('mean', 0.0):<15.3f} {semantic_metrics.get('F1@GT_Size', {}).get('mean', 0.0):<15.3f}")
    
    # Print additional semantic results if available
    if additional_semantic_metrics:
        for name, metrics in additional_semantic_metrics:
            print(f"{'SemSketch (w/ LLM)':<20} {metrics.get('P@GT_Size', {}).get('mean', 0.0):<15.3f} {metrics.get('R@GT_Size', {}).get('mean', 0.0):<15.3f} {metrics.get('F1@GT_Size', {}).get('mean', 0.0):<15.3f}")
    
    print(f"{'DeepJoin':<20} {deepjoin_metrics.get('P@GT_Size', {}).get('mean', 0.0):<15.3f} {deepjoin_metrics.get('R@GT_Size', {}).get('mean', 0.0):<15.3f} {deepjoin_metrics.get('F1@GT_Size', {}).get('mean', 0.0):<15.3f}")
    
    # Calculate percentage improvements vs DeepJoin
    if deepjoin_metrics.get('P@GT_Size', {}).get('mean', 0.0) > 0:
        p_gt_improvement = ((semantic_metrics.get('P@GT_Size', {}).get('mean', 0.0) - deepjoin_metrics.get('P@GT_Size', {}).get('mean', 0.0)) / deepjoin_metrics.get('P@GT_Size', {}).get('mean', 0.0)) * 100
        r_gt_improvement = ((semantic_metrics.get('R@GT_Size', {}).get('mean', 0.0) - deepjoin_metrics.get('R@GT_Size', {}).get('mean', 0.0)) / deepjoin_metrics.get('R@GT_Size', {}).get('mean', 0.0)) * 100
        f1_gt_improvement = ((semantic_metrics.get('F1@GT_Size', {}).get('mean', 0.0) - deepjoin_metrics.get('F1@GT_Size', {}).get('mean', 0.0)) / deepjoin_metrics.get('F1@GT_Size', {}).get('mean', 0.0)) * 100
        print(f"{'Improvement (sketch)':<20} {p_gt_improvement:+15.1f}% {r_gt_improvement:+15.1f}% {f1_gt_improvement:+15.1f}%")
        
        # Calculate improvement for SemSketch with LLM vs DeepJoin if available
        if additional_semantic_metrics:
            llm_metrics = additional_semantic_metrics[0][1] if additional_semantic_metrics else None
            if llm_metrics:
                p_gt_llm_improvement = ((llm_metrics.get('P@GT_Size', {}).get('mean', 0.0) - deepjoin_metrics.get('P@GT_Size', {}).get('mean', 0.0)) / deepjoin_metrics.get('P@GT_Size', {}).get('mean', 0.0)) * 100
                r_gt_llm_improvement = ((llm_metrics.get('R@GT_Size', {}).get('mean', 0.0) - deepjoin_metrics.get('R@GT_Size', {}).get('mean', 0.0)) / deepjoin_metrics.get('R@GT_Size', {}).get('mean', 0.0)) * 100
                f1_gt_llm_improvement = ((llm_metrics.get('F1@GT_Size', {}).get('mean', 0.0) - deepjoin_metrics.get('F1@GT_Size', {}).get('mean', 0.0)) / deepjoin_metrics.get('F1@GT_Size', {}).get('mean', 0.0)) * 100
                print(f"{'Improvement (w/ LLM)':<20} {p_gt_llm_improvement:+15.1f}% {r_gt_llm_improvement:+15.1f}% {f1_gt_llm_improvement:+15.1f}%")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic join results against DeepJoin baseline')
    
    parser.add_argument('--semantic-results', required=True,
                       help='Path to semantic join results CSV file')
    parser.add_argument('--deepjoin-results', required=True,
                       help='Path to DeepJoin results CSV file')
    parser.add_argument('--ground-truth', required=True,
                       help='Path to ground truth CSV file (Freyja format)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--k-values', type=int, nargs='*', default=[1, 5, 10, 20, 50],
                       help='K values for Precision@k, Recall@k, F1@k calculations')
    parser.add_argument('--additional-semantic-results', type=str, nargs='*',
                       help='Additional semantic result files to compare in the same table')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ground_truth = load_ground_truth(args.ground_truth)
    deepjoin_preds = load_deepjoin_results(args.deepjoin_results)
    semantic_preds, _ = load_semantic_results(args.semantic_results)
    
    # Remove self-joins
    deepjoin_preds = remove_self_joins(deepjoin_preds)
    semantic_preds = remove_self_joins(semantic_preds)
    
    print(f"Loaded {len(ground_truth)} ground truth queries")
    print(f"Loaded {len(deepjoin_preds)} DeepJoin predictions")
    print(f"Loaded {len(semantic_preds)} semantic predictions")
    
    # Diagnostic: Check average predictions per query
    deepjoin_avg_preds = np.mean([len(preds) for preds in deepjoin_preds.values()]) if deepjoin_preds else 0
    semantic_avg_preds = np.mean([len(preds) for preds in semantic_preds.values()]) if semantic_preds else 0
    max_k = max(args.k_values) if args.k_values else 50
    
    print(f"\nDiagnostic Info:")
    print(f"  Average predictions per query - DeepJoin: {deepjoin_avg_preds:.1f}, SemSketch: {semantic_avg_preds:.1f}")
    print(f"  Using top-{max_k} predictions for Average_Precision calculation")
    
    # Check how many queries have fewer than max_k predictions
    deepjoin_below_k = sum(1 for preds in deepjoin_preds.values() if len(preds) < max_k)
    semantic_below_k = sum(1 for preds in semantic_preds.values() if len(preds) < max_k)
    print(f"  Queries with <{max_k} predictions - DeepJoin: {deepjoin_below_k}/{len(deepjoin_preds)}, SemSketch: {semantic_below_k}/{len(semantic_preds)}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    semantic_metrics = calculate_metrics(semantic_preds, ground_truth, args.k_values)
    deepjoin_metrics = calculate_metrics(deepjoin_preds, ground_truth, args.k_values)
    
    # Load additional semantic results if provided
    additional_semantic_metrics_list = []
    if args.additional_semantic_results:
        for idx, add_result_path in enumerate(args.additional_semantic_results):
            if Path(add_result_path).exists():
                print(f"Loading additional semantic results: {add_result_path}")
                add_semantic_preds, _ = load_semantic_results(add_result_path)
                add_semantic_preds = remove_self_joins(add_semantic_preds)
                add_semantic_metrics = calculate_metrics(add_semantic_preds, ground_truth, args.k_values)
                
                # Generate a name for this result (e.g., "llm_pruned")
                result_name = Path(add_result_path).parent.name if 'pruned' in add_result_path else f"semantic_{idx+1}"
                additional_semantic_metrics_list.append((result_name, add_semantic_metrics))
    
    # Print quick metrics summary
    print_metrics_summary(semantic_metrics, deepjoin_metrics, args.k_values, additional_semantic_metrics_list)
    
    # Save results (so --output-dir is meaningful)
    summary = {
        "semantic_results": args.semantic_results,
        "deepjoin_results": args.deepjoin_results,
        "ground_truth": args.ground_truth,
        "k_values": args.k_values,
        "semantic_metrics": semantic_metrics,
        "deepjoin_metrics": deepjoin_metrics,
        "additional_semantic_metrics": [
            {"name": name, "metrics": metrics} for name, metrics in additional_semantic_metrics_list
        ],
    }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation summary to {output_dir / 'evaluation_summary.json'}")
    
    
    return 0

if __name__ == "__main__":
    exit(main())
