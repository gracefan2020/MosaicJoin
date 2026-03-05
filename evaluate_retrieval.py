#!/usr/bin/env python3
"""
Semantic Join Retrieval Evaluation Script

Modes:
  1. Single/comparison mode: evaluate results against ground truth
  2. LLM annotation mode: evaluate WDC candidates with LLM-verified matches

Usage:
    # Single method evaluation
    python evaluate_retrieval.py --results results.csv --ground-truth gt.csv

    # Combined experiments (CSV output for Google Sheets)
    python evaluate_retrieval.py --combined --experiments autofj freyja wt

    # LLM annotation evaluation (WDC benchmarks)
    python evaluate_retrieval.py --llm-annotation --benchmarks autofj-wdc freyja-wdc wt-wdc
"""

import json
import math
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_name(name) -> str:
    """Normalize table/column name by removing .csv suffix and lowercasing."""
    if name is None:
        return ""
    name = str(name).strip()
    if name.lower().endswith('.csv'):
        name = name[:-4]
    return name.lower()


# =============================================================================
# Data Loading
# =============================================================================

def detect_level(gt_file: str) -> str:
    """Auto-detect column vs table level from ground truth."""
    df = pd.read_csv(gt_file, nrows=1)
    cols = set(df.columns)
    if {'source_column', 'target_column'}.issubset(cols) or {'target_attr', 'candidate_attr'}.issubset(cols):
        return 'column'
    return 'table'


def load_ground_truth(gt_file: str, level: str) -> Dict:
    """Load ground truth for column or table level."""
    df = pd.read_csv(gt_file)
    ground_truth = defaultdict(set)
    
    if level == 'column':
        if 'source_table' in df.columns:
            for _, row in df.iterrows():
                src = (normalize_name(row['source_table']), normalize_name(row['source_column']))
                tgt = (normalize_name(row['target_table']), normalize_name(row['target_column']))
                ground_truth[src].add(tgt)
        elif 'target_ds' in df.columns:
            for _, row in df.iterrows():
                src = (normalize_name(row['target_ds']), normalize_name(row['target_attr']))
                tgt = (normalize_name(row['candidate_ds']), normalize_name(row['candidate_attr']))
                ground_truth[src].add(tgt)
    else:
        if 'left_table' in df.columns:
            for _, row in df.iterrows():
                ground_truth[normalize_name(row['left_table'])].add(normalize_name(row['right_table']))
        elif 'source_table' in df.columns:
            for _, row in df.iterrows():
                ground_truth[normalize_name(row['source_table'])].add(normalize_name(row['target_table']))
    
    return dict(ground_truth)


def load_results(results_file: str, level: str) -> Dict:
    """Load results for column or table level."""
    df = pd.read_csv(results_file)
    
    if level == 'column':
        results = defaultdict(list)
        for _, row in df.iterrows():
            query = (normalize_name(row['query_table']), normalize_name(row['query_column']))
            candidate = (normalize_name(row['candidate_table']), normalize_name(row['candidate_column']))
            score = float(row['similarity_score'])
            if query != candidate:
                results[query].append((candidate, score))
        for q in results:
            results[q] = sorted(results[q], key=lambda x: -x[1])
        return dict(results)
    else:
        results_dict = defaultdict(dict)
        for _, row in df.iterrows():
            query = normalize_name(row['query_table'])
            candidate = normalize_name(row['candidate_table'])
            score = float(row['similarity_score'])
            if query != candidate:
                if candidate not in results_dict[query] or score > results_dict[query][candidate]:
                    results_dict[query][candidate] = score
        return {q: sorted([(c, s) for c, s in cands.items()], key=lambda x: -x[1]) 
                for q, cands in results_dict.items()}


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_dcg(relevance: List[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]) if rel > 0)


def compute_ndcg(relevance: List[int], k: int, num_relevant: int) -> float:
    dcg = compute_dcg(relevance, k)
    idcg = compute_dcg([1] * min(k, num_relevant), k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_metrics(gt: Dict, results: Dict, k_values: List[int], level: str) -> Tuple[Dict, int, Dict]:
    """Evaluate HITS@K, Precision@K, Recall@K, NDCG@K, MRR using MACRO-AVERAGING."""
    metrics = {m: {k: 0.0 for k in k_values} for m in ['hits', 'precision', 'recall', 'ndcg', 'mrr']}
    gt_metrics = {'hits_at_gt': 0.0, 'precision_at_gt': 0.0, 'recall_at_gt': 0.0, 'ndcg_at_gt': 0.0, 'mrr': 0.0}
    per_query = {}
    total = 0
    
    for query, expected in gt.items():
        if query not in results:
            continue
        total += 1
        gt_size = len(expected)
        
        # Build relevance list
        seen, relevance = set(), []
        for item in results[query]:
            c = item[0] if level == 'table' else (item[0][0], item[0][1]) if isinstance(item[0], tuple) else item[0]
            if level == 'column':
                c = (item[0], item[1]) if len(item) == 3 else item[0]
            relevance.append(1 if c in expected and c not in seen else 0)
            if c in expected:
                seen.add(c)
        
        first_rel = next((i + 1 for i, r in enumerate(relevance) if r > 0), None)
        rr = 1.0 / first_rel if first_rel else 0.0
        
        # Metrics at |GT| size
        correct_at_gt = sum(relevance[:gt_size])
        gt_metrics['hits_at_gt'] += 1.0 if correct_at_gt > 0 else 0.0
        gt_metrics['precision_at_gt'] += correct_at_gt / gt_size
        gt_metrics['recall_at_gt'] += correct_at_gt / gt_size
        gt_metrics['ndcg_at_gt'] += compute_ndcg(relevance, gt_size, gt_size)
        gt_metrics['mrr'] += rr
        
        per_query[query] = {'gt_size': gt_size, 'reciprocal_rank': rr}
        
        for k in k_values:
            correct = sum(relevance[:k])
            metrics['hits'][k] += 1.0 if correct > 0 else 0.0
            metrics['precision'][k] += correct / k
            metrics['recall'][k] += correct / gt_size
            metrics['ndcg'][k] += compute_ndcg(relevance, k, gt_size)
            if first_rel and first_rel <= k:
                metrics['mrr'][k] += rr
    
    # Average
    for m in metrics:
        for k in k_values:
            metrics[m][k] /= total if total > 0 else 1
    for m in gt_metrics:
        if m != 'per_query':
            gt_metrics[m] /= total if total > 0 else 1
    gt_metrics['hits_at_gt_count'] = int(gt_metrics['hits_at_gt'] * total)
    gt_metrics['per_query'] = per_query
    
    return metrics, total, gt_metrics


# =============================================================================
# Output Functions
# =============================================================================

def print_metrics(metrics: Dict, total: int, name: str, k_values: List[int]):
    print(f"\n{'=' * 90}")
    print(f"{name} (n={total} queries, macro-averaged)")
    print(f"{'=' * 90}")
    print(f"{'K':>4} {'HITS@K':>14} {'Precision@K':>14} {'Recall@K':>12} {'NDCG@K':>12} {'MRR@K':>10}")
    print("-" * 80)
    for k in k_values:
        h, p, r, n, m = metrics['hits'][k], metrics['precision'][k], metrics['recall'][k], metrics['ndcg'][k], metrics['mrr'][k]
        print(f"{k:>4} {h:>8.3f} ({int(h*total):>3}) {p:>12.3f} {r:>12.3f} {n:>12.3f} {m:>10.3f}")


def print_gt_metrics(gt_metrics: Dict, total: int):
    per_query = gt_metrics['per_query']
    gt_sizes = [pq['gt_size'] for pq in per_query.values()]
    print(f"\n{'=' * 60}")
    print(f"METRICS AT GROUND TRUTH SIZE (K = |GT| per query)")
    print(f"{'=' * 60}")
    print(f"Average |GT| size: {sum(gt_sizes)/len(gt_sizes):.2f} (min: {min(gt_sizes)}, max: {max(gt_sizes)})")
    print(f"  HITS@|GT|:      {gt_metrics['hits_at_gt']:>8.3f} ({gt_metrics['hits_at_gt_count']:>3}/{total})")
    print(f"  Precision@|GT|: {gt_metrics['precision_at_gt']:>8.3f}")
    print(f"  Recall@|GT|:    {gt_metrics['recall_at_gt']:>8.3f}")
    print(f"  NDCG@|GT|:      {gt_metrics['ndcg_at_gt']:>8.3f}")
    print(f"  MRR (full):     {gt_metrics['mrr']:>8.3f}")


# =============================================================================
# Combined Experiments Configuration
# =============================================================================

EXPERIMENTS = {
    'autofj': {
        'level': 'table',
        'base': {
            'gt': 'datasets/autofj/groundtruth-joinable.csv',
            'Q128': 'autofj-experiments/autofj_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0': 'autofj-experiments/autofj_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv',
            'Q0_inverse': 'autofj-experiments/autofj_query_results_embeddinggemma_D128_Q0_inverse_chamfer_top50_slurm/all_query_results.csv',
            'Q0_symmetric': 'autofj-experiments/autofj_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'autofj-experiments/autofj_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'autofj-experiments/autofj_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_k_closest': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_k_closest_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_random': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_random_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'D0': 'autofj-experiments/autofj_query_results_embeddinggemma_D0_Q0_farthest_point_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'D64_bucket_1': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_1_top50_slurm/all_query_results.csv',
            'D64_bucket_2': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_2_top50_slurm/all_query_results.csv',
            'D64_bucket_3': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_3_top50_slurm/all_query_results.csv',
            'D64_bucket_4': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_4_top50_slurm/all_query_results.csv',
            'D64_bucket_5': 'autofj-experiments/autofj_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_5_top50_slurm/all_query_results.csv',
            'D0_bucket_1': 'autofj-experiments/autofj_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_1_top50_slurm/all_query_results.csv',
            'D0_bucket_2': 'autofj-experiments/autofj_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_2_top50_slurm/all_query_results.csv',
            'D0_bucket_3': 'autofj-experiments/autofj_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_3_top50_slurm/all_query_results.csv',
            'D0_bucket_4': 'autofj-experiments/autofj_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_4_top50_slurm/all_query_results.csv',
            'D0_bucket_5': 'autofj-experiments/autofj_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_5_top50_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_autofj.csv',
            'pexeso': 'baselines/pexeso/pexeso_fasttext_autofj.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_autofj.csv',
            'tabsketchfm': 'baselines/tabsketchfm/tabsketchfm_ft_autofj.csv',
        },
        'wdc': {
            'gt': 'datasets/autofj-wdc/groundtruth-joinable.csv',
            'Q128': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv',
            'Q0_inverse': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q0_inverse_chamfer_top50_slurm/all_query_results.csv',
            'Q0_symmetric': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_autofj-wdc.csv',
            # 'pexeso': 'baselines/pexeso/pexeso_autofj-wdc.csv',
            # 'deepjoin': 'autofj-wdc-experiments/deepjoin_autofj-wdc.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_autofj-wdc.csv',
            'tabsketchfm': 'baselines/tabsketchfm/tabsketchfm_ft_autofj-wdc.csv',
        },
    },
    'freyja': {
        'level': 'column',
        'base': {
            'gt': 'datasets/freyja/groundtruth-joinable.csv',
            'Q128': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0_chamfer': 'freyja-experiments/freyja_query_results_embeddinggemma128_D64_Q0_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q0_inverse': 'freyja-experiments/freyja_query_results_embeddinggemma128_D64_Q0_inverse_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q0_symmetric': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'freyja-experiments/freyja_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_k_closest': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_k_closest_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_random': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_random_slurm/all_query_results.csv',
            'Q0_kmeans': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_kmeans_slurm/all_query_results.csv',
            'D0': 'freyja-experiments/freyja_query_results_embeddinggemma_D0_Q0_farthest_point_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'D64_bucket_1': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_1_top50_slurm/all_query_results.csv',
            'D64_bucket_2': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_2_top50_slurm/all_query_results.csv',
            'D64_bucket_3': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_3_top50_slurm/all_query_results.csv',
            'D64_bucket_4': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_4_top50_slurm/all_query_results.csv',
            'D64_bucket_5': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_5_top50_slurm/all_query_results.csv',
            'D0_bucket_1': 'freyja-experiments/freyja_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_1_top50_slurm/all_query_results.csv',
            'D0_bucket_2': 'freyja-experiments/freyja_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_2_top50_slurm/all_query_results.csv',
            'D0_bucket_3': 'freyja-experiments/freyja_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_3_top50_slurm/all_query_results.csv',
            'D0_bucket_4': 'freyja-experiments/freyja_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_4_top50_slurm/all_query_results.csv',
            'D0_bucket_5': 'freyja-experiments/freyja_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_5_top50_slurm/all_query_results.csv',
            'Q16': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q16_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q32': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q32_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q64': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q64_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q128': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q128_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q256': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q256_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q512': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q512_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q1024': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q1024_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q2048': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q2048_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q4096': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q4096_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Q8192': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q8192_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'QAll': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm_2/all_query_results.csv',
            'Gemma256': 'freyja-experiments/freyja_query_results_embeddinggemma256_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Gemma512': 'freyja-experiments/freyja_query_results_embeddinggemma512_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'Gemma768': 'freyja-experiments/freyja_query_results_embeddinggemma768_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'MPNet768': 'freyja-experiments/freyja_query_results_mpnet768_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'BGe768': 'freyja-experiments/freyja_query_results_bge768_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'BGe384': 'freyja-experiments/freyja_query_results_bge384_D64_Q0_symmetric_chamfer_top50_farthest_point_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_freyja.csv',
            'pexeso': 'baselines/pexeso/pexeso_fasttext_freyja.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_freyja.csv',
            'tabsketchfm': 'baselines/tabsketchfm/tabsketchfm_ft_freyja.csv',
        },
        'wdc': {
            'gt': 'datasets/freyja-wdc/groundtruth-joinable.csv',
            'Q128': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv',
            'Q0_inverse': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D128_Q0_inverse_chamfer_top50_slurm/all_query_results.csv',
            'Q0_symmetric': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'freyja-wdc-experiments/freyja-wdc_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_freyja-wdc.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_freyja-wdc.csv',
            'tabsketchfm': 'baselines/tabsketchfm/tabsketchfm_ft_freyja-wdc.csv',
        },
    },
    'wt': {
        'level': 'column',
        'base': {
            'gt': 'datasets/wt/groundtruth-joinable.csv',
            'Q128': 'wt-experiments/wt_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0': 'wt-experiments/wt_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv',
            'Q0_inverse': 'wt-experiments/wt_query_results_embeddinggemma_D128_Q0_inverse_chamfer_top50_slurm/all_query_results.csv',
            'Q0_symmetric': 'wt-experiments/wt_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'wt-experiments/wt_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'wt-experiments/wt_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_k_closest': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_k_closest_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_random': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_random_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'D0': 'wt-experiments/wt_query_results_embeddinggemma_D0_Q0_farthest_point_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'D64_bucket_1': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_1_top50_slurm/all_query_results.csv',
            'D64_bucket_2': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_2_top50_slurm/all_query_results.csv',
            'D64_bucket_3': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_3_top50_slurm/all_query_results.csv',
            'D64_bucket_4': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_4_top50_slurm/all_query_results.csv',
            'D64_bucket_5': 'wt-experiments/wt_query_results_embeddinggemma_D64_Q0_farthest_point_queryBucket_5_top50_slurm/all_query_results.csv',
            'D0_bucket_1': 'wt-experiments/wt_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_1_top50_slurm/all_query_results.csv',
            'D0_bucket_2': 'wt-experiments/wt_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_2_top50_slurm/all_query_results.csv',
            'D0_bucket_3': 'wt-experiments/wt_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_3_top50_slurm/all_query_results.csv',
            'D0_bucket_4': 'wt-experiments/wt_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_4_top50_slurm/all_query_results.csv',
            'D0_bucket_5': 'wt-experiments/wt_query_results_embeddinggemma_D0_Q0_farthest_point_queryBucket_5_top50_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_wt.csv',
            'pexeso': 'baselines/pexeso/pexeso_fasttext_wt.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_wt.csv',
            'tabsketchfm': 'baselines/tabsketchfm/tabsketchfm_ft_wt.csv',
        },
        'wdc': {
            'gt': 'datasets/wt-wdc/groundtruth-joinable.csv',
            'Q128': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv',
            'Q0_inverse': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D128_Q0_inverse_chamfer_top50_slurm/all_query_results.csv',
            'Q0_symmetric': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'wt-wdc-experiments/wt-wdc_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_wt-wdc.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_wt-wdc.csv',
            'tabsketchfm': 'baselines/tabsketchfm/tabsketchfm_ft_wt-wdc.csv',
        },
    },
    'gdc': {
        'level': 'column',
        'base': {
            'gt': 'datasets/gdc/groundtruth-joinable.csv',
            'Q128': 'gdc-experiments/gdc_query_results_embeddinggemma_D64_Q64_chamfer_top50_slurm/all_query_results.csv',
            'Q0': None,
            'deepjoin': 'gdc-experiments/deepjoin_ft_gdc.csv',
            'pexeso': 'gdc-experiments/pexeso-gdc-full-ranked.csv',
        },
        'wdc': None,
    },
}


def load_bucket_queries(dataset_dir: Path, bucket: int) -> set:
    """Load query tables for a specific bucket."""
    bucket_file = dataset_dir / f"query_columns_bucket_{bucket}.csv"
    if not bucket_file.exists():
        return set()
    try:
        df = pd.read_csv(bucket_file)
        return set(df['target_ds'].str.strip().tolist())
    except Exception:
        return set()


def filter_results_by_bucket(results: Dict, bucket_queries: set, level: str) -> Dict:
    """Filter results to only include queries from the bucket."""
    # Normalize bucket queries: lowercase, no .csv extension
    bucket_queries_norm = {q.replace('.csv', '').lower() for q in bucket_queries}
    
    filtered = {}
    for query_key, candidates in results.items():
        if level == 'table':
            query_table = query_key if isinstance(query_key, str) else query_key[0]
        else:
            query_table = query_key[0] if isinstance(query_key, tuple) else query_key
        # Normalize query table name (lowercase, remove .csv if present)
        query_table_norm = query_table.replace('.csv', '').lower()
        if query_table_norm in bucket_queries_norm:
            filtered[query_key] = candidates
    return filtered


def filter_gt_by_bucket(gt: Dict, bucket_queries: set, level: str) -> Dict:
    """Filter ground truth to only include queries from the bucket."""
    # Normalize bucket queries: lowercase, no .csv extension
    bucket_queries_norm = {q.replace('.csv', '').lower() for q in bucket_queries}
    
    filtered = {}
    for query_key, expected in gt.items():
        if level == 'table':
            query_table = query_key if isinstance(query_key, str) else query_key[0]
        else:
            query_table = query_key[0] if isinstance(query_key, tuple) else query_key
        # Normalize query table name (lowercase, remove .csv if present)
        query_table_norm = query_table.replace('.csv', '').lower()
        if query_table_norm in bucket_queries_norm:
            filtered[query_key] = expected
    return filtered


def load_data(config: dict, base_path: Path, level: str, dataset_name: str = None) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict]]:
    """Load ground truth and all result variants for a config.
    
    Returns:
        gt: Full ground truth
        results: Dict of variant -> results
        bucket_gts: Dict of bucket_num -> filtered ground truth (for bucket evaluation)
    """
    gt_path = base_path / config['gt']
    if not gt_path.exists():
        print(f"    ⚠️  GT not found: {gt_path}")
        return {}, {}, {}
    
    gt = load_ground_truth(str(gt_path), level)
    results = {}
    bucket_gts = {}
    
    for variant in ['Q128', 'Q16', 'Q32', 'Q64', 'Q128', 'Q256', 'Q512', 'Q1024', 'Q2048', 'Q4096', 'Q8192', 'QAll', 'Q0', 'Q0_chamfer', 'Q0_inverse', 'Q0_symmetric', 'Q0_harmonic', 'Q0_D64_symmetric', 'Q0_D64', 'Q0_D32', 'Q0_k_closest', 'Q0_random', 'Q0_kmeans', 'D0', 'D64_bucket_1', 'D64_bucket_2', 'D64_bucket_3', 'D64_bucket_4', 'D64_bucket_5', 'D0_bucket_1', 'D0_bucket_2', 'D0_bucket_3', 'D0_bucket_4', 'D0_bucket_5', 'deepjoin', 'pexeso', 'snoopy', 'tabsketchfm', 'Gemma256', 'Gemma512', 'Gemma768', 'MPNet768', 'BGe768', 'BGe384']:
        if config.get(variant) is None:
            continue
        path = base_path / config[variant]
        if path.exists():
            results[variant] = load_results(str(path), level)
        else:
            print(f"    ⚠️  {variant} not found: {path}")
    
    # Generate bucket-specific baseline results by filtering full results
    if dataset_name:
        dataset_dir = base_path / 'datasets' / dataset_name
        for bucket in [1, 2, 3, 4, 5]:
            bucket_queries = load_bucket_queries(dataset_dir, bucket)
            if bucket_queries:
                # Filter ground truth for this bucket
                bucket_gts[bucket] = filter_gt_by_bucket(gt, bucket_queries, level)
                # Filter baseline results for this bucket
                for baseline in ['deepjoin', 'pexeso', 'snoopy', 'tabsketchfm']:
                    if baseline not in results:
                        continue
                    filtered = filter_results_by_bucket(results[baseline], bucket_queries, level)
                    results[f'{baseline}_bucket_{bucket}'] = filtered
    
    return gt, results, bucket_gts


def print_combined_table(name: str, base_metrics: Dict, combined_metrics: Dict, k_values: List[int], metrics_list: List[str], ablation: bool = False):
    """Print CSV table for combined experiments."""
    # ablation_study = 'd_sketch_size'
    # ablation_study = 'similarity_method'
    # ablation_study = 'selection_method'
    ablation_study = 'query_bucket'
    # ablation_study = 'query_sample_size'
    # ablation_study = 'embedding_model'
    for metric in metrics_list:
        mkey = metric.lower()  # evaluate_metrics uses lowercase keys
        print(f"\n{'='*100}")
        print(f"{metric}@K - {name}")
        print(f"{'='*100}")

        if ablation:
            if ablation_study == 'similarity_method':
                # cols = [f"SemSketch ({name} - chamfer)", f"SemSketch ({name} - inverse chamfer)", f"SemSketch ({name} - average chamfer)", f"SemSketch ({name} - harmonic mean chamfer)",
                        # f"SemSketch ({name}+WDC - chamfer)", f"SemSketch ({name}+WDC - inverse chamfer)", f"SemSketch ({name}+WDC - average chamfer)", f"SemSketch ({name}+WDC - harmonic mean chamfer)"]
                cols = [f"SemSketch ({name} - chamfer)", f"SemSketch ({name} - inverse chamfer)", f"SemSketch ({name} - average chamfer)", f"SemSketch ({name} - harmonic mean chamfer)"]

                print("K," + ",".join(cols))
                for k in k_values:
                    row = [str(k)]
                    # Base Q0
                    val = base_metrics.get('Q0_chamfer', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_inverse
                    val = base_metrics.get('Q0_inverse', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_symmetric
                    val = base_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # # Base Q0_harmonic
                    # val = base_metrics.get('Q0_harmonic', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    # # Combined Q0
                    # val = combined_metrics.get('Q0', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    # # Combined Q0_inverse
                    # val = combined_metrics.get('Q0_inverse', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    # # Combined Q0_symmetric
                    # val = combined_metrics.get('Q0_symmetric', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    # # Combined Q0_harmonic
                    # val = combined_metrics.get('Q0_harmonic', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    print(",".join(row))
            elif ablation_study == 'embedding_model':
                cols = [f"SemSketch (Gemma128)", f"SemSketch (Gemma256)", f"SemSketch (Gemma512)", f"SemSketch (Gemma768)", f"SemSketch (MPNet768)", f"SemSketch (BGe768)", f"SemSketch (BGe384)"]
                print("K," + ",".join(cols))
                for k in k_values:
                    row = [str(k)]
                    val = base_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('Gemma256', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('Gemma512', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('Gemma768', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('MPNet768', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('BGe768', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('BGe384', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    print(",".join(row))
            elif ablation_study == 'query_sample_size':
                cols = [f"SemSketch"]
                print("Sample Size," + ",".join(cols))
                k = 10
                print(f"Evaluating at K={k}")
                for sample_size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                    row = [str(sample_size)]
                    val = base_metrics.get(f'Q{sample_size}', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    print(",".join(row))
                row = ["All"]
                val = base_metrics.get(f'QAll', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                print(",".join(row))
            elif ablation_study == 'd_sketch_size':
                cols = [f"SemSketch ({name} - |D|=128)", f"SemSketch ({name} - |D|=64)", f"SemSketch ({name} - |D|=32)", f"SemSketch ({name} - |D|=0)",
                        f"SemSketch ({name}+WDC - |D|=128)", f"SemSketch ({name}+WDC - |D|=64)", f"SemSketch ({name}+WDC - |D|=32)"]
                print("K," + ",".join(cols))
                for k in k_values:
                    row = [str(k)]
                    # Base Q0
                    val = base_metrics.get('Q0_symmetric', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_D64
                    val = base_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_D32
                    val = base_metrics.get('Q0_D32', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_D0
                    val = base_metrics.get('D0', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0
                    val = combined_metrics.get('Q0_symmetric', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0_D64
                    val = combined_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0_D32
                    val = combined_metrics.get('Q0_D32', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    print(",".join(row))
            elif ablation_study == 'selection_method':
                # cols = [f"SemSketch ({name} - k-center)", f"SemSketch ({name} - origin)", f"SemSketch ({name} - random)"]
                cols = [f"SemSketch ({name} - first-k)"]

                print("K," + ",".join(cols))
                for k in k_values:
                    # row = [str(k)]
                    # # Base Q0_D64
                    # val = base_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    # # Base Q0_k_closest
                    # val = base_metrics.get('Q0_k_closest', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    # # Base Q0_random
                    # val = base_metrics.get('Q0_random', {}).get(mkey, {}).get(k)
                    # row.append(f"{val:.3f}" if val is not None else "")
                    val = base_metrics.get('Q0_kmeans', {}).get(mkey, {}).get(k)
                    row = [f"{val:.3f}" if val is not None else ""]
                    print(",".join(row))
            elif ablation_study == 'query_bucket':
                # Print table with buckets as rows, methods as columns
                # Use a specific K value for comparison (e.g., K=10 or first available)
                target_k = 10 if 10 in k_values else k_values[0]
                print(f"\n(Using K={target_k})")
                cols = ["Bucket", "SemSketch", "SemSketch (no sketching)", "DeepJoin", "Snoopy", "PEXESO", "TabSketchFM"]
                print(",".join(cols))
                for bucket in [1, 2, 3, 4, 5]:
                    row = [str(bucket)]
                    # SemSketch with sketching (D64_bucket_X)
                    val = base_metrics.get(f'D64_bucket_{bucket}', {}).get(mkey, {}).get(target_k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # SemSketch no sketching (D0_bucket_X)
                    val = base_metrics.get(f'D0_bucket_{bucket}', {}).get(mkey, {}).get(target_k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # DeepJoin (bucket-specific if available, otherwise empty)
                    val = base_metrics.get(f'deepjoin_bucket_{bucket}', {}).get(mkey, {}).get(target_k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Snoopy (bucket-specific if available, otherwise empty)
                    val = base_metrics.get(f'snoopy_bucket_{bucket}', {}).get(mkey, {}).get(target_k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # PEXESO (bucket-specific if available, otherwise empty)
                    val = base_metrics.get(f'pexeso_bucket_{bucket}', {}).get(mkey, {}).get(target_k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # TabSketchFM (bucket-specific if available, otherwise empty)
                    val = base_metrics.get(f'tabsketchfm_bucket_{bucket}', {}).get(mkey, {}).get(target_k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    print(",".join(row))
        else:
            cols = [f"SemSketch ({name})",f"SemSketch ({name} - no sketching)", f"DeepJoin ({name})", f"Pexeso ({name})", f"Snoopy ({name})", f"TabSketchFM ({name})"]
            print("K," + ",".join(cols))
            for k in k_values:
                row = [str(k)]
                # Base Q0_D64
                val = base_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Base Q0_D0
                val = base_metrics.get('D0', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # DeepJoin (base)
                val = base_metrics.get('deepjoin', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Pexeso (base)
                val = base_metrics.get('pexeso', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Snoopy (base)
                val = base_metrics.get('snoopy', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # TabSketchFM (base)
                val = base_metrics.get('tabsketchfm', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                print(",".join(row))


# =============================================================================
# LLM Annotation Evaluation (using full result files)
# =============================================================================

LLM_ANNOTATION_CONFIG = {
    'autofj-wdc': {
        'level': 'table',
        # 'annotation_file': 'scripts/wdc_top_k_annotation/llm-annotations/autofj-wdc_joinability_annotations.csv',
        # 'annotation_file': 'scripts/wdc_top_k_annotation_llm/autofj-wdc_annotation_llm.csv',
        'annotation_file': 'scripts/wdc_top_k_annotation_llm_og/autofj-wdc_annotation_llm_merged.csv',

        'gt': 'datasets/autofj-wdc/groundtruth-joinable.csv',
        'results': {
            'semsketch': 'scripts/wdc_top_k/semsketch_D64_avg_chamfer_autofj-wdc.csv',
            'semsketch_D0': 'scripts/wdc_top_k/semsketch_D0_autofj-wdc.csv',
            'deepjoin': 'scripts/wdc_top_k/deepjoin_ft_autofj-wdc.csv',
            'snoopy': 'scripts/wdc_top_k/snoopy_ft_autofj-wdc.csv',
        },
    },
    'freyja-wdc': {
        'level': 'column',
        # 'annotation_file': 'scripts/wdc_top_k_annotation/llm-annotations/freyja-wdc_joinability_annotations.csv',
        # 'annotation_file': 'scripts/wdc_top_k_annotation_llm/freyja-wdc_annotation_llm.csv',
        'annotation_file': 'scripts/wdc_top_k_annotation_llm_og/freyja-wdc_annotation_llm_merged.csv',


        'gt': 'datasets/freyja-wdc/groundtruth-joinable.csv',
        'results': {
            'semsketch': 'scripts/wdc_top_k/semsketch_D64_avg_chamfer_freyja-wdc.csv',
            'semsketch_D0': 'scripts/wdc_top_k/semsketch_D0_freyja-wdc.csv',
            'deepjoin': 'scripts/wdc_top_k/deepjoin_ft_freyja-wdc.csv',
            'snoopy': 'scripts/wdc_top_k/snoopy_ft_freyja-wdc.csv',
        },
    },
    'wt-wdc': {
        'level': 'column',
        # 'annotation_file': 'scripts/wdc_top_k_annotation/llm-annotations/wt-wdc_joinability_annotations.csv',
        # 'annotation_file': 'scripts/wdc_top_k_annotation_llm/wt-wdc_annotation_llm.csv',
        'annotation_file': 'scripts/wdc_top_k_annotation_llm_og/wt-wdc_annotation_llm_merged.csv',

        'gt': 'datasets/wt-wdc/groundtruth-joinable.csv',
        'results': {
            'semsketch': 'scripts/wdc_top_k/semsketch_D64_avg_chamfer_wt-wdc.csv',
            'semsketch_D0': 'scripts/wdc_top_k/semsketch_D0_wt-wdc.csv',
            'deepjoin': 'scripts/wdc_top_k/deepjoin_ft_wt-wdc.csv',
            'snoopy': 'scripts/wdc_top_k/snoopy_ft_wt-wdc.csv',
        },
    },
}


def parse_matched_pairs(matched_pairs_str: str) -> List:
    """Parse the matched_pairs JSON column."""
    if pd.isna(matched_pairs_str) or not matched_pairs_str:
        return []
    try:
        parsed = json.loads(matched_pairs_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def load_annotation_lookup(annotation_file: str, level: str) -> Dict[Tuple, Optional[bool]]:
    """
    Load LLM annotation CSV as a lookup table.
    
    Supports two formats:
    1. New format with 'joinability' column: match if "equijoin" or "semantic"
    2. Old format with 'confidence' column: match if confidence > 0.5
    
    For column-level benchmarks, if annotation file doesn't have query_column/candidate_column,
    we store table-level keys and the evaluate function will handle the lookup.
    
    Returns: {key: has_llm_match} where has_llm_match is True/False
    """
    df = pd.read_csv(annotation_file)
    lookup = {}
    
    # Detect format
    has_joinability = 'joinability' in df.columns
    has_confidence = 'confidence' in df.columns
    has_column_info = 'query_column' in df.columns and 'candidate_column' in df.columns
    
    for _, row in df.iterrows():
        query_table = normalize_name(row['query_table'])
        candidate_table = normalize_name(row['candidate_table'])
        
        # Determine match based on format
        if has_joinability:
            joinability = str(row.get('joinability', '')).strip().lower()
            has_llm_match = joinability in ('equijoin', 'semantic')
        elif has_confidence:
            confidence = float(row.get('confidence', 0))
            has_llm_match = confidence > 0.7
        else:
            # No match column found, skip
            continue
        
        # Build key based on available columns
        if level == 'table' or not has_column_info:
            # Table-level key: (query_table, candidate_table)
            key = (query_table, candidate_table)
        else:
            # Column-level key: ((query_table, query_column), (candidate_table, candidate_column))
            query_column = normalize_name(row['query_column'])
            candidate_column = normalize_name(row['candidate_column'])
            key = ((query_table, query_column), (candidate_table, candidate_column))
        
        lookup[key] = has_llm_match
    
    # Store metadata about lookup type
    lookup['_has_column_info'] = has_column_info
    
    return lookup


def evaluate_with_llm_annotations(
    gt: Dict, 
    results: Dict,
    annotation_lookup: Dict[Tuple, Optional[bool]],
    k_values: List[int],
    level: str
) -> Tuple[Dict, int]:
    """
    Evaluate results using GT + LLM annotation lookup.

    A candidate is a MATCH if:
    - It's in the ground truth, OR
    - It's in annotation_lookup with has_llm_match=True, OR
    - It's NOT in annotation_lookup (wasn't evaluated by LLM, passed lexical threshold)

    A candidate is NOT a match only if:
    - It's in annotation_lookup with has_llm_match=False (LLM said no match)

    Returns: (metrics_dict, total_queries)
    """
    metrics = {m: {k: 0.0 for k in k_values} for m in ['hits', 'precision', 'ndcg', 'mrr']}
    total = 0
    
    # Check if annotation lookup has column info or is table-level only
    has_column_info = annotation_lookup.get('_has_column_info', False)

    # Only evaluate queries that exist in ground truth (same as GT-only evaluation)
    for query in gt.keys():
        if query not in results:
            continue
        candidates = results[query]
        gt_set = gt[query]

        # Build relevance list
        relevance = []
        for item in candidates:
            candidate = item[0]
            query_table = query[0] if isinstance(query, tuple) else query
            candidate_table = candidate[0] if isinstance(candidate, tuple) else candidate
            lookup_key = (query_table, candidate_table)

            # Check if it's a match
            in_gt = candidate in gt_set
            if in_gt:
                is_match = True
            elif not lookup_key[1].startswith('target_') and not in_gt:
                # From original benchmark (not from WDC) but it is not expected table
                is_match = False
            elif lookup_key in annotation_lookup:
                # In annotation file -> use LLM result (True = match, False = not joinable)
                is_match = annotation_lookup[lookup_key]
            else:
                # Not in annotation lookup - candidate passed lexical threshold
                is_match = False
            relevance.append(1 if is_match else 0)

        if not relevance:
            continue

        total += 1
        # For NDCG, use total number of matches (GT + LLM verified) as num_relevant
        num_relevant = sum(relevance)
        if num_relevant == 0:
            num_relevant = len(gt_set)  # Fallback to GT size if no matches found
        
        # Compute reciprocal rank (position of first relevant item)
        first_rel = next((i + 1 for i, r in enumerate(relevance) if r > 0), None)
        rr = 1.0 / first_rel if first_rel else 0.0

        for k in k_values:
            correct = sum(relevance[:k])
            metrics['hits'][k] += 1.0 if correct > 0 else 0.0
            metrics['precision'][k] += correct / k if k > 0 else 0.0
            metrics['ndcg'][k] += compute_ndcg(relevance, k, num_relevant)
            if first_rel and first_rel <= k:
                metrics['mrr'][k] += rr

    # Average
    if total > 0:
        for m in metrics:
            for k in k_values:
                metrics[m][k] /= total

    return metrics, total


def print_llm_annotation_metrics(
    benchmark: str,
    all_metrics: Dict[str, Dict],
    k_values: List[int]
):
    """Print LLM annotation evaluation results."""
    print(f"\n{'=' * 90}")
    print(f"LLM Annotation Evaluation: {benchmark}")
    print(f"{'=' * 90}")
    
    methods = ['semsketch', 'semsketch_D0', 'deepjoin', 'snoopy']
    header = "K," + ",".join(methods)
    # # Print Hits@K table
    # print(f"\nHits@K:")
    # print(header)
    # for k in k_values:
    #     row = [str(k)]
    #     for method in methods:
    #         val = all_metrics[method]['hits'].get(k, 0.0)
    #         row.append(f"{val:.3f}")
    #     print(",".join(row))
    
    # Print Precision@K table
    print(f"\nPrecision@K:")
    print(header)
    for k in k_values:
        row = [str(k)]
        for method in methods:
            val = all_metrics[method]['precision'].get(k, 0.0)
            row.append(f"{val:.3f}")
        print(",".join(row))
    
    # Print NDCG@K table
    print(f"\nNDCG@K:")
    print(header)
    for k in k_values:
        row = [str(k)]
        for method in methods:
            val = all_metrics[method]['ndcg'].get(k, 0.0)
            row.append(f"{val:.3f}")
        print(",".join(row))
    
    # # Print MRR@K table
    # print(f"\nMRR@K:")
    # print(header)
    # for k in k_values:
    #     row = [str(k)]
    #     for method in methods:
    #         val = all_metrics[method]['mrr'].get(k, 0.0)
    #         row.append(f"{val:.3f}")
    #     print(",".join(row))
    
    # Print query counts
    print(f"\nQuery counts per method:")
    for method in methods:
        print(f"  {method}: {all_metrics[method]['total']} queries")


def run_llm_annotation_evaluation(args):
    """Run LLM annotation evaluation for WDC benchmarks using full result files."""
    base_path = Path(args.base_path)
    k_values = [k for k in args.k_values if k <= 10]  # Up to top-10
    
    benchmarks = args.benchmarks if args.benchmarks else list(LLM_ANNOTATION_CONFIG.keys())
    all_results = {}
    
    for benchmark in benchmarks:

        config = LLM_ANNOTATION_CONFIG.get(benchmark)
        if not config:
            print(f"Unknown benchmark: {benchmark}")
            continue
        
        level = config['level']
        annotation_path = base_path / config['annotation_file']
        gt_path = base_path / config['gt']
        result_files = config.get('results', {})
        
        if not annotation_path.exists():
            print(f"⚠️  Annotation file not found: {annotation_path}")
            exit(1)
        if not gt_path.exists():
            print(f"⚠️  Ground truth not found: {gt_path}")
            exit(1)
        
        # Load ground truth
        gt = load_ground_truth(str(gt_path), level)
        
        # Load annotation lookup
        annotation_lookup = load_annotation_lookup(str(annotation_path), level)
        
        # Evaluate each method's results
        all_metrics = {}
        
        for method, result_file in result_files.items():
            result_path = base_path / result_file
            if not result_path.exists():
                print(f"  ⚠️  {method} results not found: {result_path}")
                continue
            
            # Load full results for this method
            results = load_results(str(result_path), level)
            
            # Evaluate with LLM annotations
            metrics, total = evaluate_with_llm_annotations(
                gt, results, annotation_lookup, k_values, level
            )
            metrics['total'] = total
            all_metrics[method] = metrics
        
        # Print results
        if all_metrics:
            print_llm_annotation_metrics(benchmark, all_metrics, k_values)
        
        # Store for saving
        all_results[benchmark] = {
            'level': level,
            'k_values': k_values,
            'metrics': all_metrics,
        }
    
    # Save results to JSON if requested
    if args.save_results:
        save_dir = Path(args.save_results)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = save_dir / 'llm_annotation_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Saved LLM annotation results to {output_file}")
    
    return 0


def run_single(args):
    """Run single/comparison evaluation."""
    if not Path(args.ground_truth).exists():
        print(f"❌ Ground truth not found: {args.ground_truth}")
        return 1
    if not Path(args.results).exists():
        print(f"❌ Results not found: {args.results}")
        return 1
    
    level = args.level if args.level != 'auto' else detect_level(args.ground_truth)
    print(f"Evaluation level: {level.upper()}")
    
    gt = load_ground_truth(args.ground_truth, level)
    print(f"\nGround truth: {len(gt)} queries")
    
    results = load_results(args.results, level)
    print(f"{args.name} results: {len(results)} queries")
    metrics, total, gt_metrics = evaluate_metrics(gt, results, args.k_values, level)
    
    print_metrics(metrics, total, args.name, args.k_values)
    print_gt_metrics(gt_metrics, total)
    
    if args.baseline and Path(args.baseline).exists():
        baseline_results = load_results(args.baseline, level)
        print(f"\n{args.baseline_name} results: {len(baseline_results)} queries")
        bl_metrics, bl_total, bl_gt_metrics = evaluate_metrics(gt, baseline_results, args.k_values, level)
        print_metrics(bl_metrics, bl_total, args.baseline_name, args.k_values)
        print_gt_metrics(bl_gt_metrics, bl_total)
    
    print("\n" + "=" * 70)
    return 0


def run_combined_experiments(args):
    """Run combined experiments evaluation (CSV output)."""
    base_path = Path(args.base_path)
    all_results = {}
    
    for exp_name in args.experiments:
        exp = EXPERIMENTS.get(exp_name)
        if not exp:
            print(f"Unknown experiment: {exp_name}")
            continue
        
        level = exp['level']
        print(f"\n{'#' * 80}")
        print(f"# {exp_name.upper()} (level: {level})")
        print(f"{'#' * 80}")
        
        # Load base data
        base_config = exp.get('base', {})
        base_gt, base_results, base_bucket_gts = load_data(base_config, base_path, level, dataset_name=exp_name)
        
        # Load WDC data if available
        wdc_config = exp.get('wdc')
        if wdc_config:
            wdc_gt, wdc_results, wdc_bucket_gts = load_data(wdc_config, base_path, level, dataset_name=f"{exp_name}-wdc")
        else:
            wdc_gt, wdc_results, wdc_bucket_gts = {}, {}, {}
        
        # Compute metrics for base
        base_metrics = {}
        for variant, results in base_results.items():
            # Use bucket-specific GT for bucket variants
            if '_bucket_' in variant:
                bucket_num = int(variant.split('_bucket_')[1])
                gt_to_use = base_bucket_gts.get(bucket_num, base_gt)
            else:
                gt_to_use = base_gt
            metrics, total, _ = evaluate_metrics(gt_to_use, results, args.k_values, level)
            base_metrics[variant] = metrics
            base_metrics[f'{variant}_total'] = total
        
        # Compute metrics for WDC (combined)
        combined_metrics = {}
        for variant, results in wdc_results.items():
            # Use bucket-specific GT for bucket variants
            if '_bucket_' in variant:
                bucket_num = int(variant.split('_bucket_')[1])
                gt_to_use = wdc_bucket_gts.get(bucket_num, wdc_gt)
            else:
                gt_to_use = wdc_gt
            metrics, total, _ = evaluate_metrics(gt_to_use, results, args.k_values, level)
            combined_metrics[variant] = metrics
            combined_metrics[f'{variant}_total'] = total
        
        # Print combined table
        print_combined_table(exp_name, base_metrics, combined_metrics, args.k_values, args.metrics, ablation=args.ablation)
        
        # Store for saving
        all_results[exp_name] = {
            'level': level,
            'k_values': args.k_values,
            'base_metrics': base_metrics,
            'combined_metrics': combined_metrics,
        }
    
    # Save results to JSON if requested
    if args.save_results:
        save_dir = Path(args.save_results)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = save_dir / f'combined_experiments_{args.experiments}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Saved combined experiment results to {output_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Semantic Join Retrieval Evaluation')
    
    # Mode selection
    parser.add_argument('--combined', action='store_true', help='Run combined experiments evaluation')
    parser.add_argument('--llm-annotation', action='store_true', 
                       help='Run LLM annotation evaluation for WDC benchmarks')
    
    # Combined mode args
    parser.add_argument('--experiments', nargs='+', choices=['autofj', 'freyja', 'wt', 'gdc'],
                       default=['autofj', 'freyja', 'wt'])
    parser.add_argument('--metrics', nargs='+', choices=['HITS', 'Precision', 'Recall', 'NDCG', 'MRR'],
                       default=['HITS', 'Precision', 'Recall', 'NDCG', 'MRR'])
    parser.add_argument('--base-path', type=str, default='.')
    parser.add_argument('--ablation', action='store_true', help='Run ablation studies')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Directory to save results as JSON files for plotting')
    
    # LLM annotation mode args
    parser.add_argument('--benchmarks', nargs='+', 
                       choices=['autofj-wdc', 'freyja-wdc', 'wt-wdc'],
                       default=None,
                       help='WDC benchmarks to evaluate (default: all)')
    
    # Single mode args
    parser.add_argument('--results', type=str, help='Results CSV file')
    parser.add_argument('--ground-truth', type=str, help='Ground truth CSV file')
    parser.add_argument('--level', choices=['table', 'column', 'auto'], default='auto')
    parser.add_argument('--name', type=str, default='Method')
    parser.add_argument('--baseline', type=str, help='Optional baseline results CSV')
    parser.add_argument('--baseline-name', type=str, default='Baseline')
    
    # Common args
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 3, 5, 10, 20, 30, 40, 50])
    
    args = parser.parse_args()
    
    if args.llm_annotation:
        return run_llm_annotation_evaluation(args)
    elif args.combined:
        return run_combined_experiments(args)
    else:
        if not args.results or not args.ground_truth:
            parser.error("--results and --ground-truth required for single evaluation mode")
        return run_single(args)


if __name__ == "__main__":
    exit(main() or 0)
