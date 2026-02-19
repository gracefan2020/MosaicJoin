#!/usr/bin/env python3
"""
Semantic Join Retrieval Evaluation Script

Modes:
  1. Single/comparison mode: evaluate results against ground truth

Usage:
    # Single method evaluation
    python evaluate_retrieval.py --results results.csv --ground-truth gt.csv

    # Combined experiments (CSV output for Google Sheets)
    python evaluate_retrieval.py --combined --experiments autofj freyja wt
"""

import math
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


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
                src = (str(row['source_table']).replace('.csv', '').lower(), str(row['source_column']).lower())
                tgt = (str(row['target_table']).replace('.csv', '').lower(), str(row['target_column']).lower())
                ground_truth[src].add(tgt)
        elif 'target_ds' in df.columns:
            for _, row in df.iterrows():
                src = (str(row['target_ds']).replace('.csv', '').lower(), str(row['target_attr']).lower())
                tgt = (str(row['candidate_ds']).replace('.csv', '').lower(), str(row['candidate_attr']).lower())
                ground_truth[src].add(tgt)
    else:
        if 'left_table' in df.columns:
            for _, row in df.iterrows():
                ground_truth[str(row['left_table']).replace('.csv', '').lower()].add(
                    str(row['right_table']).replace('.csv', '').lower())
        elif 'source_table' in df.columns:
            for _, row in df.iterrows():
                ground_truth[str(row['source_table']).replace('.csv', '').lower()].add(
                    str(row['target_table']).replace('.csv', '').lower())
    
    return dict(ground_truth)


def load_results(results_file: str, level: str) -> Dict:
    """Load results for column or table level."""
    df = pd.read_csv(results_file)
    
    if level == 'column':
        results = defaultdict(list)
        for _, row in df.iterrows():
            query = (str(row['query_table']).replace('.csv', '').lower(), str(row['query_column']).lower())
            candidate = (str(row['candidate_table']).replace('.csv', '').lower(), str(row['candidate_column']).lower())
            score = float(row['similarity_score'])
            if query != candidate:
                results[query].append((candidate, score))
        for q in results:
            results[q] = sorted(results[q], key=lambda x: -x[1])
        return dict(results)
    else:
        results_dict = defaultdict(dict)
        for _, row in df.iterrows():
            query = str(row['query_table']).replace('.csv', '').lower()
            candidate = str(row['candidate_table']).replace('.csv', '').lower()
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
            # 'deepjoin': 'autofj-experiments/deepjoin_ft_autofj_grace.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_autofj.csv',
            'pexeso': 'baselines/pexeso/pexeso_fasttext_autofj.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_autofj.csv',
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
        },
    },
    'freyja': {
        'level': 'column',
        'base': {
            'gt': 'datasets/freyja/groundtruth-joinable.csv',
            'Q128': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv',
            'Q0': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv',
            'Q0_inverse': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_inverse_chamfer_top50_slurm/all_query_results.csv',
            'Q0_symmetric': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_harmonic': 'freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_harmonic_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D64': 'freyja-experiments/freyja_query_results_embeddinggemma_D64_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'Q0_D32': 'freyja-experiments/freyja_query_results_embeddinggemma_D32_Q0_symmetric_chamfer_top50_slurm/all_query_results.csv',
            'deepjoin': 'baselines/deepjoin/deepjoin_freyja.csv',
            'pexeso': 'baselines/pexeso/pexeso_fasttext_freyja.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_freyja.csv',
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
            'deepjoin': 'baselines/deepjoin/deepjoin_wt.csv',
            'pexeso': 'baselines/pexeso/pexeso_fasttext_wt.csv',
            'snoopy': 'baselines/snoopy/snoopy_ft_wt.csv',
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


def load_data(config: dict, base_path: Path, level: str) -> Tuple[Dict, Dict[str, Dict]]:
    """Load ground truth and all result variants for a config."""
    gt_path = base_path / config['gt']
    if not gt_path.exists():
        print(f"    ⚠️  GT not found: {gt_path}")
        return {}, {}
    
    gt = load_ground_truth(str(gt_path), level)
    results = {}
    
    for variant in ['Q128', 'Q0', 'Q0_inverse', 'Q0_symmetric', 'Q0_harmonic', 'Q0_D64_symmetric', 'Q0_D64', 'Q0_D32', 'deepjoin', 'pexeso', 'snoopy']:
        if config.get(variant) is None:
            continue
        path = base_path / config[variant]
        if path.exists():
            results[variant] = load_results(str(path), level)
        else:
            print(f"    ⚠️  {variant} not found: {path}")
    
    return gt, results


def print_combined_table(name: str, base_metrics: Dict, combined_metrics: Dict, k_values: List[int], metrics_list: List[str], ablation: bool = False):
    """Print CSV table for combined experiments."""
    ablation_study = 'd_sketch_size'
    # ablation_study = 'similarity_method'
    for metric in metrics_list:
        mkey = metric.lower()
        print(f"\n{'='*100}")
        print(f"{metric}@K - {name}")
        print(f"{'='*100}")

        if ablation:
            if ablation_study == 'similarity_method':
                cols = [f"SemSketch ({name} - chamfer)", f"SemSketch ({name} - inverse chamfer)", f"SemSketch ({name} - average chamfer)", f"SemSketch ({name} - harmonic mean chamfer)",
                        f"SemSketch ({name}+WDC - chamfer)", f"SemSketch ({name}+WDC - inverse chamfer)", f"SemSketch ({name}+WDC - average chamfer)", f"SemSketch ({name}+WDC - harmonic mean chamfer)"]
                print("K," + ",".join(cols))
                for k in k_values:
                    row = [str(k)]
                    # Base Q0
                    val = base_metrics.get('Q0', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_inverse
                    val = base_metrics.get('Q0_inverse', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_symmetric
                    val = base_metrics.get('Q0_symmetric', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Base Q0_harmonic
                    val = base_metrics.get('Q0_harmonic', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0
                    val = combined_metrics.get('Q0', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0_inverse
                    val = combined_metrics.get('Q0_inverse', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0_symmetric
                    val = combined_metrics.get('Q0_symmetric', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    # Combined Q0_harmonic
                    val = combined_metrics.get('Q0_harmonic', {}).get(mkey, {}).get(k)
                    row.append(f"{val:.3f}" if val is not None else "")
                    print(",".join(row))
            if ablation_study == 'd_sketch_size':
                cols = [f"SemSketch ({name} - |D|=128)", f"SemSketch ({name} - |D|=64)", f"SemSketch ({name} - |D|=32)"
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
        else:
            cols = [f"SemSketch ({name})", f"SemSketch ({name}+WDC)", f"DeepJoin ({name})"
                    , f"DeepJoin ({name} + WDC)", f"Pexeso ({name})", f"Pexeso ({name} + WDC)", f"Snoopy ({name})", f"Snoopy ({name} + WDC)"]
            print("K," + ",".join(cols))
            for k in k_values:
                row = [str(k)]
                # Base Q0_D64
                val = base_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Combined Q0_D64
                val = combined_metrics.get('Q0_D64', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # DeepJoin (base)
                val = base_metrics.get('deepjoin', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # DeepJoin (combined)
                val = combined_metrics.get('deepjoin', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Pexeso (base)
                val = base_metrics.get('pexeso', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Pexeso (combined)
                val = combined_metrics.get('pexeso', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Snoopy (base)
                val = base_metrics.get('snoopy', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                # Snoopy (combined)
                val = combined_metrics.get('snoopy', {}).get(mkey, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "")
                print(",".join(row))


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
        base_gt, base_results = load_data(base_config, base_path, level)
        
        # Load WDC data if available
        wdc_config = exp.get('wdc')
        if wdc_config:
            wdc_gt, wdc_results = load_data(wdc_config, base_path, level)
        else:
            wdc_gt, wdc_results = {}, {}
        
        # Compute metrics for base
        base_metrics = {}
        for variant, results in base_results.items():
            metrics, total, _ = evaluate_metrics(base_gt, results, args.k_values, level)
            base_metrics[variant] = metrics
        
        # Compute metrics for WDC (combined)
        combined_metrics = {}
        for variant, results in wdc_results.items():
            metrics, total, _ = evaluate_metrics(wdc_gt, results, args.k_values, level)
            combined_metrics[variant] = metrics
        
        # Print combined table
        print_combined_table(exp_name, base_metrics, combined_metrics, args.k_values, args.metrics, ablation=args.ablation)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='Semantic Join Retrieval Evaluation')
    
    # Mode selection
    parser.add_argument('--combined', action='store_true', help='Run combined experiments evaluation')
    
    # Combined mode args
    parser.add_argument('--experiments', nargs='+', choices=['autofj', 'freyja', 'wt', 'gdc'],
                       default=['autofj', 'freyja', 'wt'])
    parser.add_argument('--metrics', nargs='+', choices=['HITS', 'Precision', 'Recall', 'NDCG', 'MRR'],
                       default=['HITS', 'Precision', 'Recall', 'NDCG', 'MRR'])
    parser.add_argument('--base-path', type=str, default='.')
    parser.add_argument('--ablation', action='store_true', help='Run ablation studies')
    
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
    
    if args.combined:
        return run_combined_experiments(args)
    else:
        if not args.results or not args.ground_truth:
            parser.error("--results and --ground-truth required for single evaluation mode")
        return run_single(args)


if __name__ == "__main__":
    exit(main() or 0)
