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
    """Load ground truth from CSV file."""
    gt_df = pd.read_csv(gt_path)
    gt_df['target_ds'] = gt_df['target_ds'].str.replace('.csv', '', regex=False).str.lower()
    gt_df['candidate_ds'] = gt_df['candidate_ds'].str.replace('.csv', '', regex=False).str.lower()
    
    by_query: Dict[str, Set[str]] = {}
    for _, r in gt_df.iterrows():
        q = f"{r['target_ds']}.{r['target_attr']}"
        c = f"{r['candidate_ds']}.{r['candidate_attr']}"
        by_query.setdefault(q, set()).add(c)
    
    return by_query

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

def load_semantic_results(semantic_path: str, similarity_threshold: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
    """Load semantic join results from CSV file, filtering by similarity threshold."""
    df = pd.read_csv(semantic_path)
    
    # Filter by similarity threshold
    df_filtered = df[df['similarity_score'] >= similarity_threshold]
    print(f"Filtered semantic results: {len(df)} -> {len(df_filtered)} (threshold: {similarity_threshold})")
    
    out: Dict[str, List[Tuple[str, float]]] = {}
    for _, r in df_filtered.iterrows():
        # Remove .csv extension from table names to match ground truth format
        query_table = str(r['query_table']).replace('.csv', '').lower()
        candidate_table = str(r['candidate_table']).replace('.csv', '').lower()
        
        q = f"{query_table}.{r['query_column']}"
        c = f"{candidate_table}.{r['candidate_column']}"
        out.setdefault(q, []).append((c, float(r['similarity_score'])))
    
    # Sort by similarity score descending
    for k in out:
        out[k].sort(key=lambda x: x[1], reverse=True)
    
    return out

def remove_self_joins(preds: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    """Remove self-joins from predictions."""
    cleaned: Dict[str, List[Tuple[str, float]]] = {}
    for q, items in preds.items():
        cleaned[q] = [(c, s) for c, s in items if c != q]
    return cleaned

def calculate_metrics(predictions: Dict[str, List[Tuple[str, float]]], 
                     ground_truth: Dict[str, Set[str]], 
                     k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Dict[str, float]]:
    """Calculate Precision@k, Recall@k, and F1@k metrics."""
    metrics = {}
    
    # Calculate total precision, recall, F1 (not @k)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for query, preds in predictions.items():
        if query not in ground_truth:
            continue
        
        gt_set = ground_truth[query]
        pred_set = set([cand for cand, _ in preds])
        
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate total metrics
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
    
    metrics['Total_Precision'] = {'mean': total_precision, 'std': 0.0, 'median': total_precision}
    metrics['Total_Recall'] = {'mean': total_recall, 'std': 0.0, 'median': total_recall}
    metrics['Total_F1'] = {'mean': total_f1, 'std': 0.0, 'median': total_f1}
    
    # Calculate @k metrics
    for k in k_values:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for query, preds in predictions.items():
            if query not in ground_truth:
                continue
            
            gt_set = ground_truth[query]
            top_k_preds = [cand for cand, _ in preds[:k]]
            pred_set = set(top_k_preds)
            
            # Calculate metrics
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
        
        # Only calculate if we have valid scores
        if precision_scores:
            metrics[f'P@{k}'] = {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores),
                'median': np.median(precision_scores)
            }
            metrics[f'R@{k}'] = {
                'mean': np.mean(recall_scores),
                'std': np.std(recall_scores),
                'median': np.median(recall_scores)
            }
            metrics[f'F1@{k}'] = {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'median': np.median(f1_scores)
            }
        else:
            metrics[f'P@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'R@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'F1@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
    
    return metrics

def load_column_samples(table: str, column: str, datalake_root: str, k: int) -> List[str]:
    """Load sample values from a column in CSV or Parquet format."""
    # Try different case variations of the table name
    def transform_table_name(table_name: str) -> str:
        """Transform table name to match actual filenames."""
        parts = table_name.split('_')
        transformed_parts = []
        for part in parts:
            if part.startswith('adventureworks'):
                # Special case for AdventureWorks
                transformed = part.replace('adventureworks', 'AdventureWorks')
            else:
                # Special handling for compound words
                if 'country' in part.lower() and 'region' in part.lower():
                    transformed = 'CountryRegion'
                elif 'state' in part.lower() and 'province' in part.lower():
                    transformed = 'stateprovince'  # This one is lowercase in the actual file
                elif 'sales' in part.lower() and 'territory' in part.lower():
                    transformed = 'SalesTerritory'
                elif 'credit' in part.lower() and 'card' in part.lower():
                    transformed = 'CreditCard'
                elif 'currency' in part.lower() and 'rate' in part.lower():
                    transformed = 'CurrencyRate'
                elif 'country' in part.lower() and 'region' in part.lower() and 'currency' in part.lower():
                    transformed = 'CountryRegionCurrency'
                else:
                    # Title case for other parts
                    transformed = part.title()
            transformed_parts.append(transformed)
        return '_'.join(transformed_parts)
    
    table_variations = [
        table,  # original case
        table.title(),  # Title Case
        table.upper(),  # UPPER CASE
        table.lower(),  # lower case
        table.replace('_', '').title(),  # Remove underscores and title case
        table.replace('_', ' ').title().replace(' ', ''),  # Remove spaces and underscores
        transform_table_name(table),  # Smart transformation
        # Try lowercasing the smart transformation
        transform_table_name(table).lower(),
        # Try lowercasing original table name
        table.lower(),
    ]
    
    values: List[str] = []
    for table_var in table_variations:
        csv_path = Path(datalake_root) / f"{table_var}.csv"
        pq_path = Path(datalake_root) / f"{table_var}.parquet"
        
        try:
            if csv_path.exists():
                ser = pd.read_csv(csv_path, usecols=[column], dtype=str)[column]
            elif pq_path.exists():
                ser = pd.read_parquet(pq_path, columns=[column]).astype(str)[column]
            else:
                continue
                
            # Dropna, strip, take unique order-preserving
            seen = set()
            for v in ser.dropna().astype(str):
                sv = v.strip()
                if not sv:
                    continue
                if sv in seen:
                    continue
                seen.add(sv)
                values.append(sv)
                if len(values) >= k:
                    break
            break  # Found the file, stop trying other variations
        except Exception:
            continue
    
    return values

def get_value_set(table: str, column: str, datalake_root: str) -> Set[str]:
    """Get normalized unique values from a column for overlap computation."""
    # Try different case variations of the table name
    def transform_table_name(table_name: str) -> str:
        """Transform table name to match actual filenames."""
        parts = table_name.split('_')
        transformed_parts = []
        for part in parts:
            if part.startswith('adventureworks'):
                # Special case for AdventureWorks
                transformed = part.replace('adventureworks', 'AdventureWorks')
            else:
                # Special handling for compound words
                if 'country' in part.lower() and 'region' in part.lower():
                    transformed = 'CountryRegion'
                elif 'state' in part.lower() and 'province' in part.lower():
                    transformed = 'stateprovince'  # This one is lowercase in the actual file
                elif 'sales' in part.lower() and 'territory' in part.lower():
                    transformed = 'SalesTerritory'
                elif 'credit' in part.lower() and 'card' in part.lower():
                    transformed = 'CreditCard'
                elif 'currency' in part.lower() and 'rate' in part.lower():
                    transformed = 'CurrencyRate'
                elif 'country' in part.lower() and 'region' in part.lower() and 'currency' in part.lower():
                    transformed = 'CountryRegionCurrency'
                else:
                    # Title case for other parts
                    transformed = part.title()
            transformed_parts.append(transformed)
        return '_'.join(transformed_parts)
    
    table_variations = [
        table,  # original case
        table.title(),  # Title Case
        table.upper(),  # UPPER CASE
        table.lower(),  # lower case
        table.replace('_', '').title(),  # Remove underscores and title case
        table.replace('_', ' ').title().replace(' ', ''),  # Remove spaces and underscores
        transform_table_name(table),  # Smart transformation
        # Try lowercasing the smart transformation
        transform_table_name(table).lower(),
        # Try lowercasing original table name
        table.lower(),
    ]
    
    vals: Set[str] = set()
    for table_var in table_variations:
        csv_path = Path(datalake_root) / f"{table_var}.csv"
        pq_path = Path(datalake_root) / f"{table_var}.parquet"
        
        try:
            if csv_path.exists():
                ser = pd.read_csv(csv_path, usecols=[column], dtype=str)[column]
            elif pq_path.exists():
                ser = pd.read_parquet(pq_path, columns=[column]).astype(str)[column]
            else:
                continue
                
            # Normalize similar to value_overlap: remove non-word chars, trim, lowercase
            ser = ser.dropna().astype(str)
            ser = ser.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
            vals = set(ser[ser != ''])
            break  # Found the file, stop trying other variations
        except Exception:
            continue
    
    return vals

def split_key(key: str) -> Tuple[str, str]:
    """Split table.column key into table and column parts."""
    if '.' in key:
        t, c = key.split('.', 1)
        return t, c
    return key, ''

def print_sample_info(q_table: str, q_col: str, c_table: str, c_col: str, 
                      q_samples: List[str], c_samples: List[str], 
                      overlap: float, intersection_fraction: float, category: str):
    """Print sample values and metrics to console for inspection."""
    print(f"\n=== {category.upper()} ===")
    print(f"Query: {q_table}.{q_col}")
    print(f"Candidate: {c_table}.{c_col}")
    print(f"Query samples: {q_samples}")
    print(f"Candidate samples: {c_samples}")
    print(f"Jaccard similarity: {overlap}")
    print(f"Set Containment: {intersection_fraction}")
    print("-" * 50)

def analyze_disagreements(semantic_preds: Dict[str, List[Tuple[str, float]]],
                        deepjoin_preds: Dict[str, List[Tuple[str, float]]],
                        ground_truth: Dict[str, Set[str]], 
                        datalake_dir: str = 'datasets/freyja-semantic-join/datalake',
                        sample_count: int = 5,
                        print_samples: bool = False) -> Dict[str, List[Dict]]:
    """Analyze disagreements between semantic and DeepJoin predictions."""
    
    disagreements = {
        'semantic_right_deepjoin_wrong': [],
        'deepjoin_right_semantic_wrong': [],
        'semantic_false_positives': [],
        'deepjoin_false_positives': []
    }
    
    for query in ground_truth:
        if query not in semantic_preds or query not in deepjoin_preds:
            continue
        
        gt_set = ground_truth[query]
        semantic_set = set([cand for cand, _ in semantic_preds[query]])
        deepjoin_set = set([cand for cand, _ in deepjoin_preds[query]])
        
        # Semantic right, DeepJoin wrong
        semantic_correct = semantic_set.intersection(gt_set)
        deepjoin_correct = deepjoin_set.intersection(gt_set)
        
        semantic_only_correct = semantic_correct - deepjoin_correct
        for cand in semantic_only_correct:
            q_table, q_col = split_key(query)
            c_table, c_col = split_key(cand)
            
            # Load sample values and compute metrics
            q_samples = load_column_samples(q_table, q_col, datalake_dir, sample_count)
            c_samples = load_column_samples(c_table, c_col, datalake_dir, sample_count)
            set_q = get_value_set(q_table, q_col, datalake_dir)
            set_c = get_value_set(c_table, c_col, datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            intersection_fraction = round(len(set_q & set_c) / len(set_q), 3) if set_q else 0.0
            
            if print_samples:
                print_sample_info(q_table, q_col, c_table, c_col, q_samples, c_samples, 
                                 overlap, intersection_fraction, 
                                 "Semantic Right, DeepJoin Wrong")
            
            disagreements['semantic_right_deepjoin_wrong'].append({
                'query': query,
                'candidate': cand,
                'semantic_score': next(score for c, score in semantic_preds[query] if c == cand),
                'deepjoin_score': next((score for c, score in deepjoin_preds[query] if c == cand), 0.0),
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:sample_count]),
                'Jaccard_similarity': overlap,
                'intersection_query_fraction': intersection_fraction,
            })
        
        # DeepJoin right, semantic wrong
        deepjoin_only_correct = deepjoin_correct - semantic_correct
        for cand in deepjoin_only_correct:
            q_table, q_col = split_key(query)
            c_table, c_col = split_key(cand)
            
            # Load sample values and compute metrics
            q_samples = load_column_samples(q_table, q_col, datalake_dir, sample_count)
            c_samples = load_column_samples(c_table, c_col, datalake_dir, sample_count)
            set_q = get_value_set(q_table, q_col, datalake_dir)
            set_c = get_value_set(c_table, c_col, datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            intersection_fraction = round(len(set_q & set_c) / len(set_q), 3) if set_q else 0.0
            
            if print_samples:
                print_sample_info(q_table, q_col, c_table, c_col, q_samples, c_samples, 
                                 overlap, intersection_fraction, 
                                 "DeepJoin Right, Semantic Wrong")
            
            disagreements['deepjoin_right_semantic_wrong'].append({
                'query': query,
                'candidate': cand,
                'semantic_score': next((score for c, score in semantic_preds[query] if c == cand), 0.0),
                'deepjoin_score': next(score for c, score in deepjoin_preds[query] if c == cand),
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:sample_count]),
                'Jaccard_similarity': overlap,
                'intersection_query_fraction': intersection_fraction,
            })
        
        # False positives
        semantic_fp = semantic_set - gt_set
        for cand in semantic_fp:
            q_table, q_col = split_key(query)
            c_table, c_col = split_key(cand)
            
            # Load sample values and compute metrics
            q_samples = load_column_samples(q_table, q_col, datalake_dir, sample_count)
            c_samples = load_column_samples(c_table, c_col, datalake_dir, sample_count)
            set_q = get_value_set(q_table, q_col, datalake_dir)
            set_c = get_value_set(c_table, c_col, datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            intersection_fraction = round(len(set_q & set_c) / len(set_q), 3) if set_q else 0.0
            
            if print_samples:
                print_sample_info(q_table, q_col, c_table, c_col, q_samples, c_samples, 
                                 overlap, intersection_fraction, 
                                 "Semantic False Positive")
            
            disagreements['semantic_false_positives'].append({
                'query': query,
                'candidate': cand,
                'semantic_score': next(score for c, score in semantic_preds[query] if c == cand),
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:sample_count]),
                'Jaccard_similarity': overlap,
                'intersection_query_fraction': intersection_fraction,
            })
        
        deepjoin_fp = deepjoin_set - gt_set
        for cand in deepjoin_fp:
            q_table, q_col = split_key(query)
            c_table, c_col = split_key(cand)
            
            # Load sample values and compute metrics
            q_samples = load_column_samples(q_table, q_col, datalake_dir, sample_count)
            c_samples = load_column_samples(c_table, c_col, datalake_dir, sample_count)
            set_q = get_value_set(q_table, q_col, datalake_dir)
            set_c = get_value_set(c_table, c_col, datalake_dir)
            overlap = round((len(set_q & set_c) / len(set_q | set_c)), 3) if (set_q or set_c) else 0.0
            intersection_fraction = round(len(set_q & set_c) / len(set_q), 3) if set_q else 0.0
            
            if print_samples:
                print_sample_info(q_table, q_col, c_table, c_col, q_samples, c_samples, 
                                 overlap, intersection_fraction, 
                                 "DeepJoin False Positive")
            
            disagreements['deepjoin_false_positives'].append({
                'query': query,
                'candidate': cand,
                'deepjoin_score': next(score for c, score in deepjoin_preds[query] if c == cand),
                'query_samples': ','.join(q_samples),
                'candidate_samples': ','.join(c_samples),
                'overlapping_values': ','.join(list(set_q & set_c)[:sample_count]),
                'Jaccard_similarity': overlap,
                'intersection_query_fraction': intersection_fraction,
            })
    
    return disagreements

def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic join results against DeepJoin baseline')
    
    parser.add_argument('--semantic-results', required=True,
                       help='Path to semantic join results CSV file')
    parser.add_argument('--deepjoin-results', required=True,
                       help='Path to DeepJoin results CSV file')
    parser.add_argument('--ground-truth', required=True,
                       help='Path to ground truth CSV file')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--k-values', type=int, nargs='*', default=[1, 5, 10, 20, 50],
                       help='K values for Precision@k, Recall@k, F1@k calculations')
    parser.add_argument('--datalake-dir', required=False, default='datasets/freyja-semantic-join/datalake',
                       help='Root directory of the datalake for sampling column values')
    parser.add_argument('--sample-count', type=int, default=5, help='Number of sample values per column to include')
    parser.add_argument('--print-samples', action='store_true', help='Print sample values and metrics to console')
    parser.add_argument('--similarity-threshold', type=float, default=0.7, 
                       help='Similarity threshold for filtering semantic results (default: 0.7)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # Load data
    ground_truth = load_ground_truth(args.ground_truth)
    deepjoin_preds = load_deepjoin_results(args.deepjoin_results)
    semantic_preds = load_semantic_results(args.semantic_results, args.similarity_threshold)
    
    # Remove self-joins
    deepjoin_preds = remove_self_joins(deepjoin_preds)
    semantic_preds = remove_self_joins(semantic_preds)
    
    print(f"Loaded {len(ground_truth)} ground truth queries")
    print(f"Loaded {len(deepjoin_preds)} DeepJoin predictions")
    print(f"Loaded {len(semantic_preds)} semantic predictions")
    
    # Calculate metrics
    print("Calculating metrics...")
    
    semantic_metrics = calculate_metrics(semantic_preds, ground_truth, args.k_values)
    deepjoin_metrics = calculate_metrics(deepjoin_preds, ground_truth, args.k_values)
    
    # Analyze disagreements
    print("Analyzing disagreements...")
    disagreements = analyze_disagreements(semantic_preds, deepjoin_preds, ground_truth, 
                                         args.datalake_dir, args.sample_count, args.print_samples)
    
    # Save results
    print("Saving results...")
    
    # Save metrics
    metrics_summary = {
        'semantic_metrics': semantic_metrics,
        'deepjoin_metrics': deepjoin_metrics,
        'comparison': {}
    }
    
    # Add total comparison metrics
    metrics_summary['comparison']['Total_Precision'] = {
        'semantic': semantic_metrics['Total_Precision']['mean'],
        'deepjoin': deepjoin_metrics['Total_Precision']['mean'],
        'improvement': semantic_metrics['Total_Precision']['mean'] - deepjoin_metrics['Total_Precision']['mean']
    }
    metrics_summary['comparison']['Total_Recall'] = {
        'semantic': semantic_metrics['Total_Recall']['mean'],
        'deepjoin': deepjoin_metrics['Total_Recall']['mean'],
        'improvement': semantic_metrics['Total_Recall']['mean'] - deepjoin_metrics['Total_Recall']['mean']
    }
    metrics_summary['comparison']['Total_F1'] = {
        'semantic': semantic_metrics['Total_F1']['mean'],
        'deepjoin': deepjoin_metrics['Total_F1']['mean'],
        'improvement': semantic_metrics['Total_F1']['mean'] - deepjoin_metrics['Total_F1']['mean']
    }
    
    # Add comparison metrics
    for k in args.k_values:
        metrics_summary['comparison'][f'P@{k}'] = {
            'semantic': semantic_metrics[f'P@{k}']['mean'],
            'deepjoin': deepjoin_metrics[f'P@{k}']['mean'],
            'improvement': semantic_metrics[f'P@{k}']['mean'] - deepjoin_metrics[f'P@{k}']['mean']
        }
        metrics_summary['comparison'][f'R@{k}'] = {
            'semantic': semantic_metrics[f'R@{k}']['mean'],
            'deepjoin': deepjoin_metrics[f'R@{k}']['mean'],
            'improvement': semantic_metrics[f'R@{k}']['mean'] - deepjoin_metrics[f'R@{k}']['mean']
        }
        metrics_summary['comparison'][f'F1@{k}'] = {
            'semantic': semantic_metrics[f'F1@{k}']['mean'],
            'deepjoin': deepjoin_metrics[f'F1@{k}']['mean'],
            'improvement': semantic_metrics[f'F1@{k}']['mean'] - deepjoin_metrics[f'F1@{k}']['mean']
        }
    
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save disagreement analysis
    for category, data in disagreements.items():
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_dir / f'{category}.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_queries': len(ground_truth),
        'semantic_predictions': sum(len(preds) for preds in semantic_preds.values()),
        'deepjoin_predictions': sum(len(preds) for preds in deepjoin_preds.values()),
        'disagreements': {
            'semantic_right_deepjoin_wrong': len(disagreements['semantic_right_deepjoin_wrong']),
            'deepjoin_right_semantic_wrong': len(disagreements['deepjoin_right_semantic_wrong']),
            'semantic_false_positives': len(disagreements['semantic_false_positives']),
            'deepjoin_false_positives': len(disagreements['deepjoin_false_positives'])
        }
    }
    
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"\nSUMMARY:")
    print(f"Total queries: {summary_stats['total_queries']}")
    print(f"Semantic predictions: {summary_stats['semantic_predictions']}")
    print(f"DeepJoin predictions: {summary_stats['deepjoin_predictions']}")
    
    # Print results in table format (methods as rows, metrics as columns)
    print(f"\n{'='*120}")
    print(f"EVALUATION RESULTS TABLE")
    print(f"{'='*120}")
    
    # Overall metrics table
    print(f"\nOVERALL METRICS:")
    print(f"{'Method':<12} \t\t\t {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*12} \t\t\t {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'SemSketch':<12} \t\t\t {semantic_metrics['Total_Precision']['mean']:<12.3f} {semantic_metrics['Total_Recall']['mean']:<12.3f} {semantic_metrics['Total_F1']['mean']:<12.3f}")
    print(f"{'DeepJoin':<12} \t\t\t {deepjoin_metrics['Total_Precision']['mean']:<12.3f} {deepjoin_metrics['Total_Recall']['mean']:<12.3f} {deepjoin_metrics['Total_F1']['mean']:<12.3f}")
    
    # Calculate percentage improvements for overall metrics
    precision_improvement = ((semantic_metrics['Total_Precision']['mean'] - deepjoin_metrics['Total_Precision']['mean']) / deepjoin_metrics['Total_Precision']['mean']) * 100
    recall_improvement = ((semantic_metrics['Total_Recall']['mean'] - deepjoin_metrics['Total_Recall']['mean']) / deepjoin_metrics['Total_Recall']['mean']) * 100
    f1_improvement = ((semantic_metrics['Total_F1']['mean'] - deepjoin_metrics['Total_F1']['mean']) / deepjoin_metrics['Total_F1']['mean']) * 100
    
    print(f"{'Improvement over DeepJoin':<12}{precision_improvement:+12.1f}%{recall_improvement:+12.1f}%{f1_improvement:+12.1f}%")
    
    # @k metrics table
    print(f"\nMETRICS BY TOP-K:")
    # Create column headers
    headers = ['Method']
    for k in args.k_values:
        headers.extend([f'\tP@{k}', f'R@{k}', f'F1@{k}'])
    
    # Print header row
    header_line = ""
    for header in headers:
        header_line += f"{header:<12}"
    print(header_line)
    
    # Print separator line
    separator_line = ""
    for header in headers:
        separator_line += f"{'-'*12}"
    print(separator_line)
    
    # Print SemSketch row
    semsketch_line = "SemSketch   \t"
    for k in args.k_values:
        semsketch_line += f"{semantic_metrics[f'P@{k}']['mean']:<12.3f}"
        semsketch_line += f"{semantic_metrics[f'R@{k}']['mean']:<12.3f}"
        semsketch_line += f"{semantic_metrics[f'F1@{k}']['mean']:<12.3f}"
    print(semsketch_line)
    
    # Print DeepJoin row
    deepjoin_line = "DeepJoin    \t"
    for k in args.k_values:
        deepjoin_line += f"{deepjoin_metrics[f'P@{k}']['mean']:<12.3f}"
        deepjoin_line += f"{deepjoin_metrics[f'R@{k}']['mean']:<12.3f}"
        deepjoin_line += f"{deepjoin_metrics[f'F1@{k}']['mean']:<12.3f}"
    print(deepjoin_line)
    
    # Print percentage improvement row
    improvement_line = "Improvement over DeepJoin"
    for k in args.k_values:
        # Calculate percentage improvements
        p_improvement = ((semantic_metrics[f'P@{k}']['mean'] - deepjoin_metrics[f'P@{k}']['mean']) / deepjoin_metrics[f'P@{k}']['mean']) * 100
        r_improvement = ((semantic_metrics[f'R@{k}']['mean'] - deepjoin_metrics[f'R@{k}']['mean']) / deepjoin_metrics[f'R@{k}']['mean']) * 100
        f1_improvement = ((semantic_metrics[f'F1@{k}']['mean'] - deepjoin_metrics[f'F1@{k}']['mean']) / deepjoin_metrics[f'F1@{k}']['mean']) * 100
        
        improvement_line += f"{p_improvement:+12.1f}%"
        improvement_line += f"{r_improvement:+12.1f}%"
        improvement_line += f"{f1_improvement:+12.1f}%"
    print(improvement_line)
    
    # Disagreement analysis table
    print(f"\nDISAGREEMENT ANALYSIS:")
    print(f"{'Category':<30} {'Count':<10}")
    print(f"{'-'*30} {'-'*10}")
    print(f"{'SemSketch right, DeepJoin wrong':<30} {summary_stats['disagreements']['semantic_right_deepjoin_wrong']:<10}")
    print(f"{'DeepJoin right, SemSketch wrong':<30} {summary_stats['disagreements']['deepjoin_right_semantic_wrong']:<10}")
    print(f"{'SemSketch false positives':<30} {summary_stats['disagreements']['semantic_false_positives']:<10}")
    print(f"{'DeepJoin false positives':<30} {summary_stats['disagreements']['deepjoin_false_positives']:<10}")
    
    print(f"\n{'='*120}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
