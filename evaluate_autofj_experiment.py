"""
AutoFJ Benchmark Evaluation Script

This script evaluates semantic join results on the AutoFJ benchmark,
providing both joinability and entity linking metrics.

Usage:
    python evaluate_autofj_experiment.py \
        --semantic-results autofj_query_results/all_query_results.csv \
        --output-dir autofj_evaluation_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

from query_time import save_entity_linking_results
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_semantic_results(semantic_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load semantic join results from CSV file."""
    df = pd.read_csv(semantic_path)
    
    out: Dict[str, List[Tuple[str, float]]] = {}
    
    for _, r in df.iterrows():
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
    """Calculate Precision@k, Recall@k, F1@k, and Hits@k metrics.
    
    Hits@k: For each query, checks if at least one ground truth column appears
    in the top-k predictions (binary: 1 if yes, 0 if no), then averages over all queries.
    """
    metrics = {}
    
    # Calculate average precision, recall, f1 using adaptive k (all predictions per query)
    # Each query may return different number of results, so we use all of them
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    pred_counts = []  # Track number of predictions per query
    
    for query, preds in predictions.items():
        if query not in ground_truth:
            continue
        
        gt_set = ground_truth[query]
        # Use all predictions for this query (adaptive k based on actual number returned)
        pred_set = set([cand for cand, _ in preds])
        pred_counts.append(len(pred_set))
        
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
        
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_f1.append(f1)
    
    metrics['Average_Precision'] = {'mean': np.mean(avg_precision), 'std': np.std(avg_precision), 'median': np.median(avg_precision)}
    metrics['Average_Recall'] = {'mean': np.mean(avg_recall), 'std': np.std(avg_recall), 'median': np.median(avg_recall)}
    metrics['Average_F1'] = {'mean': np.mean(avg_f1), 'std': np.std(avg_f1), 'median': np.median(avg_f1)}
    # Store average number of predictions per query
    metrics['Avg_Pred_Count'] = {'mean': np.mean(pred_counts) if pred_counts else 0.0, 'std': np.std(pred_counts) if pred_counts else 0.0, 'median': np.median(pred_counts) if pred_counts else 0.0}
    
    # Calculate @k metrics
    for k in k_values:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        hits_scores = []  # Hits@K: binary (1 if any GT in top-K, 0 otherwise)
        
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
                hit = 0.0  # No predictions = no hit
            else:
                tp = len(pred_set.intersection(gt_set))
                precision = tp / len(pred_set)
                recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                hit = 1.0 if tp > 0 else 0.0  # Hit if at least one GT column is in top-K
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            hits_scores.append(hit)
        
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
            metrics[f'Hits@{k}'] = {
                'mean': np.mean(hits_scores),
                'std': np.std(hits_scores),
                'median': np.median(hits_scores)
            }
        else:
            metrics[f'P@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'R@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'F1@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
            metrics[f'Hits@{k}'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
    
    # Calculate metrics at ground truth size (adaptive k per query)
    precision_at_gt_size = []
    recall_at_gt_size = []
    f1_at_gt_size = []
    gt_sizes = []
    
    for query, preds in predictions.items():
        if query not in ground_truth:
            continue
        
        gt_set = ground_truth[query]
        gt_size = len(gt_set)
        gt_sizes.append(gt_size)
        
        # Use top-k predictions where k = size of ground truth
        top_k_preds = [cand for cand, _ in preds[:gt_size]]
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
        
        precision_at_gt_size.append(precision)
        recall_at_gt_size.append(recall)
        f1_at_gt_size.append(f1)
    
    # Store metrics at ground truth size
    if precision_at_gt_size:
        metrics['P@GT_Size'] = {
            'mean': np.mean(precision_at_gt_size),
            'std': np.std(precision_at_gt_size),
            'median': np.median(precision_at_gt_size)
        }
        metrics['R@GT_Size'] = {
            'mean': np.mean(recall_at_gt_size),
            'std': np.std(recall_at_gt_size),
            'median': np.median(recall_at_gt_size)
        }
        metrics['F1@GT_Size'] = {
            'mean': np.mean(f1_at_gt_size),
            'std': np.std(f1_at_gt_size),
            'median': np.median(f1_at_gt_size)
        }
        # Also store average ground truth size for reference
        metrics['Avg_GT_Size'] = {
            'mean': np.mean(gt_sizes),
            'std': np.std(gt_sizes),
            'median': np.median(gt_sizes)
        }
    else:
        metrics['P@GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        metrics['R@GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        metrics['F1@GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        metrics['Avg_GT_Size'] = {'mean': 0.0, 'std': 0.0, 'median': 0.0}
    
    return metrics


def load_autofj_joinability_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Load AutoFJ joinability ground truth from CSV file.
    
    Format: dataset, left_table, right_table
    Returns: Dict mapping query_table.column -> set of candidate_table.column
    """
    gt_df = pd.read_csv(gt_path)
    
    by_query: Dict[str, Set[str]] = {}
    for _, r in gt_df.iterrows():
        left_table = str(r['left_table']).replace('.csv', '').lower()
        right_table = str(r['right_table']).replace('.csv', '').lower()
        
        # Both tables are joinable with each other
        # Query from left table, candidate is right table (using 'title' column based on autofj_query_columns.csv)
        q = f"{left_table}.title"
        c = f"{right_table}.title"
        by_query.setdefault(q, set()).add(c)
        
        # Also add reverse: query from right table, candidate is left table
        q_reverse = f"{right_table}.title"
        c_reverse = f"{left_table}.title"
        by_query.setdefault(q_reverse, set()).add(c_reverse)
    
    return by_query


def load_autofj_entity_linking_ground_truth(gt_path: str) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
    """Load AutoFJ entity linking ground truth from CSV file.
    
    Format: id_l, title_l, id_r, title_r, dataset
    Returns: Dict mapping (query_table, query_value) -> set of (candidate_table, candidate_value)
    """
    gt_df = pd.read_csv(gt_path)
    
    # Group by dataset to get table names
    entity_matches: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}

    # Load joinability mapping once (used to resolve left/right table names per dataset)
    joinable_gt = pd.read_csv(Path(gt_path).parent / "groundtruth-joinable.csv")
    
    for _, r in gt_df.iterrows():
        dataset = str(r['dataset'])
        # Normalize titles to lowercase to match the normalized values in saved match files
        # (match files are normalized because embeddings use normalized values)
        title_l = str(r['title_l']).strip().lower()
        title_r = str(r['title_r']).strip().lower()
        
        dataset_row = joinable_gt[joinable_gt['dataset'] == dataset]
        
        if len(dataset_row) > 0:
            # Table names are normalized to lowercase (to match semantic/joinability keys)
            left_table = str(dataset_row.iloc[0]['left_table']).replace('.csv', '').lower()
            right_table = str(dataset_row.iloc[0]['right_table']).replace('.csv', '').lower()
            
            # Create mapping: (left_table, title_l) -> (right_table, title_r)
            key = (left_table, title_l)
            if key not in entity_matches:
                entity_matches[key] = set()
            entity_matches[key].add((right_table, title_r))
            
            # Also add reverse mapping
            key_reverse = (right_table, title_r)
            if key_reverse not in entity_matches:
                entity_matches[key_reverse] = set()
            entity_matches[key_reverse].add((left_table, title_l))
    
    return entity_matches


def evaluate_entity_linking(semantic_results_path: str, 
                           entity_linking_gt: Dict[Tuple[str, str], Set[Tuple[str, str]]],
                           joinability_gt: Dict[str, Set[str]],
                           use_contributing_entities: bool = False,
                           precision_targets: Optional[List[float]] = None) -> Dict[str, float]:
    """Evaluate entity linking performance using saved query-time value matches.
    
    Semantics (what you requested):
      * At query time, we run the semantic join and (optionally) save
        value-level matches for the top-1 candidate column of each query
        using `run_query_processing.py --save-entity-links`.
      * At evaluation time, we DO NOT recompute value matches. Instead, we:
          - Load those saved match files from the query results directory.
          - For each (query_table, query_column, candidate_table, candidate_column)
            pair, treat those saved value pairs as the predicted entity links.
          - Compare them against `groundtruth-entity-linking.csv` restricted
            to that (query, candidate) pair:
                - Precision = |P ∩ G| / |P|
                - Recall    = |P ∩ G| / |G|
      * We still use joinability ground truth to filter: we only evaluate
        a saved (query, candidate) pair if that candidate column is a
        true joinable column for the query.
    """
    # We no longer need semantic_preds for value matching itself, but we
    # keep it to know which queries exist; filtering is done via GT + files.
    semantic_preds = load_semantic_results(semantic_results_path)
    semantic_preds = remove_self_joins(semantic_preds)
    
    # Use sets directly to save memory (avoid storing duplicates)
    all_matches: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    all_ground_truth: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    # Keep max similarity score per predicted match (for precision-target sweeps)
    pred_scores: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {}
    pair_stats: Dict[Tuple[str, str, str, str], Dict] = {}
    # For macro-averaged metrics per query/column pair
    pair_pred_sets: Dict[Tuple[str, str, str, str], Set[Tuple[Tuple[str, str], Tuple[str, str]]]] = {}
    pair_gt_sets: Dict[Tuple[str, str, str, str], Set[Tuple[Tuple[str, str], Tuple[str, str]]]] = {}
    
    # Directory where query-time results (and value matches) were saved
    query_results_dir = Path(semantic_results_path).parent
    # Search recursively for match files (handles SLURM job_*/ subdirectories)
    if use_contributing_entities:
        # Pattern: query_*_*_*_*_contributing_entities.csv (from save_contributing_entities)
        # Format: query_{id}_{table}_{column}_{candidate_table}_{candidate_column}_contributing_entities.csv
        value_match_files = sorted(query_results_dir.glob("**/*_contributing_entities.csv"))
        print(f"\nUsing contributing entities files (entities that contributed to retrieval)")
        print(f"  Found {len(value_match_files)} contributing entity files")
    else:
        # Pattern: query_*_matches.csv (from run_query_processing.py --save-entity-links)
        value_match_files = sorted(query_results_dir.glob("**/query_*_matches.csv"))
        print(f"\nUsing saved value match files (from query-time --save-entity-links)")
        print(f"  Found {len(value_match_files)} value match files")
    
    if not value_match_files:
        file_type = "contributing entities" if use_contributing_entities else "value match"
        print(f"\nWarning: No saved {file_type} files found in {query_results_dir} (searched recursively). "
              f"Run query processing with --save-entity-links first.")
    
    processed_pairs: Set[Tuple[str, str, str, str]] = set()
    eval_stats = {
        "queries_processed": 0,
        "pairs_with_gt": 0,
        "value_match_files": len(value_match_files),
    }
    
    # Iterate over saved value-match files (one per query, top-1 candidate)
    for vm_file in tqdm(value_match_files, desc="Evaluating entity linking from saved value matches"):
        try:
            # Try to read similarity_score if present (contributing entities always include it)
            try:
                df = pd.read_csv(
                    vm_file,
                    usecols=[
                        'query_table', 'query_column', 'query_value',
                        'candidate_table', 'candidate_column', 'candidate_value',
                        'similarity_score'
                    ],
                )
                has_score = True
            except ValueError:
                # Older files may not have similarity_score
                df = pd.read_csv(
                    vm_file,
                    usecols=[
                        'query_table', 'query_column', 'query_value',
                        'candidate_table', 'candidate_column', 'candidate_value'
                    ],
                )
                has_score = False
        except Exception as e:
            print(f"  Warning: could not read {vm_file}: {e}")
            continue
        
        if df.empty:
            continue
        
        # All rows in this file share the same query/candidate metadata
        qt_raw = str(df['query_table'].iloc[0])
        qc = str(df['query_column'].iloc[0])
        ct_raw = str(df['candidate_table'].iloc[0])
        cc = str(df['candidate_column'].iloc[0])
        
        # Normalize table names: remove .csv extension and lowercase to match GT format
        qt = qt_raw.replace('.csv', '').lower()
        ct = ct_raw.replace('.csv', '').lower()
        
        query_id = f"{qt}.{qc}"
        cand_id = f"{ct}.{cc}"
        
        # Filter by joinability ground truth: only evaluate if this
        # candidate is truly joinable with the query
        if query_id not in joinability_gt or cand_id not in joinability_gt[query_id]:
            continue
        
        pair_key = (qt, qc, ct, cc)
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        eval_stats["queries_processed"] += 1
        
        # Build P (predictions) and G (ground truth) for this pair
        # Vectorized operations for speed (much faster than iterrows())
        q_vals = df['query_value'].astype(str).str.strip().str.lower()
        c_vals = df['candidate_value'].astype(str).str.strip().str.lower()
        
        # Create all query_keys and candidate_keys at once
        query_keys = [(qt, qv) for qv in q_vals]
        candidate_keys = [(ct, cv) for cv in c_vals]
        
        # Build predicted matches in bulk
        pair_predicted_matches = [(qk, ck) for qk, ck in zip(query_keys, candidate_keys)]
        # Add to global set (automatically deduplicates)
        all_matches.update(pair_predicted_matches)

        # Track max similarity score per predicted match (if available)
        if has_score:
            scores = df['similarity_score'].astype(float).to_numpy()
            for m, s in zip(pair_predicted_matches, scores.tolist()):
                prev = pred_scores.get(m)
                if prev is None or s > prev:
                    pred_scores[m] = s
        else:
            # No scores: treat all predictions as score=1.0 for sweep purposes
            for m in pair_predicted_matches:
                if m not in pred_scores:
                    pred_scores[m] = 1.0
        
        # Build ground truth matches efficiently
        # Use set to track unique GT matches to avoid duplicates
        pair_gt_matches_set: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
        pair_has_gt = False
        
        # Iterate through unique query_keys to avoid redundant GT lookups
        for query_key in set(query_keys):
            if query_key in entity_linking_gt:
                for gt_table, gt_val in entity_linking_gt[query_key]:
                    if gt_table == ct:
                        gt_match = (query_key, (gt_table, gt_val))
                        if gt_match not in pair_gt_matches_set:
                            pair_gt_matches_set.add(gt_match)
                            all_ground_truth.add(gt_match)  # Use set.add() instead of list.append()
                            pair_has_gt = True
        
        pair_gt_matches = list(pair_gt_matches_set)
        
        pair_stats[pair_key] = {
            "num_predicted_matches": len(set(pair_predicted_matches)),
            "num_ground_truth_matches": len(set(pair_gt_matches)),
            # We don't have full stats here; we approximate with counts from file
            "total_query_values": len(df['query_value'].unique()),
            "total_candidate_values": len(df['candidate_value'].unique()),
        }
        pair_pred_sets[pair_key] = set(pair_predicted_matches)
        pair_gt_sets[pair_key] = set(pair_gt_matches)
        if pair_has_gt:
            eval_stats["pairs_with_gt"] += 1
    
    # Diagnostics
    print(f"\nEntity Linking Evaluation Statistics (from saved query-time value matches, top-1 candidate):")
    print(f"  Value match files found: {eval_stats['value_match_files']}")
    print(f"  Evaluated pairs (query, candidate): {len(processed_pairs)}")
    print(f"  Pairs with ground truth: {eval_stats['pairs_with_gt']}")
    
    # all_matches and all_ground_truth are already sets, no need to convert
    predicted_set = all_matches
    gt_set = all_ground_truth
    
    tp = len(predicted_set & gt_set)
    fp = len(predicted_set) - tp
    fn = len(gt_set) - tp
    
    # Lightweight diagnostics (no large data structures)
    print(f"\nMatch Statistics:")
    print(f"  |predicted_set| = {len(predicted_set):,}")
    print(f"  |gt_set| = {len(gt_set):,}")
    print(f"  TP = {tp:,}, FP = {fp:,}, FN = {fn:,}")
    
    # Sample matches for debugging (only if sets are small enough)
    if len(predicted_set) > 0 and len(predicted_set) < 1000:
        sample_pred = next(iter(predicted_set))
        print(f"  Sample predicted match: {sample_pred}")
    if len(gt_set) > 0 and len(gt_set) < 1000:
        sample_gt = next(iter(gt_set))
        print(f"  Sample GT match: {sample_gt}")
    
    if len(predicted_set) == 0 and len(gt_set) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_predicted_matches": 0,
            "total_ground_truth_matches": 0,
            "pair_stats": pair_stats,
            "all_matches": [],
            "all_ground_truth": [],
        }
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Sanity check: if precision=0 and recall=1, this is mathematically impossible
    if precision == 0.0 and recall == 1.0:
        print(f"\nWARNING: Precision=0 and Recall=1 is mathematically inconsistent!")
        print(f"  This suggests a bug in the calculation or match construction.")
        print(f"  TP={tp}, FP={fp}, FN={fn}, |predicted|={len(predicted_set)}, |gt|={len(gt_set)}")

    # Macro-averaged metrics over query/column pairs (only pairs with any GT)
    per_pair_precisions = []
    per_pair_recalls = []
    per_pair_f1s = []
    for pk in pair_gt_sets:
        gt_p = pair_gt_sets[pk]
        if not gt_p:
            continue  # skip pairs with no ground truth
        pred_p = pair_pred_sets.get(pk, set())
        tp_p = len(pred_p & gt_p)
        fp_p = len(pred_p) - tp_p
        fn_p = len(gt_p) - tp_p
        prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
        rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0
        per_pair_precisions.append(prec_p)
        per_pair_recalls.append(rec_p)
        per_pair_f1s.append(f1_p)

    avg_precision = float(np.mean(per_pair_precisions)) if per_pair_precisions else 0.0
    avg_recall = float(np.mean(per_pair_recalls)) if per_pair_recalls else 0.0
    avg_f1 = float(np.mean(per_pair_f1s)) if per_pair_f1s else 0.0
    
    # Precision-target sweep: maximize recall subject to precision >= target
    recall_at_precision_targets: Dict[str, float] = {}
    threshold_at_precision_targets: Dict[str, float] = {}
    pred_count_at_precision_targets: Dict[str, int] = {}
    if precision_targets:
        items = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
        total_gt = len(gt_set)
        tp_run = 0
        pred_run = 0
        best = {t: {"recall": 0.0, "threshold": float("inf"), "pred": 0} for t in precision_targets}
        for match, score in items:
            pred_run += 1
            if match in gt_set:
                tp_run += 1
            if pred_run == 0:
                continue
            prec_run = tp_run / pred_run
            rec_run = (tp_run / total_gt) if total_gt > 0 else 0.0
            for t in precision_targets:
                if prec_run >= t and rec_run >= best[t]["recall"]:
                    best[t] = {"recall": rec_run, "threshold": score, "pred": pred_run}
        for t in precision_targets:
            recall_at_precision_targets[str(t)] = float(best[t]["recall"])
            threshold_at_precision_targets[str(t)] = float(best[t]["threshold"]) if best[t]["threshold"] != float("inf") else 0.0
            pred_count_at_precision_targets[str(t)] = int(best[t]["pred"])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_predicted_matches": len(predicted_set),
        "total_ground_truth_matches": len(gt_set),
        "avg_precision_per_query": avg_precision,
        "avg_recall_per_query": avg_recall,
        "avg_f1_per_query": avg_f1,
        "recall_at_precision_targets": recall_at_precision_targets,
        "threshold_at_precision_targets": threshold_at_precision_targets,
        "predicted_count_at_precision_targets": pred_count_at_precision_targets,
        "pair_stats": pair_stats,
        "all_matches": list(predicted_set),
        "all_ground_truth": list(gt_set),
    }


def print_joinability_results(metrics: Dict[str, Dict[str, float]], k_values: List[int],
                              k1_analysis: Dict = None):
    """Print joinability evaluation results."""
    print("\n" + "="*80)
    print("JOINABILITY EVALUATION RESULTS")
    print("="*80)
    
    # Print metrics for all requested k
    ks_sorted = sorted(set(k_values))
    print("\nSummary by k:")
    for k in ks_sorted:
        p_key = f"P@{k}"
        r_key = f"R@{k}"
        f_key = f"F1@{k}"
        h_key = f"Hits@{k}"
        if p_key in metrics and r_key in metrics and f_key in metrics:
            p = metrics[p_key]
            r = metrics[r_key]
            f = metrics[f_key]
            h = metrics.get(h_key, {'mean': 0.0, 'std': 0.0})
            print(f"  k = {k:3d}: "
                  f"P@{k} = {p['mean']:.3f}, "
                  f"R@{k} = {r['mean']:.3f}, "
                  f"F1@{k} = {f['mean']:.3f}, "
                  f"Hits@{k} = {h['mean']:.3f})")
    
    # Optional: we keep the k=1 detailed analysis function around, but
    # we no longer invoke it by default to avoid slow per-query value checks.


def print_entity_linking_results(metrics: Dict[str, float]):
    """Print entity linking evaluation results."""
    print("\n" + "="*80)
    print("ENTITY LINKING EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")

    # Optional: recall at fixed precision targets (AutoFJ-style)
    if metrics.get("recall_at_precision_targets"):
        print("\nRecall at fixed precision targets (maximize recall s.t. precision >= target):")
        for t, r in metrics["recall_at_precision_targets"].items():
            thr = metrics.get("threshold_at_precision_targets", {}).get(t, None)
            n = metrics.get("predicted_count_at_precision_targets", {}).get(t, None)
            if thr is not None and n is not None:
                print(f"  Precision ≥ {t}: Recall = {r:.3f} (threshold ≥ {thr:.3f}, predicted={n})")
            else:
                print(f"  Precision ≥ {t}: Recall = {r:.3f}")
    
    print(f"\nDetailed Statistics:")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Total Predicted Matches: {metrics['total_predicted_matches']}")
    print(f"  Total Ground Truth Matches: {metrics['total_ground_truth_matches']}")
    
    if metrics['total_predicted_matches'] > 0:
        precision_pct = (metrics['true_positives'] / metrics['total_predicted_matches']) * 100
        print(f"  Precision: {precision_pct:.1f}% ({metrics['true_positives']}/{metrics['total_predicted_matches']})")
    
    if metrics['total_ground_truth_matches'] > 0:
        recall_pct = (metrics['true_positives'] / metrics['total_ground_truth_matches']) * 100
        print(f"  Recall: {recall_pct:.1f}% ({metrics['true_positives']}/{metrics['total_ground_truth_matches']})")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate semantic join results on AutoFJ benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate both joinability and entity linking
  python evaluate_autofj_experiment.py \\
      --semantic-results autofj_query_results/all_query_results.csv \\
      --output-dir autofj_evaluation_results

  # Only evaluate joinability (faster)
  python evaluate_autofj_experiment.py \\
      --semantic-results autofj_query_results/all_query_results.csv \\
      --output-dir autofj_evaluation_results \\
      --skip-entity-linking

  # Only evaluate entity linking
  python evaluate_autofj_experiment.py \\
      --semantic-results autofj_query_results/all_query_results.csv \\
      --output-dir autofj_evaluation_results \\
      --skip-joinability
        """
    )
    
    parser.add_argument('--semantic-results', required=True,
                       help='Path to semantic join results CSV file')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--joinability-gt', 
                       default='datasets/autofj_join_benchmark/groundtruth-joinable.csv',
                       help='Path to joinability ground truth (default: datasets/autofj_join_benchmark/groundtruth-joinable.csv)')
    parser.add_argument('--entity-linking-gt',
                       default='datasets/autofj_join_benchmark/groundtruth-entity-linking.csv',
                       help='Path to entity linking ground truth (default: datasets/autofj_join_benchmark/groundtruth-entity-linking.csv)')
    parser.add_argument('--k-values', type=int, nargs='*',
                       default=[1, 3, 5, 10],
                       help='K values for Precision@k, Recall@k, F1@k, and Hits@k calculations '
                            '(default: 1 3 5 10)')
    parser.add_argument('--skip-joinability', action='store_true',
                       help='Skip joinability evaluation')
    parser.add_argument('--skip-entity-linking', action='store_true',
                       help='Skip entity linking evaluation')
    parser.add_argument('--use-contributing-entities', action='store_true',
                       help='Use contributing entities files instead of threshold-based matches for entity linking evaluation')
    parser.add_argument('--precision-targets', type=float, nargs='*',
                       help='AutoFJ-style: report recall at fixed precision targets by sweeping similarity_score thresholds '
                            '(maximize recall s.t. precision >= target). Example: --precision-targets 0.7 0.8 0.9')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("AUTOFJ BENCHMARK EVALUATION")
    print("="*80)
    print(f"Semantic Results: {args.semantic_results}")
    print(f"Output Directory: {args.output_dir}")
    
    # Load semantic results
    print("\nLoading semantic results...")
    semantic_preds = load_semantic_results(args.semantic_results)
    semantic_preds = remove_self_joins(semantic_preds)
    print(f"Loaded {len(semantic_preds)} semantic predictions")
    
    results = {}
    
    # Evaluate joinability
    if not args.skip_joinability:
        print("\n" + "="*80)
        print("EVALUATING JOINABILITY")
        print("="*80)
        
        joinability_gt = load_autofj_joinability_ground_truth(args.joinability_gt)
        print(f"Loaded {len(joinability_gt)} joinability ground truth queries")
        
        # Calculate metrics
        print("Calculating joinability metrics...")
        joinability_metrics = calculate_metrics(semantic_preds, joinability_gt, args.k_values)
        
        # Print results for all requested k (no per-query k=1 analysis)
        print_joinability_results(joinability_metrics, args.k_values)
        
        # Save results
        joinability_output = output_dir / "joinability_metrics.json"
        with open(joinability_output, 'w') as f:
            json.dump(joinability_metrics, f, indent=2)
        print(f"\n✓ Saved joinability metrics to {joinability_output}")
        
        results['joinability'] = joinability_metrics
    
    # Evaluate entity linking
    if not args.skip_entity_linking:
        print("\n" + "="*80)
        print("EVALUATING ENTITY LINKING")
        print("="*80)
        print("Note: This may take a while as it performs value-level matching...")
        
        entity_linking_gt = load_autofj_entity_linking_ground_truth(args.entity_linking_gt)
        print(f"Loaded {len(entity_linking_gt)} entity linking ground truth entries")
        
        # Evaluate entity linking
        print("Calculating entity linking metrics...")
        # Load joinability GT to know which columns are true matches
        joinability_gt_for_el = load_autofj_joinability_ground_truth(args.joinability_gt)
        entity_linking_metrics = evaluate_entity_linking(
            args.semantic_results,
            entity_linking_gt,
            joinability_gt_for_el,
            use_contributing_entities=args.use_contributing_entities,
            precision_targets=args.precision_targets,
        )
        
        # Print results
        print_entity_linking_results(entity_linking_metrics)
        
        # Save aggregate metrics (without detailed data)
        metrics_summary = {k: v for k, v in entity_linking_metrics.items() 
                          if k not in ('pair_stats', 'all_matches', 'all_ground_truth')}
        entity_linking_output = output_dir / "entity_linking_metrics.json"
        with open(entity_linking_output, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\n✓ Saved entity linking metrics to {entity_linking_output}")
        
        # Save detailed results (matches and per-pair stats) if available
        if 'all_matches' in entity_linking_metrics and 'pair_stats' in entity_linking_metrics:
            detailed_output_dir = output_dir / "entity_linking_detailed"
            save_entity_linking_results(
                entity_linking_metrics['all_matches'],
                entity_linking_metrics['all_ground_truth'],
                entity_linking_metrics['pair_stats'],
                detailed_output_dir,
                format="csv"  # Use CSV for universal compatibility
            )
            print(f"✓ Saved detailed entity linking results to {detailed_output_dir}")
        
        results['entity_linking'] = metrics_summary
    
    # Save combined results summary
    if results:
        summary_output = output_dir / "evaluation_summary.json"
        summary = {
            'semantic_results_file': args.semantic_results,
            'k_values': args.k_values,
            'results': results
        }
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved evaluation summary to {summary_output}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
