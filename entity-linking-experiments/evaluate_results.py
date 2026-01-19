"""
Evaluation Script for Semantic Join Results

Evaluates:
1. Joinability: HITS@K (K=1,3,5,7,10) for column matching
2. Entity Linking: Precision and Recall for value-level matching
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def load_joinability_ground_truth(gt_file: str) -> Dict[str, Set[str]]:
    """Load joinability ground truth.
    
    Supports both AutoFJ format (left_table, right_table) and 
    Freyja format (target_ds, target_attr, candidate_ds, candidate_attr).
    
    Returns:
        Dictionary mapping query_table -> set of expected candidate tables
    """
    df = pd.read_csv(gt_file)
    ground_truth = defaultdict(set)
    
    # Detect format based on columns
    if 'left_table' in df.columns:
        # AutoFJ format
        for _, row in df.iterrows():
            left = row['left_table'].replace('.csv', '').lower()
            right = row['right_table'].replace('.csv', '').lower()
            ground_truth[left].add(right)
    elif 'target_ds' in df.columns:
        # Freyja format
        for _, row in df.iterrows():
            query = row['target_ds'].replace('.csv', '').lower()
            candidate = row['candidate_ds'].replace('.csv', '').lower()
            ground_truth[query].add(candidate)
    else:
        raise ValueError(f"Unknown ground truth format. Columns: {df.columns.tolist()}")
    
    return dict(ground_truth)


def load_query_results(results_file: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load query results (ranked candidates per query).
    
    Returns:
        Dictionary mapping query_table -> [(candidate_table, score), ...] ranked by score
    """
    df = pd.read_csv(results_file)
    
    results = defaultdict(list)
    
    for _, row in df.iterrows():
        query_table = str(row['query_table']).replace('.csv', '').lower()
        candidate_table = str(row['candidate_table']).replace('.csv', '').lower()
        score = float(row['similarity_score'])
        
        results[query_table].append((candidate_table, score))
    
    # Sort each query's candidates by score (descending)
    for query in results:
        results[query] = sorted(results[query], key=lambda x: -x[1])
    
    return dict(results)


def evaluate_hits_at_k(ground_truth: Dict[str, Set[str]], 
                       results: Dict[str, List[Tuple[str, float]]], 
                       k_values: List[int]) -> Dict[int, float]:
    """Evaluate HITS@K for joinability.
    
    HITS@K = proportion of queries where at least one correct table appears in top-K candidates
    """
    hits = {k: 0 for k in k_values}
    total_queries = 0
    
    # Track detailed results
    query_details = []
    
    for query_table, expected_candidates in ground_truth.items():
        # Find this query in results
        # Try exact match first, then prefix match
        query_key = None
        query_base = query_table.replace('_left', '').lower()
        
        if query_table.lower() in results:
            query_key = query_table.lower()
        elif query_table.lower() + '_left' in results:
            query_key = query_table.lower() + '_left'
        else:
            # Try prefix match
            for q in results.keys():
                if q.startswith(query_base) or query_base.startswith(q.replace('_left', '')):
                    query_key = q
                    break
        
        if query_key is None:
            continue
        
        total_queries += 1
        
        candidates = results[query_key]
        candidate_tables = [c[0] for c in candidates]
        
        # Convert expected candidates to base names (without _left/_right)
        expected_bases = set()
        for exp in expected_candidates:
            expected_bases.add(exp.replace('_right', '').replace('_left', '').lower())
        
        for k in k_values:
            top_k_candidates = set(c.lower() for c in candidate_tables[:k])
            
            # Check if ANY expected candidate appears in top-K
            # Match exactly or by base name (for AutoFJ: query_left should match candidate_right)
            found_in_top_k = False
            for exp in expected_candidates:
                exp_lower = exp.lower()
                exp_base = exp_lower.replace('_right', '').replace('_left', '')
                
                for cand in top_k_candidates:
                    cand_base = cand.replace('_right', '').replace('_left', '')
                    # Exact match OR base name match with opposite suffix
                    if exp_lower == cand or (exp_base == cand_base and 
                        (('_left' in query_key and '_right' in cand) or 
                         ('_right' in query_key and '_left' in cand) or
                         ('_left' not in query_key and '_right' not in query_key))):
                        found_in_top_k = True
                        break
                if found_in_top_k:
                    break
            
            if found_in_top_k:
                hits[k] += 1
        
        # Store details - find first matching candidate
        found_at = -1
        first_match = None
        for i, c in enumerate(candidate_tables):
            c_lower = c.lower()
            c_base = c_lower.replace('_right', '').replace('_left', '')
            
            for exp in expected_candidates:
                exp_lower = exp.lower()
                exp_base = exp_lower.replace('_right', '').replace('_left', '')
                
                # Exact match OR base name match with opposite suffix
                if exp_lower == c_lower or (exp_base == c_base and 
                    (('_left' in query_key and '_right' in c_lower) or 
                     ('_right' in query_key and '_left' in c_lower) or
                     ('_left' not in query_key and '_right' not in query_key))):
                    found_at = i + 1
                    first_match = c
                    break
            if first_match:
                break
        
        query_details.append({
            'query': query_key,
            'expected': list(expected_candidates)[:3],  # Show first 3 expected
            'num_expected': len(expected_candidates),
            'found_at_rank': found_at if found_at > 0 else 'Not found',
            'first_match': first_match,
            'top_3_candidates': candidate_tables[:3] if candidates else []
        })
    
    # Calculate HITS@K as proportion
    hits_at_k = {k: hits[k] / total_queries if total_queries > 0 else 0.0 
                 for k in k_values}
    
    return hits_at_k, total_queries, query_details


def load_entity_linking_ground_truth(gt_file: str) -> Set[Tuple[str, str, str, str]]:
    """Load entity linking ground truth.
    
    Returns:
        Set of (left_table, left_value, right_table, right_value) tuples
    """
    df = pd.read_csv(gt_file)
    
    ground_truth = set()
    
    for _, row in df.iterrows():
        dataset = str(row['dataset']).lower()
        left_value = str(row['title_l']).strip().lower()
        right_value = str(row['title_r']).strip().lower()
        
        # The left and right tables are {dataset}_left and {dataset}_right
        left_table = f"{dataset}_left"
        right_table = f"{dataset}_right"
        
        ground_truth.add((left_table, left_value, right_table, right_value))
    
    return ground_truth


def load_predicted_entity_links(pred_file: str) -> Set[Tuple[str, str, str, str]]:
    """Load predicted entity links.
    
    Returns:
        Set of (query_table, query_value, candidate_table, candidate_value) tuples
    """
    df = pd.read_csv(pred_file)
    
    predictions = set()
    
    for _, row in df.iterrows():
        query_table = str(row['query_table']).strip().lower()
        query_value = str(row['query_value']).strip().lower()
        candidate_table = str(row['candidate_table']).strip().lower()
        candidate_value = str(row['candidate_value']).strip().lower()
        
        predictions.add((query_table, query_value, candidate_table, candidate_value))
    
    return predictions


def evaluate_entity_linking(ground_truth: Set[Tuple[str, str, str, str]], 
                            predictions: Set[Tuple[str, str, str, str]]) -> Dict[str, float]:
    """Evaluate entity linking precision and recall.
    
    Note: We match based on (left_value, right_value) pairs, ignoring table names
    since the LLM might predict links between different table pairs.
    """
    # Extract just value pairs for comparison
    gt_value_pairs = set()
    for left_table, left_value, right_table, right_value in ground_truth:
        gt_value_pairs.add((left_value, right_value))
        # Also add the reverse since entity linking can be bidirectional
        gt_value_pairs.add((right_value, left_value))
    
    pred_value_pairs = set()
    for query_table, query_value, candidate_table, candidate_value in predictions:
        pred_value_pairs.add((query_value, candidate_value))
    
    # Calculate intersection
    true_positives = pred_value_pairs & gt_value_pairs
    
    # Precision = TP / (TP + FP) = TP / |predictions|
    precision = len(true_positives) / len(pred_value_pairs) if pred_value_pairs else 0.0
    
    # Recall = TP / (TP + FN) = TP / |ground_truth|
    # Use original ground truth size (not doubled)
    gt_original_size = len(ground_truth)
    recall = len(true_positives) / gt_original_size if gt_original_size > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': len(true_positives),
        'total_predictions': len(pred_value_pairs),
        'total_ground_truth': gt_original_size
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic join results')
    parser.add_argument('--query-results', type=str, 
                       default='autofj_query_results_k1024_t0.1_top10_slurm/all_query_results.csv',
                       help='Path to query results CSV')
    parser.add_argument('--joinability-gt', type=str,
                       default='../datasets/autofj_join_benchmark/groundtruth-joinable.csv',
                       help='Path to joinability ground truth CSV')
    parser.add_argument('--entity-links', type=str,
                       default='autofj_query_results_k1024_t0.1_top10_slurm/llm_entity_links.csv',
                       help='Path to predicted entity links CSV')
    parser.add_argument('--entity-linking-gt', type=str,
                       default='../datasets/autofj_join_benchmark/groundtruth-entity-linking.csv',
                       help='Path to entity linking ground truth CSV')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional output file for results JSON')
    args = parser.parse_args()
    
    print("=" * 80)
    print("SEMANTIC JOIN EVALUATION RESULTS")
    print("=" * 80)
    
    # =========================================================================
    # Part 1: Joinability Evaluation (HITS@K)
    # =========================================================================
    print("\n" + "=" * 40)
    print("JOINABILITY EVALUATION (HITS@K)")
    print("=" * 40)
    
    if Path(args.query_results).exists() and Path(args.joinability_gt).exists():
        print(f"\nQuery results: {args.query_results}")
        print(f"Ground truth: {args.joinability_gt}")
        
        # Load data
        gt_joinable = load_joinability_ground_truth(args.joinability_gt)
        query_results = load_query_results(args.query_results)
        
        print(f"\nGround truth pairs: {len(gt_joinable)}")
        print(f"Queries in results: {len(query_results)}")
        
        # Evaluate HITS@K
        k_values = [1, 3, 5, 7, 10]
        hits_at_k, total_queries, query_details = evaluate_hits_at_k(gt_joinable, query_results, k_values)
        
        print(f"\nEvaluated queries: {total_queries}")
        print("\nHITS@K Results:")
        print("-" * 30)
        for k in k_values:
            print(f"  HITS@{k}: {hits_at_k[k]:.4f} ({int(hits_at_k[k] * total_queries)}/{total_queries})")
        
        # Show some examples
        print("\nSample query details (first 5):")
        for detail in query_details[:5]:
            print(f"  Query: {detail['query']}")
            expected_str = detail['expected'] if isinstance(detail['expected'], str) else detail['expected'][:3]
            print(f"    Expected ({detail.get('num_expected', 1)} total): {expected_str}")
            print(f"    Found at rank: {detail['found_at_rank']}")
            if detail.get('first_match'):
                print(f"    First match: {detail['first_match']}")
            print(f"    Top 3 candidates: {detail['top_3_candidates']}")
    else:
        print(f"\n❌ Could not find query results or ground truth file")
        print(f"Query results: {args.query_results}")
        print(f"Ground truth: {args.joinability_gt}")
        hits_at_k = {}
    
    # =========================================================================
    # Part 2: Entity Linking Evaluation (Precision/Recall)
    # =========================================================================
    print("\n" + "=" * 40)
    print("ENTITY LINKING EVALUATION")
    print("=" * 40)
    
    if Path(args.entity_links).exists() and Path(args.entity_linking_gt).exists():
        print(f"\nPredicted links: {args.entity_links}")
        print(f"Ground truth: {args.entity_linking_gt}")
        
        # Load data
        gt_entity_links = load_entity_linking_ground_truth(args.entity_linking_gt)
        pred_entity_links = load_predicted_entity_links(args.entity_links)
        
        print(f"\nGround truth entity pairs: {len(gt_entity_links)}")
        print(f"Predicted entity pairs: {len(pred_entity_links)}")
        
        # Evaluate
        el_metrics = evaluate_entity_linking(gt_entity_links, pred_entity_links)
        
        print("\nEntity Linking Results:")
        print("-" * 30)
        print(f"  Precision: {el_metrics['precision']:.4f}")
        print(f"  Recall:    {el_metrics['recall']:.4f}")
        print(f"  F1 Score:  {el_metrics['f1']:.4f}")
        print(f"\n  True Positives:    {el_metrics['true_positives']}")
        print(f"  Total Predictions: {el_metrics['total_predictions']}")
        print(f"  Total Ground Truth: {el_metrics['total_ground_truth']}")
        
        # Show some example predictions
        print("\nSample predictions (first 10):")
        pred_list = list(pred_entity_links)[:10]
        for qt, qv, ct, cv in pred_list:
            print(f"  {qv[:30]:30s} <-> {cv[:30]:30s}")
    else:
        print(f"\n❌ Could not find entity links or ground truth file")
        print(f"Predicted links: {args.entity_links}")
        print(f"Ground truth: {args.entity_linking_gt}")
        el_metrics = {}
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if hits_at_k:
        print("\nJoinability (HITS@K):")
        for k in [1, 3, 5, 10]:
            if k in hits_at_k:
                print(f"  HITS@{k}: {hits_at_k[k]:.4f}")
    
    if el_metrics:
        print("\nEntity Linking:")
        print(f"  Precision: {el_metrics['precision']:.4f}")
        print(f"  Recall:    {el_metrics['recall']:.4f}")
        print(f"  F1 Score:  {el_metrics['f1']:.4f}")
    
    # Save results to JSON if requested
    if args.output:
        import json
        results = {
            'joinability': {
                'hits_at_k': hits_at_k,
                'total_queries': total_queries if hits_at_k else 0
            },
            'entity_linking': el_metrics
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

