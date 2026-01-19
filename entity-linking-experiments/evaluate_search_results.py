"""
Evaluation Script for Table Retrieval Results

Compares SemSketch vs DeepJoin on:
- HITS@K, Precision@K, Recall@K for K = 1 to 20
- Shows examples where each method succeeds and the other fails
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def load_ground_truth(gt_file: str) -> Dict[str, str]:
    """Load joinability ground truth.
    
    Returns:
        Dictionary mapping query_table -> expected_candidate_table
    """
    df = pd.read_csv(gt_file)
    ground_truth = {}
    
    if 'left_table' in df.columns:
        for _, row in df.iterrows():
            left = row['left_table'].replace('.csv', '').lower()
            right = row['right_table'].replace('.csv', '').lower()
            ground_truth[left] = right
    elif 'target_ds' in df.columns:
        for _, row in df.iterrows():
            query = row['target_ds'].replace('.csv', '').lower()
            candidate = row['candidate_ds'].replace('.csv', '').lower()
            ground_truth[query] = candidate
    else:
        raise ValueError(f"Unknown ground truth format. Columns: {df.columns.tolist()}")
    
    return ground_truth


def load_query_results(results_file: str, exclude_self: bool = True) -> Dict[str, List[Tuple[str, float]]]:
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
        
        # Skip self-matches
        if exclude_self and query_table == candidate_table:
            continue
        
        results[query_table].append((candidate_table, score))
    
    # Sort by score descending
    for query in results:
        results[query] = sorted(results[query], key=lambda x: -x[1])
    
    return dict(results)


def evaluate_metrics(ground_truth: Dict[str, str], 
                     results: Dict[str, List[Tuple[str, float]]], 
                     k_values: List[int]) -> Tuple[Dict[str, Dict[int, float]], int, Dict[str, int]]:
    """Evaluate HITS@K, Precision@K, Recall@K.
    
    For this benchmark (1 relevant per query):
    - HITS@K = proportion of queries with correct answer in top-K
    - Precision@K = avg(1/K if found in top-K, else 0) = HITS@K / K
    - Recall@K = same as HITS@K (since there's only 1 relevant per query)
    
    Returns:
        metrics: {'hits': {k: rate}, 'precision': {k: rate}, 'recall': {k: rate}}
        total_queries: number of evaluated queries
        query_ranks: {query: rank_of_correct_answer} (-1 if not found)
    """
    hits = {k: 0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    total = 0
    query_ranks = {}
    
    for query, expected in ground_truth.items():
        if query not in results:
            continue
        total += 1
        
        candidates = [c[0] for c in results[query]]
        
        # Find rank of correct answer
        rank = -1
        for i, cand in enumerate(candidates):
            if cand == expected:
                rank = i + 1
                break
        query_ranks[query] = rank
        
        # Check metrics at each K
        for k in k_values:
            if 0 < rank <= k:
                hits[k] += 1
                precision_sum[k] += 1.0 / k  # Precision = 1/K when found
    
    metrics = {
        'hits': {k: hits[k] / total if total > 0 else 0.0 for k in k_values},
        'precision': {k: precision_sum[k] / total if total > 0 else 0.0 for k in k_values},
        'recall': {k: hits[k] / total if total > 0 else 0.0 for k in k_values}  # Same as HITS for 1 relevant
    }
    return metrics, total, query_ranks


def print_metrics(metrics: Dict[str, Dict[int, float]], total: int, name: str, k_values: List[int]):
    """Print formatted metrics table."""
    print(f"\n{'=' * 60}")
    print(f"{name} (n={total} queries)")
    print(f"{'=' * 60}")
    print(f"{'K':>4} {'HITS@K':>12} {'Precision@K':>14} {'Recall@K':>12}")
    print("-" * 50)
    for k in k_values:
        h = metrics['hits'][k]
        p = metrics['precision'][k]
        r = metrics['recall'][k]
        count = int(h * total)
        print(f"{k:>4} {h:>10.4f} ({count:>2}) {p:>12.4f} {r:>12.4f}")


def find_disagreements(ranks1: Dict[str, int], ranks2: Dict[str, int], k: int = 1):
    """Find queries where methods disagree at HITS@k."""
    method1_wins = []  # method1 correct, method2 wrong
    method2_wins = []  # method2 correct, method1 wrong
    
    common = set(ranks1.keys()) & set(ranks2.keys())
    
    for query in common:
        r1 = ranks1[query]
        r2 = ranks2[query]
        
        m1_hit = 0 < r1 <= k
        m2_hit = 0 < r2 <= k
        
        if m1_hit and not m2_hit:
            method1_wins.append(query)
        elif m2_hit and not m1_hit:
            method2_wins.append(query)
    
    return method1_wins, method2_wins


def analyze_why(query: str, expected: str,
                results_correct: Dict[str, List[Tuple[str, float]]],
                results_wrong: Dict[str, List[Tuple[str, float]]]) -> str:
    """Analyze why one method succeeded and the other failed."""
    
    reasons = []
    
    if query not in results_correct or query not in results_wrong:
        return "Missing results"
    
    correct_list = results_correct[query]
    wrong_list = results_wrong[query]
    
    # Get scores
    correct_score = next((s for c, s in correct_list if c == expected), None)
    wrong_score = next((s for c, s in wrong_list if c == expected), None)
    
    if correct_list and wrong_list:
        top_correct = correct_list[0]
        top_wrong = wrong_list[0]
        
        # Check if wrong method ranked a similar-domain table higher
        correct_base = expected.replace('_right', '').replace('_left', '')
        wrong_top_base = top_wrong[0].replace('_right', '').replace('_left', '')
        
        if wrong_score is None:
            reasons.append("correct table not in top results")
        elif wrong_score and correct_score:
            score_diff = correct_score - wrong_score
            if abs(score_diff) > 0.1:
                reasons.append(f"score gap: {correct_score:.3f} vs {wrong_score:.3f}")
        
        # Check if wrong method confused with similar domain
        if wrong_top_base != correct_base:
            # Check semantic similarity of wrong choice
            if any(x in wrong_top_base for x in ['league', 'tournament', 'season', 'match']):
                if any(x in correct_base for x in ['league', 'tournament', 'season', 'match']):
                    reasons.append(f"confused with similar sports domain: {top_wrong[0]}")
            elif any(x in wrong_top_base for x in ['bishop', 'parliament', 'noble', 'monarch']):
                if any(x in correct_base for x in ['bishop', 'parliament', 'noble', 'monarch']):
                    reasons.append(f"confused with similar person domain: {top_wrong[0]}")
            else:
                reasons.append(f"ranked {top_wrong[0]} higher")
    
    return "; ".join(reasons) if reasons else "marginal score difference"


def print_disagreement_examples(queries: List[str], 
                                 gt: Dict[str, str],
                                 results_correct: Dict[str, List[Tuple[str, float]]],
                                 results_wrong: Dict[str, List[Tuple[str, float]]],
                                 ranks_correct: Dict[str, int],
                                 ranks_wrong: Dict[str, int],
                                 correct_name: str, 
                                 wrong_name: str,
                                 max_examples: int = 10):
    """Print detailed disagreement examples with analysis."""
    
    for i, query in enumerate(queries[:max_examples]):
        expected = gt[query]
        
        # Analyze why
        why = analyze_why(query, expected, results_correct, results_wrong)
        
        print(f"\n{i+1}. Query: {query}")
        print(f"   Expected: {expected}")
        print(f"   Why {wrong_name} failed: {why}")
        print(f"   ")
        
        # Correct method
        rc = ranks_correct.get(query, -1)
        correct_score = None
        print(f"   ✓ {correct_name} (found at rank {rc}):")
        if query in results_correct:
            for j, (cand, score) in enumerate(results_correct[query][:5]):
                marker = "→" if cand == expected else " "
                if cand == expected:
                    correct_score = score
                print(f"      {marker} {j+1}. {cand} ({score:.4f})")
        
        print(f"   ")
        
        # Wrong method
        rw = ranks_wrong.get(query, -1)
        if rw > 0:
            print(f"   ✗ {wrong_name} (found at rank {rw}, too late):")
        else:
            print(f"   ✗ {wrong_name} (not found in results):")
        if query in results_wrong:
            for j, (cand, score) in enumerate(results_wrong[query][:5]):
                marker = "→" if cand == expected else " "
                print(f"      {marker} {j+1}. {cand} ({score:.4f})")
    
    if len(queries) > max_examples:
        print(f"\n... and {len(queries) - max_examples} more examples")


def main():
    parser = argparse.ArgumentParser(description='Compare SemSketch vs DeepJoin table retrieval')
    parser.add_argument('--semsketch-results', type=str, 
                       default='autofj_query_results_k1024_t0.1_top10_slurm/all_query_results.csv',
                       help='Path to SemSketch results CSV')
    parser.add_argument('--deepjoin-results', type=str,
                       default='autofj_deepjoin_baseline_k10_n10_t0.1/all_query_results.csv',
                       help='Path to DeepJoin results CSV')
    parser.add_argument('--ground-truth', type=str,
                       default='../datasets/autofj_join_benchmark/groundtruth-joinable.csv',
                       help='Path to ground truth CSV')
    parser.add_argument('--k-disagree', type=int, default=1,
                       help='K value for disagreement analysis')
    parser.add_argument('--max-examples', type=int, default=10,
                       help='Max examples to show per disagreement type')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TABLE RETRIEVAL EVALUATION: SemSketch vs DeepJoin")
    print("=" * 70)
    
    # Load ground truth
    if not Path(args.ground_truth).exists():
        print(f"❌ Ground truth not found: {args.ground_truth}")
        return 1
    
    gt = load_ground_truth(args.ground_truth)
    print(f"\nGround truth: {len(gt)} query-candidate pairs")
    
    # K values for evaluation
    k_values = [1, 3, 5, 10, 20]
    
    # Load and evaluate SemSketch
    ss_results = None
    ss_ranks = None
    ss_metrics = None
    if Path(args.semsketch_results).exists():
        print(f"SemSketch results: {args.semsketch_results}")
        ss_results = load_query_results(args.semsketch_results, exclude_self=True)
        ss_metrics, ss_total, ss_ranks = evaluate_metrics(gt, ss_results, k_values)
        print_metrics(ss_metrics, ss_total, "SemSketch", k_values)
    else:
        print(f"⚠️  SemSketch results not found: {args.semsketch_results}")
    
    # Load and evaluate DeepJoin
    dj_results = None
    dj_ranks = None
    dj_metrics = None
    if Path(args.deepjoin_results).exists():
        print(f"DeepJoin results: {args.deepjoin_results}")
        dj_results = load_query_results(args.deepjoin_results, exclude_self=True)
        dj_metrics, dj_total, dj_ranks = evaluate_metrics(gt, dj_results, k_values)
        print_metrics(dj_metrics, dj_total, "DeepJoin", k_values)
    else:
        print(f"⚠️  DeepJoin results not found: {args.deepjoin_results}")
    
    # Comparison
    if ss_ranks and dj_ranks:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY (HITS@K)")
        print("=" * 70)
        
        print(f"\n{'K':>4} {'SemSketch':>12} {'DeepJoin':>12} {'Δ':>10} {'Winner':>12}")
        print("-" * 55)
        for k in k_values:
            ss = ss_metrics['hits'][k]
            dj = dj_metrics['hits'][k]
            diff = ss - dj
            winner = "SemSketch" if diff > 0 else ("DeepJoin" if diff < 0 else "Tie")
            print(f"{k:>4} {ss:>12.4f} {dj:>12.4f} {diff:>+10.4f} {winner:>12}")
        
        # Disagreement analysis
        print("\n" + "=" * 70)
        print(f"DISAGREEMENT ANALYSIS (at HITS@{args.k_disagree})")
        print("=" * 70)
        
        ss_wins, dj_wins = find_disagreements(ss_ranks, dj_ranks, k=args.k_disagree)
        
        both_correct = sum(1 for q in ss_ranks if q in dj_ranks 
                          and 0 < ss_ranks[q] <= args.k_disagree 
                          and 0 < dj_ranks[q] <= args.k_disagree)
        both_wrong = sum(1 for q in ss_ranks if q in dj_ranks 
                        and (ss_ranks[q] <= 0 or ss_ranks[q] > args.k_disagree)
                        and (dj_ranks[q] <= 0 or dj_ranks[q] > args.k_disagree))
        
        print(f"\nBoth correct:                      {both_correct}")
        print(f"Both wrong:                        {both_wrong}")
        print(f"SemSketch correct, DeepJoin wrong: {len(ss_wins)}")
        print(f"DeepJoin correct, SemSketch wrong: {len(dj_wins)}")
        
        # Examples: SemSketch wins
        print("\n" + "=" * 70)
        print(f"EXAMPLES: SemSketch correct, DeepJoin wrong ({len(ss_wins)} total)")
        print("=" * 70)
        
        if ss_wins:
            print_disagreement_examples(
                ss_wins, gt, ss_results, dj_results, ss_ranks, dj_ranks,
                "SemSketch", "DeepJoin", max_examples=args.max_examples
            )
        else:
            print("\nNo examples found.")
        
        # Examples: DeepJoin wins
        print("\n" + "=" * 70)
        print(f"EXAMPLES: DeepJoin correct, SemSketch wrong ({len(dj_wins)} total)")
        print("=" * 70)
        
        if dj_wins:
            print_disagreement_examples(
                dj_wins, gt, dj_results, ss_results, dj_ranks, ss_ranks,
                "DeepJoin", "SemSketch", max_examples=args.max_examples
            )
        else:
            print("\nNo examples found.")
    
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
