import os
import argparse
import csv
import pickle
from typing import Dict, List, Tuple, Set

from dataprocess.process_table_tosentence import process_table_sentense
from Forward1 import process_onedataset
import hnsw_search as hnsw_search_module

import pandas as pd
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


def calculate_traditional_metrics(predictions: Dict[str, List[Tuple[str, float]]], 
                                ground_truth: Dict[str, Set[str]],
                                k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, float]:
    """Calculate Precision@k, Recall@k, and F1@k metrics."""
    
    metrics = {}
    
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
        
        # Calculate mean metrics
        if precision_scores:
            metrics[f'P@{k}'] = np.mean(precision_scores)
            metrics[f'R@{k}'] = np.mean(recall_scores)
            metrics[f'F1@{k}'] = np.mean(f1_scores)
        else:
            metrics[f'P@{k}'] = 0.0
            metrics[f'R@{k}'] = 0.0
            metrics[f'F1@{k}'] = 0.0
    
    # Also calculate overall metrics (using all predictions)
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
    
    metrics['true_positives'] = total_tp
    metrics['total_predictions'] = total_tp + total_fp
    metrics['total_ground_truth'] = total_tp + total_fn
    metrics['precision'] = total_precision
    metrics['recall'] = total_recall
    metrics['f1'] = total_f1
    
    return metrics


def get_column_names(table_name, col_index, lake_dir):
    """Get column name from CSV file."""
    try:
        clean_name = table_name.replace("datalake-", "")
        if not clean_name.endswith('.csv'):
            clean_name += '.csv'
        
        csv_path = os.path.join(lake_dir, clean_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, nrows=0)
            if col_index < len(df.columns):
                return df.columns[col_index]
        return f"col_{col_index}"
    except Exception:
        return f"col_{col_index}"


def clean_table_name(table_name):
    """Remove 'datalake-' prefix from table name if present."""
    return table_name.replace("datalake-", "")


def main():
    parser = argparse.ArgumentParser(description="DeepJoin: Joinable Table Discovery with Pre-trained Language Models")
    
    # Dataset and output paths
    parser.add_argument("--dataset_root", required=True, 
                       help="Path to dataset root containing lake/ and queries/ subdirectories")
    parser.add_argument("--out_dir", required=True, 
                       help="Directory to store intermediate files and final results")
    
    # Model configuration
    parser.add_argument("--model_dir", default="sentence-transformers/all-mpnet-base-v2", 
                       help="HuggingFace model name or local model directory path")
    
    # Dataset structure
    parser.add_argument("--lake_subdir", default="lake", 
                       help="Subdirectory under dataset_root containing data lake CSV files")
    parser.add_argument("--queries_subdir", default="queries", 
                       help="Subdirectory under dataset_root containing query CSV files")
    
    # Preprocessing parallelism
    parser.add_argument("--split_lake", type=int, default=8, 
                       help="Number of parallel processes for preprocessing data lake tables")
    parser.add_argument("--split_queries", type=int, default=4, 
                       help="Number of parallel processes for preprocessing query tables")

    
    # Retrieval parameters (matching original DeepJoin)
    parser.add_argument("--K", type=int, default=10, 
                       help="Top-K candidate tables to return per query")
    parser.add_argument("--N", type=int, default=10, 
                       help="Number of nearest neighbor columns retrieved from HNSW index per query column")
    parser.add_argument("--threshold", type=float, default=0.6, 
                       help="Cosine similarity threshold for column matching (0.0-1.0)")
    parser.add_argument("--encoder", type=str, default="cl", 
                       choices=["cl", "sherlock", "sato"],
                       help="Encoder type (cl=column-level, sherlock, sato)")
    
    # Evaluation (optional)
    parser.add_argument("--ground_truth", default=None, 
                       help="Path to ground truth CSV for evaluation (optional)")

    args = parser.parse_args()

    # Setup paths
    dataset_root = os.path.abspath(args.dataset_root)
    out_dir = os.path.abspath(args.out_dir)
    lake_dir = os.path.join(dataset_root, args.lake_subdir)
    queries_dir = os.path.join(dataset_root, args.queries_subdir)

    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")

    # File paths for intermediate results
    lake_sentences = os.path.join(out_dir, "lake_sentences.pkl")
    query_sentences = os.path.join(out_dir, "query_sentences.pkl")
    lake_embeddings = os.path.join(out_dir, "lake_embeddings.pkl")
    query_embeddings = os.path.join(out_dir, "query_embeddings.pkl")

    # Step 1: Preprocess CSVs -> column-level sentences
    print("Step 1: Preprocessing data lake tables...")
    process_table_sentense(
        filepathstore=out_dir,
        datadir=lake_dir,
        data_pkl_name=os.path.basename(lake_sentences),
        tmppath=tmp_dir,
        split_num=args.split_lake,
        sampling_mode="frequent",  # Original DeepJoin uses frequent sampling
    )
    
    print("Step 1: Preprocessing query tables...")
    process_table_sentense(
        filepathstore=out_dir,
        datadir=queries_dir,
        data_pkl_name=os.path.basename(query_sentences),
        tmppath=tmp_dir,
        split_num=args.split_queries,
        sampling_mode="frequent",
    )

    # Step 2: Embed sentences with MPNet
    print("Step 2: Generating embeddings for data lake...")
    model_path = args.model_dir if os.path.isdir(args.model_dir) else args.model_dir
    process_onedataset(
        dataset_file=lake_sentences,
        model_name=model_path,
        storepath=out_dir,
        output_filename=os.path.basename(lake_embeddings),
    )
    
    print("Step 2: Generating embeddings for queries...")
    process_onedataset(
        dataset_file=query_sentences,
        model_name=model_path,
        storepath=out_dir,
        output_filename=os.path.basename(query_embeddings),
    )

    # Step 3: Build HNSW index and perform retrieval
    print("Step 3: Building HNSW index...")
    index_path = os.path.join(out_dir, "hnsw_index.bin")
    searcher = hnsw_search_module.HNSWSearcher(
        table_path=lake_embeddings, 
        index_path=index_path, 
        scale=1.0
    )

    print("Step 4: Processing queries...")
    queries = pickle.load(open(query_embeddings, "rb"))
    
    # Cache column names to avoid repeated CSV reads
    print("Caching column names...")
    column_name_cache = {}  # {(table_name, col_index, dir): column_name}
    
    def get_column_names_cached(table_name, col_index, dir_path):
        """Get column name with caching."""
        cache_key = (table_name, col_index, dir_path)
        if cache_key not in column_name_cache:
            column_name_cache[cache_key] = get_column_names(table_name, col_index, dir_path)
        return column_name_cache[cache_key]
    
    results_csv = os.path.join(out_dir, f"deepjoin_results_K{args.K}_N{args.N}_T{args.threshold}.csv")
    if os.path.exists(results_csv):
        os.remove(results_csv)

    # Step 4: Query processing and result output
    # Track statistics per query column
    candidates_per_query_col = []  # List of (query_table.col, num_candidates)
    
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_table", "query_col", "candidate_table", "candidate_col", "score"])
        
        for q in queries:
            q_name = q[0]
            
            # Use original DeepJoin topk interface
            res, _ = searcher.topk(
                enc=args.encoder, 
                query=q, 
                K=args.K, 
                N=args.N, 
                threshold=args.threshold
            )
            
            # Collect all column-level matches per query column, sorted by similarity
            query_col_matches = {}  # {query_col_name: [(cand_table, cand_col, sim), ...]}
            
            # Process results: res contains (score, union_columns, cand_name) tuples
            for score, union_columns, cand_name in res:
                if union_columns:
                    for (qi, ci, sim) in union_columns:
                        query_col_name = get_column_names_cached(q_name, qi, queries_dir)
                        candidate_col_name = get_column_names_cached(cand_name, ci, lake_dir)
                        
                        # Skip self-matches
                        if q_name == cand_name and query_col_name == candidate_col_name:
                            continue
                        
                        # Collect matches per query column
                        if query_col_name not in query_col_matches:
                            query_col_matches[query_col_name] = []
                        query_col_matches[query_col_name].append((
                            clean_table_name(cand_name),
                            candidate_col_name,
                            sim
                        ))
            
            # For each query column, take top K results (sorted by similarity)
            for query_col_name, matches in query_col_matches.items():
                # Sort by similarity (descending) and take top K
                matches_sorted = sorted(matches, key=lambda x: x[2], reverse=True)
                top_k_matches = matches_sorted[:args.K]
                
                # Track count
                query_col_key = f"{clean_table_name(q_name)}.{query_col_name}"
                candidates_per_query_col.append((query_col_key, len(top_k_matches)))
                
                # Write top K results for this query column
                for cand_table, cand_col, sim in top_k_matches:
                    w.writerow([
                        clean_table_name(q_name), 
                        query_col_name, 
                        cand_table, 
                        cand_col, 
                        f"{sim:.4f}"
                    ])

    print(f"Results saved to: {results_csv}")
    
    # Print statistics about candidates per query column
    if candidates_per_query_col:
        counts = [count for _, count in candidates_per_query_col]
        if counts:
            avg_count = sum(counts) / len(counts)
            min_count = min(counts)
            max_count = max(counts)
            print(f"\nCandidates per query column statistics:")
            print(f"  Total query columns: {len(candidates_per_query_col)}")
            print(f"  Average candidates per query column: {avg_count:.2f}")
            print(f"  Min candidates per query column: {min_count}")
            print(f"  Max candidates per query column: {max_count}")
        else:
            print("\nNo candidates found for any query column.")

    # Optional: Evaluate against ground truth
    if args.ground_truth and os.path.isfile(args.ground_truth):
        try:
            print("\nEvaluating against ground truth...")
            ground_truth = load_ground_truth(args.ground_truth)
            deepjoin_predictions = load_deepjoin_results(results_csv)
            
            # Filter to ground truth queries only and skip self-joins
            gt_queries = set(ground_truth.keys())
            deepjoin_filtered = {}
            for q in gt_queries:
                if q in deepjoin_predictions:
                    # Skip candidates that are the same as the query column
                    deepjoin_filtered[q] = [(c, s) for c, s in deepjoin_predictions[q] if c != q]
            
            print(f"Ground truth queries: {len(ground_truth)}")
            print(f"DeepJoin predictions: {len(deepjoin_filtered)}")
            
            traditional_results = calculate_traditional_metrics(deepjoin_filtered, ground_truth)
            
            print("\nEvaluation (column-pair level):")
            print(f"  TP: {traditional_results['true_positives']}")
            print(f"  Total Predictions: {traditional_results['total_predictions']}")
            print(f"  Total Ground Truth: {traditional_results['total_ground_truth']}")
            print(f"  Overall Precision: {traditional_results['precision']:.3f}")
            print(f"  Overall Recall: {traditional_results['recall']:.3f}")
            print(f"  Overall F1: {traditional_results['f1']:.3f}")
            
            print("\nTop-K Metrics:")
            print(f"  {'K':<4} {'P@K':<10} {'R@K':<10} {'F1@K':<10}")
            print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10}")
            for k in [1, 5, 10, 20, 50]:
                if f'P@{k}' in traditional_results:
                    print(f"  {k:<4} {traditional_results[f'P@{k}']:<10.3f} {traditional_results[f'R@{k}']:<10.3f} {traditional_results[f'F1@{k}']:<10.3f}")
            
        except Exception as e:
            print(f"[WARN] Failed to evaluate: {e}")


if __name__ == "__main__":
    main()
