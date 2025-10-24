import os
import argparse
import csv
import pickle
from typing import Dict, List, Tuple, Set

from dataprocess.process_table_tosentence import process_table_sentense
from Forward1 import process_onedataset
import hnsw_search as hnsw_search_module

import pandas as pd


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


def remove_self_joins(preds: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    """Remove self-joins from predictions."""
    cleaned: Dict[str, List[Tuple[str, float]]] = {}
    for q, items in preds.items():
        cleaned[q] = [(c, s) for c, s in items if c != q]
    return cleaned


def calculate_traditional_metrics(predictions: Dict[str, List[Tuple[str, float]]], 
                                ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """Calculate traditional Precision, Recall, and F1 metrics."""
    
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
    
    return {
        'true_positives': total_tp,
        'total_predictions': total_tp + total_fp,
        'total_ground_truth': total_tp + total_fn,
        'precision': total_precision,
        'recall': total_recall,
        'f1': total_f1
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end DeepJoin runner")
    
    # Dataset and output paths
    parser.add_argument("--dataset_root", required=True, 
                       help="Path to dataset root containing lake/ and queries/ subdirectories")
    parser.add_argument("--out_dir", required=True, 
                       help="Directory to store intermediate files (sentences, embeddings) and final results")
    
    # Model configuration
    parser.add_argument("--model_dir", default="all-mpnet-base-v2", 
                       help="Local model directory path OR HuggingFace model name (e.g., 'sentence-transformers/all-mpnet-base-v2'). "
                            "If local path doesn't exist, will download from HF.")
    
    # Dataset structure
    parser.add_argument("--lake_subdir", default="lake", 
                       help="Subdirectory under dataset_root containing data lake CSV files")
    parser.add_argument("--queries_subdir", default="queries", 
                       help="Subdirectory under dataset_root containing query CSV files")
    
    # Preprocessing parallelism
    parser.add_argument("--split_lake", type=int, default=8, 
                       help="Number of parallel processes for preprocessing data lake tables. "
                            "Higher = faster but more memory usage. Adjust based on CPU cores.")
    parser.add_argument("--split_queries", type=int, default=4, 
                       help="Number of parallel processes for preprocessing query tables. "
                            "Usually smaller than lake since fewer query tables.")
    
    # Indexing and retrieval
    parser.add_argument("--scale", type=float, default=1.0, 
                       help="Fraction of data lake tables to index (0.0-1.0]. "
                            "1.0 = use all tables, 0.5 = use 50% randomly sampled. "
                            "Lower values speed up iteration on large datasets.")
    
    # Retrieval parameters
    parser.add_argument("--K", type=int, default=10, 
                       help="Top-K candidate tables to return per query. "
                            "Ignored if --all_above_threshold is used.")
    parser.add_argument("--N", type=int, default=10, 
                       help="Number of nearest neighbor columns retrieved from HNSW index per query column. "
                            "Higher N = more candidates considered, slower but potentially better recall.")
    parser.add_argument("--threshold", type=float, default=0.7, 
                       help="Cosine similarity threshold for column matching (0.0-1.0). "
                            "Only column pairs with similarity >= threshold are considered matches. "
                            "Higher = stricter matching, fewer false positives.")
    parser.add_argument("--min_matches", type=int, default=1,
                       help="Minimum number of matched columns required to output a candidate table.")
    
    # Retrieval mode
    parser.add_argument("--all_above_threshold", action="store_true", default=False, 
                       help="Return ALL candidates with >= threshold matches (disable top-K truncation). "
                            "Use this for threshold-based retrieval instead of top-K ranking.")
    parser.add_argument("--exact_matching", action="store_true", default=False,
                       help="Use exact matching instead of HNSW (slower but more accurate)")

    # Evaluation (optional)
    parser.add_argument("--ground_truth", default=None, 
                       help="Path to ground truth CSV for evaluation (optional)")
    parser.add_argument("--gt_query_col", default="query_table", 
                       help="Ground truth column name for query table identifiers")
    parser.add_argument("--gt_candidate_col", default="candidate_table", 
                       help="Ground truth column name for candidate table identifiers")

    # Optional: Restrict queries to a list from CSV (to match evaluation set)
    parser.add_argument("--queries_csv", default=None,
                       help="CSV file listing query tables to run (e.g., freyja_query_columns.csv)")
    parser.add_argument("--queries_csv_col", default=None,
                       help="Column in --queries_csv that contains query table names (e.g., target_ds)")

    # Self-join mode: use datalake as both query set and candidate set
    parser.add_argument("--self_join", action="store_true", default=False,
                       help="Use datalake tables as queries as well (discover joinable pairs within the lake)")

    # Sampling parameters
    parser.add_argument("--sampling_mode", type=str, default="frequent", 
                       choices=["frequent", "random", "mixed", "weighted", "priority_sampling"],
                       help="Sampling strategy for column values: 'frequent' (original Deepjoin), "
                            "'random', 'mixed', 'weighted', or 'priority_sampling'")

    args = parser.parse_args()

    # Setup paths
    dataset_root = os.path.abspath(args.dataset_root)
    out_dir = os.path.abspath(args.out_dir)
    lake_dir = os.path.join(dataset_root, args.lake_subdir)
    queries_dir = os.path.join(dataset_root, args.queries_subdir)

    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")

    lake_sentences = os.path.join(out_dir, "freyja_lake_sentences.pkl")
    query_sentences = os.path.join(out_dir, "freyja_queries_sentences.pkl")

    # Step 1: Preprocess CSVs -> column-level sentences
    # Converts each CSV table into a list of "sentences" describing each column's values
    # Format: [(table_name, [sentence1, sentence2, ...]), ...]
    
    # Check if lake sentences already exist
    if not os.path.exists(lake_sentences):
        print(f"Creating lake sentences: {lake_sentences}")
        process_table_sentense(
            filepathstore=out_dir,
            datadir=lake_dir,
            data_pkl_name=os.path.basename(lake_sentences),
            tmppath=tmp_dir,
            split_num=args.split_lake,
            sampling_mode=args.sampling_mode,
        )
    else:
        print(f"Using existing lake sentences: {lake_sentences}")
    
    if not args.self_join:
        # Check if query sentences already exist
        if not os.path.exists(query_sentences):
            print(f"Creating query sentences: {query_sentences}")
            process_table_sentense(
                filepathstore=out_dir,
                datadir=queries_dir,
                data_pkl_name=os.path.basename(query_sentences),
                tmppath=tmp_dir,
                split_num=args.split_queries,
                sampling_mode=args.sampling_mode,
            )
        else:
            print(f"Using existing query sentences: {query_sentences}")

    # Step 2: Embed sentences with MPNet
    # Uses SentenceTransformer to convert text sentences into dense vectors
    # Each table becomes: (table_name, numpy_array[columns, embedding_dim])
    model_path = args.model_dir if os.path.isdir(args.model_dir) else "sentence-transformers/all-mpnet-base-v2"
    # Generate embeddings from sentence pickles; write to distinct output files
    lake_embeddings = os.path.join(out_dir, "freyja_lake_embeddings.pkl")
    query_embeddings = os.path.join(out_dir, "freyja_queries_embeddings.pkl")
    process_onedataset(
        dataset_file=lake_sentences,
        model_name=model_path,
        storepath=out_dir,
        output_filename=os.path.basename(lake_embeddings),
    )
    if args.self_join:
        # In self-join, queries are the lake itself
        query_sentences = lake_sentences
    else:
        process_onedataset(
            dataset_file=query_sentences,
            model_name=model_path,
            storepath=out_dir,
            output_filename=os.path.basename(query_embeddings),
        )

    # Step 3: Build index and perform retrieval
    if args.exact_matching:
        # Exact matching: compare all query columns against all candidate columns
        print("Using exact matching (slower but more accurate)...")
        searcher = None  # Will implement exact matching in the query loop
    else:
        # HNSW (Hierarchical Navigable Small World) is a fast approximate nearest neighbor search
        # Indexes all data lake table columns for sublinear search time
        index_path = os.path.join(out_dir, "hnsw_index.bin")
        searcher = hnsw_search_module.HNSWSearcher(table_path=lake_embeddings, index_path=index_path, scale=args.scale)

    queries = pickle.load(open(query_embeddings if not args.self_join else lake_embeddings, "rb"))

    # Optional filter: restrict queries to those listed in a CSV
    if args.queries_csv and os.path.isfile(args.queries_csv) and args.queries_csv_col:
        try:
            qdf = pd.read_csv(args.queries_csv)
            if args.queries_csv_col in qdf.columns:
                # Normalize names: remove .csv for comparison
                allowed = set(qdf[args.queries_csv_col].astype(str).str.replace(".csv", "", regex=False))
                def clean_table_name(name):
                    return name.replace("datalake-", "").replace(".csv", "")
                queries = [q for q in queries if clean_table_name(q[0]) in allowed]
                print(f"Restricted queries to {len(queries)} tables from {args.queries_csv}")
            else:
                print(f"[WARN] Column {args.queries_csv_col} not found in {args.queries_csv}; skipping restriction")
        except Exception as e:
            print(f"[WARN] Failed to read/parse queries_csv {args.queries_csv}: {e}")
    results_csv = os.path.join(out_dir, f"deepjoin_results_{args.sampling_mode}_K{args.K}_N{args.N}_T{args.threshold}.csv")
    if os.path.exists(results_csv):
        os.remove(results_csv)

    # Step 4: Query processing and result output
    # For each query table, find candidate tables and compute column-level matches
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_table", "candidate_table", "query_col", "candidate_col", "score"])
        
        # Helper function to get column names from CSV files
        def get_column_names(table_name, col_index):
            try:
                # Remove datalake- prefix if present and add .csv extension
                clean_name = table_name.replace("datalake-", "")
                if not clean_name.endswith('.csv'):
                    clean_name += '.csv'
                
                # Try to find the CSV file in the datalake directory
                csv_path = os.path.join(lake_dir, clean_name)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, nrows=0)  # Read only header
                    if col_index < len(df.columns):
                        return df.columns[col_index]
                return f"col_{col_index}"  # Fallback to index if file not found
            except Exception:
                return f"col_{col_index}"  # Fallback to index on any error
        
        for q in queries:
            q_name = q[0]
            
            if args.exact_matching:
                # Exact matching: compare against all tables in the lake
                lake_tables = pickle.load(open(lake_sentences, "rb"))
                for cand in lake_tables:
                    cand_name = cand[0]
                    if args.self_join and cand_name == q_name:
                        continue
                    
                    # Use the same verification function as HNSW
                    # Create a temporary searcher instance to access the _verify method
                    temp_searcher = hnsw_search_module.HNSWSearcher(table_path=lake_sentences, index_path="", scale=1.0)
                    score, union_columns = temp_searcher._verify(q[1], cand[1], args.threshold)
                    
                    if union_columns and len(union_columns) >= args.min_matches:
                        for (qi, ci, sim) in union_columns:
                            query_col_name = get_column_names(q_name, qi)
                            candidate_col_name = get_column_names(cand_name, ci)
                            w.writerow([q_name, cand_name, query_col_name, candidate_col_name, f"{sim:.4f}"])
            else:
                # HNSW-based matching
                if args.all_above_threshold:
                    # Threshold-only mode: return ALL candidates with >= threshold matches
                    # No top-K truncation - useful when you want all matches above a threshold
                    query_cols = [col for col in q[1]]
                    candidates = searcher._find_candidates(query_cols, N=args.N)
                    for cand in candidates:
                        cand_name = cand[0]
                        if args.self_join and cand_name == q_name:
                            continue
                        score, union_columns = searcher._verify(q[1], cand[1], args.threshold)
                        if union_columns and len(union_columns) >= args.min_matches:
                            for (qi, ci, sim) in union_columns:
                                query_col_name = get_column_names(q_name, qi)
                                candidate_col_name = get_column_names(cand_name, ci)
                                w.writerow([q_name, cand_name, query_col_name, candidate_col_name, f"{sim:.4f}"])
                else:
                    # Default: Top-K mode - return best K candidates ranked by total similarity score
                    res, _ = searcher.topk(enc="cl", query=q, K=args.K, N=args.N, threshold=args.threshold)
                    for score, union_columns, cand_name in res:
                        if args.self_join and cand_name == q_name:
                            continue
                        if union_columns and len(union_columns) >= args.min_matches:
                            for (qi, ci, sim) in union_columns:
                                query_col_name = get_column_names(q_name, qi)
                                candidate_col_name = get_column_names(cand_name, ci)
                                w.writerow([q_name, cand_name, query_col_name, candidate_col_name, f"{sim:.4f}"])

    print("Saved:", results_csv)

    # Optional: Evaluate against ground truth using built-in evaluation functions
    if args.ground_truth and os.path.isfile(args.ground_truth):
        try:
            print("Loading ground truth...")
            ground_truth = load_ground_truth(args.ground_truth)
            
            print("Loading DeepJoin results...")
            deepjoin_predictions = load_deepjoin_results(results_csv)
            
            print(f"Ground truth: {len(ground_truth)} query tables")
            print(f"DeepJoin: {len(deepjoin_predictions)} query tables")
            
            # Remove self-joins
            deepjoin_predictions = remove_self_joins(deepjoin_predictions)
            
            # Filter to ground truth queries only
            gt_queries = set(ground_truth.keys())
            deepjoin_filtered = {q: deepjoin_predictions[q] for q in gt_queries if q in deepjoin_predictions}
            
            print(f"Filtering to ground truth queries only: {len(deepjoin_filtered)} queries")
            
            # Calculate traditional metrics using the same function as evaluate_semantic_join.py
            traditional_results = calculate_traditional_metrics(deepjoin_filtered, ground_truth)
            
            print("Evaluation (column-pair level):")
            print(f"  TP: {traditional_results['true_positives']}  Total Predictions: {traditional_results['total_predictions']}  Total Ground Truth: {traditional_results['total_ground_truth']}")
            print(f"  Precision: {traditional_results['precision']:.3f}  Recall: {traditional_results['recall']:.3f}  F1: {traditional_results['f1']:.3f}")
            
        except Exception as e:
            print(f"[WARN] Failed to evaluate using built-in functions: {e}")
            print("Falling back to simple table-pair evaluation...")
            
            # Fallback to simple evaluation
            gt = pd.read_csv(args.ground_truth)
            gt_q = args.gt_query_col
            gt_c = args.gt_candidate_col
            if gt_q not in gt.columns or gt_c not in gt.columns:
                print(f"[WARN] Ground truth missing required columns: {gt_q}, {gt_c}")
                return

            # Load predictions: set of (query_table, candidate_table)
            preds = pd.read_csv(results_csv)
            
            # Clean table names by removing datalake- prefix and .csv extension for comparison
            def clean_table_name(name):
                clean = name.replace("datalake-", "").replace(".csv", "")
                return clean
            
            pred_pairs = set(zip(preds["query_table"].apply(clean_table_name), 
                                preds["candidate_table"].apply(clean_table_name)))

            # Ground truth: set of (query_table, candidate_table) - also clean names
            gt_pairs = set(zip(gt[gt_q].apply(lambda x: x.replace(".csv", "")), 
                              gt[gt_c].apply(lambda x: x.replace(".csv", ""))))

            tp = len(pred_pairs & gt_pairs)
            fp = len(pred_pairs - gt_pairs)
            fn = len(gt_pairs - pred_pairs)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

            print("Evaluation (table-pair level):")
            print(f"  TP: {tp}  FP: {fp}  FN: {fn}")
            print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")


if __name__ == "__main__":
    main()


