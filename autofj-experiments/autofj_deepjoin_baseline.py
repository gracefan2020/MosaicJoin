from autofj import AutoFJ
from autofj.datasets import load_data
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys


def load_valid_queries(query_columns_path):
    """Load valid (query_table, query_col) pairs from autofj_query_columns.csv."""
    if not Path(query_columns_path).exists():
        print(f"Warning: Query columns file not found: {query_columns_path}")
        return None
    
    query_cols_df = pd.read_csv(query_columns_path)
    valid_queries = set()
    for _, row in query_cols_df.iterrows():
        query_table = str(row['target_ds']).strip()
        query_col = str(row['target_attr']).strip()
        valid_queries.add((query_table, query_col))
    
    print(f"Loaded {len(valid_queries)} valid queries from {query_columns_path}")
    return valid_queries


def create_semantic_results_from_deepjoin(deepjoin_results_path, output_path):
    """Create semantic results CSV from deepjoin results for evaluation."""
    deepjoin_df = pd.read_csv(deepjoin_results_path)
    # Rename columns to match expected format
    semantic_df = deepjoin_df.rename(columns={
        'query_col': 'query_column',
        'candidate_col': 'candidate_column',
        'score': 'similarity_score'
    })
    semantic_df.to_csv(output_path, index=False)
    return output_path

def generate_entity_linking_from_deepjoin(deepjoin_results_path, datalake_dir, output_dir, target_precision=0.9, 
                                         start_index=None, end_index=None, pairs_file=None, query_columns_path=None):
    """Generate entity linking results from deepjoin joinability results.
    
    Args:
        deepjoin_results_path: Path to deepjoin results CSV (query_table, query_col, candidate_table, candidate_col, score)
        datalake_dir: Directory containing the CSV data files
        output_dir: Directory to save entity linking match files
        target_precision: Precision target for AutoFJ (default 0.9)
        query_columns_path: Path to autofj_query_columns.csv to filter valid queries (optional)
    """
    deepjoin_df = pd.read_csv(deepjoin_results_path)
    datalake_path = Path(datalake_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load valid queries if query_columns_path is provided
    valid_queries = None
    if query_columns_path:
        valid_queries = load_valid_queries(query_columns_path)
        if valid_queries:
            # Filter deepjoin_df to only include valid queries
            original_count = len(deepjoin_df)
            deepjoin_df = deepjoin_df[
                deepjoin_df.apply(
                    lambda row: (str(row['query_table']).strip(), str(row['query_col']).strip()) in valid_queries,
                    axis=1
                )
            ]
            filtered_count = len(deepjoin_df)
            print(f"Filtered deepjoin results: {original_count} -> {filtered_count} rows (kept {filtered_count/original_count*100:.1f}%)")
    
    # If pairs_file is provided, load unique pairs from it (for SLURM parallelization)
    if pairs_file and Path(pairs_file).exists():
        unique_pairs_df = pd.read_csv(pairs_file)
        # Filter to the chunk for this job
        if start_index is not None and end_index is not None:
            unique_pairs_df = unique_pairs_df.iloc[start_index:end_index+1]
        # Create a set of pairs to process
        pairs_to_process = set()
        for _, row in unique_pairs_df.iterrows():
            pairs_to_process.add((str(row['query_table']), str(row['query_col']), 
                                 str(row['candidate_table']), str(row['candidate_col'])))
    else:
        pairs_to_process = None
    
    # Group by (query_table, query_col, candidate_table, candidate_col) to process each pair once
    processed_pairs = set()
    
    # Statistics tracking
    stats = {
        "total_pairs": 0,
        "skipped_missing_files": 0,
        "skipped_missing_columns": 0,
        "skipped_missing_id": 0,
        "skipped_empty_after_clean": 0,
        "autofj_failed": 0,
        "autofj_empty": 0,
        "missing_result_columns": 0,
        "matches_saved": 0,
    }
    
    # Track error types for debugging
    error_types = {}
    
    # Process ALL candidates returned by DeepJoin (top-K per query, not just top-1)
    # This improves recall by generating entity linking matches for all discovered joinable pairs
    for _, row in tqdm(deepjoin_df.iterrows(), total=len(deepjoin_df), desc="Processing deepjoin results"):
        query_table = str(row['query_table'])
        query_col = str(row['query_col'])
        candidate_table = str(row['candidate_table'])
        candidate_col = str(row['candidate_col'])
        score = float(row['score'])
        
        pair_key = (query_table, query_col, candidate_table, candidate_col)
        if pair_key in processed_pairs:
            continue
        
        # If pairs_to_process is set (SLURM mode), only process pairs in the set
        if pairs_to_process is not None and pair_key not in pairs_to_process:
            continue
        
        processed_pairs.add(pair_key)
        stats["total_pairs"] += 1
        
        # Load the CSV files
        query_file = datalake_path / query_table
        candidate_file = datalake_path / candidate_table
        
        if not query_file.exists() or not candidate_file.exists():
            stats["skipped_missing_files"] += 1
            continue
        
        try:
            query_df = pd.read_csv(query_file)
            candidate_df = pd.read_csv(candidate_file)
        except Exception as e:
            print(f"Error loading {query_file} or {candidate_file}: {e}")
            continue
        
        # Strip whitespace from column names and normalize query/candidate column names
        query_df.columns = query_df.columns.str.strip()
        candidate_df.columns = candidate_df.columns.str.strip()
        query_col = query_col.strip()
        candidate_col = candidate_col.strip()
        
        if query_col not in query_df.columns or candidate_col not in candidate_df.columns:
            stats["skipped_missing_columns"] += 1
            continue
        
        # Check if id column exists
        if 'id' not in query_df.columns or 'id' not in candidate_df.columns:
            stats["skipped_missing_id"] += 1
            continue
        
        # Prepare tables for AutoFJ: extract id and the join column, rename to "title"
        # AutoFJ will create 'autofj_id' internally, so we keep 'id' as the id_column name
        try:
            # Select columns explicitly
            left_cols = ['id', query_col]
            right_cols = ['id', candidate_col]
            
            # Verify all columns exist before selection
            if not all(col in query_df.columns for col in left_cols):
                continue
            if not all(col in candidate_df.columns for col in right_cols):
                continue
            
            left_table = query_df[left_cols].copy()
            left_table.rename(columns={query_col: 'title'}, inplace=True)
            
            right_table = candidate_df[right_cols].copy()
            right_table.rename(columns={candidate_col: 'title'}, inplace=True)
        except (KeyError, ValueError) as e:
            continue
        
        # Verify columns exist after selection and renaming
        if 'id' not in left_table.columns or 'title' not in left_table.columns:
            continue
        if 'id' not in right_table.columns or 'title' not in right_table.columns:
            continue
        
        # Remove any rows with missing values
        left_table = left_table.dropna(subset=['id', 'title'])
        right_table = right_table.dropna(subset=['id', 'title'])
        
        # Skip if tables are empty after cleaning
        if left_table.empty or right_table.empty:
            stats["skipped_empty_after_clean"] += 1
            continue
        
        # Run AutoFJ to find entity matches
        # NOTE: AutoFJ source has been patched to preserve autofj_id for blocking
        # even when on=['title'] is specified, so we can use on=['title'] for faster joins.
        try:
            fj = AutoFJ(precision_target=target_precision, verbose=False)
            # Use on=['title'] to only join on the title column (faster than on=None)
            # AutoFJ will preserve autofj_id for blocking but won't process it as a join column
            result = fj.join(left_table, right_table, id_column='id', on=None)
        except Exception as e:
            # Track AutoFJ failures
            stats["autofj_failed"] += 1
            error_type = type(e).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            # Print first few errors to help debug
            if stats["autofj_failed"] <= 5:
                print("left_table", left_table.columns)
                print("right_table", right_table.columns)
                print(f"\nAutoFJ error for {query_table}.{query_col} -> {candidate_table}.{candidate_col}: {error_type}: {e}")
            continue
        
        if result.empty:
            stats["autofj_empty"] += 1
            # Log first few empty results to help debug
            if stats["autofj_empty"] <= 5:
                print(f"AutoFJ returned empty result for {query_table}.{query_col} -> {candidate_table}.{candidate_col} "
                      f"(precision_target={target_precision}, left_rows={len(left_table)}, right_rows={len(right_table)})")
            continue
        
        # Validate result has expected columns
        required_cols = ['title_l', 'title_r']
        if not all(col in result.columns for col in required_cols):
            stats["missing_result_columns"] += 1
            continue
        
        # Extract matched values and format for evaluation
        # Result has columns: id_l, title_l, id_r, title_r
        matches = []
        for _, match_row in result.iterrows():
            try:
                query_value = str(match_row['title_l']).strip()
                candidate_value = str(match_row['title_r']).strip()
            except KeyError:
                # Skip rows with missing columns
                continue
            
            matches.append({
                'query_table': query_table,
                'query_column': query_col,
                'query_value': query_value,
                'candidate_table': candidate_table,
                'candidate_column': candidate_col,
                'candidate_value': candidate_value,
                'similarity_score': score  # Use the deepjoin score
            })
        
        if matches:
            # Save matches file in format expected by evaluate_autofj_experiment.py
            # Format: query_{id}_{table}_{column}_{candidate_table}_{candidate_column}_matches.csv
            # For simplicity, use a sanitized filename
            safe_query = query_table.replace('.csv', '').replace('/', '_')
            safe_candidate = candidate_table.replace('.csv', '').replace('/', '_')
            match_filename = f"query_{safe_query}_{query_col}_{safe_candidate}_{candidate_col}_matches.csv"
            match_file = output_path / match_filename
            
            matches_df = pd.DataFrame(matches)
            matches_df.to_csv(match_file, index=False)
            stats["matches_saved"] += 1
    
    # Also create semantic results CSV in the same directory for evaluation
    # Use filtered deepjoin_df if filtering was applied
    semantic_results_path = output_path / "all_query_results.csv"
    if valid_queries:
        # Create semantic results from filtered dataframe
        semantic_df = deepjoin_df.rename(columns={
            'query_col': 'query_column',
            'candidate_col': 'candidate_column',
            'score': 'similarity_score'
        })
        semantic_df.to_csv(semantic_results_path, index=False)
    else:
    create_semantic_results_from_deepjoin(deepjoin_results_path, semantic_results_path)
    
    print(f"\n{'='*60}")
    print(f"Processing Statistics:")
    print(f"{'='*60}")
    print(f"Total unique pairs processed: {stats['total_pairs']}")
    print(f"  - Skipped (missing files): {stats['skipped_missing_files']}")
    print(f"  - Skipped (missing columns): {stats['skipped_missing_columns']}")
    print(f"  - Skipped (missing 'id' column): {stats['skipped_missing_id']}")
    print(f"  - Skipped (empty after cleaning): {stats['skipped_empty_after_clean']}")
    print(f"  - AutoFJ failed (exceptions): {stats['autofj_failed']}")
    if error_types:
        print(f"    Error breakdown:")
        for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      - {err_type}: {count}")
    print(f"  - AutoFJ returned empty results: {stats['autofj_empty']}")
    print(f"  - Missing result columns: {stats['missing_result_columns']}")
    print(f"  - Match files saved: {stats['matches_saved']}")
    print(f"{'='*60}")
    print(f"\nGenerated entity linking files in {output_path}")
    print(f"Created semantic results CSV: {semantic_results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate AutoFJ entity linking from DeepJoin results')
    parser.add_argument('deepjoin_results', help='Path to DeepJoin results CSV')
    parser.add_argument('datalake_dir', help='Path to datalake directory')
    parser.add_argument('output_dir', help='Output directory for match files')
    parser.add_argument('target_precision', type=float, help='AutoFJ precision target')
    parser.add_argument('--start-index', type=int, help='Start index for SLURM parallelization')
    parser.add_argument('--end-index', type=int, help='End index for SLURM parallelization')
    parser.add_argument('--pairs-file', help='Path to unique pairs CSV file (for SLURM)')
    parser.add_argument('--query-columns', help='Path to autofj_query_columns.csv to filter valid queries')
    
    args = parser.parse_args()
    generate_entity_linking_from_deepjoin(
        args.deepjoin_results, 
        args.datalake_dir, 
        args.output_dir, 
        args.target_precision,
        start_index=args.start_index,
        end_index=args.end_index,
        pairs_file=args.pairs_file,
        query_columns_path=args.query_columns
    )


    