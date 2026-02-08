"""
Query Time Processing Script

Processes semantic join queries using pre-built sketches.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

from query_time import (
    SemanticJoinQueryProcessor,
    QueryConfig,
    QueryColumn,
    save_query_results,
)


def load_query_specifications(query_file: Path) -> List[Dict[str, str]]:
    """Load query specifications from CSV file."""
    try:
        df = pd.read_csv(query_file)
        queries = []
        for _, row in df.iterrows():
            queries.append({
                'target_ds': row['target_ds'],
                'target_attr': row['target_attr']
            })
        return queries
    except Exception as e:
        print(f"Error loading query specifications: {e}")
        return []


def load_query_values_from_datalake(datalake_dir: Path, table_name: str, column_name: str) -> List[str]:
    """Load values from a CSV file in the datalake for a specific column."""
    try:
        csv_file = datalake_dir / f"{table_name}.csv"
        if not csv_file.exists() and table_name.endswith('.csv'):
            csv_file = datalake_dir / table_name
        
        if not csv_file.exists():
            print(f"Warning: Table file not found: {csv_file}")
            return []
        
        df = pd.read_csv(csv_file)
        if column_name in df.columns:
            values = df[column_name].dropna().astype(str).tolist()
            return values
        else:
            print(f"Warning: Column '{column_name}' not found in {csv_file}")
            return []
    except Exception as e:
        print(f"Error loading {table_name}.{column_name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Query time processing for semantic join pipeline")
    
    parser.add_argument("datalake_dir", type=str, help="Path to datalake directory")
    parser.add_argument("sketches_dir", type=str, help="Path to sketches directory")
    parser.add_argument("query_file", type=str, help="Path to query specifications CSV file")
    parser.add_argument("--output-dir", type=str, default="query_results",
                        help="Output directory for query results")
    parser.add_argument("--top-k-return", type=int, default=50,
                        help="Number of final results to return per query")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                        help="Similarity threshold for semantic matching")
    parser.add_argument("--sketch-size", type=int, default=1024,
                        help="Number of representative vectors per sketch")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for MPNet model (auto, cpu, cuda, mps)")
    parser.add_argument("--similarity-method", type=str, default="mean",
                        choices=["chamfer", "mean", "greedy_match", "top_k_mean", "max"],
                        help="Similarity computation method")
    parser.add_argument("--top-k-for-mean", type=int, default=100,
                        help="Number of top pairs to average (for top_k_mean method)")
    parser.add_argument("--embeddings-dir", type=str,
                        help="Path to embeddings directory (optional)")
    parser.add_argument("--max-queries", type=int,
                        help="Maximum number of queries to process")
    parser.add_argument("--query-indices", type=int, nargs="*",
                        help="Specific query indices to process (0-based)")
    parser.add_argument("--embedding-model", type=str, default="mpnet",
                        choices=["mpnet", "embeddinggemma"],
                        help="Embedding model to use (must match datalake embeddings)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        choices=[128, 256, 512, 768],
                        help="Embedding dimension for embeddinggemma (default: 128)")
    
    args = parser.parse_args()
    
    datalake_path = Path(args.datalake_dir)
    sketches_path = Path(args.sketches_dir)
    query_file_path = Path(args.query_file)
    output_dir = Path(args.output_dir)
    
    if not datalake_path.exists():
        print(f"Error: Datalake directory {datalake_path} does not exist")
        return 1
    
    if not sketches_path.exists():
        print(f"Error: Sketches directory {sketches_path} does not exist")
        return 1
    
    if not query_file_path.exists():
        print(f"Error: Query file {query_file_path} does not exist")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading query specifications from {query_file_path}")
    query_specs = load_query_specifications(query_file_path)
    
    if not query_specs:
        print("No query specifications found")
        return 1
    
    print(f"Found {len(query_specs)} query specifications")
    
    if args.query_indices:
        query_specs = [query_specs[i] for i in args.query_indices if i < len(query_specs)]
        print(f"Processing {len(query_specs)} specified queries")
    elif args.max_queries:
        query_specs = query_specs[:args.max_queries]
        print(f"Processing first {len(query_specs)} queries")
    
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    
    query_config = QueryConfig(
        top_k_return=args.top_k_return,
        similarity_threshold=args.similarity_threshold,
        sketch_size=args.sketch_size,
        device=args.device,
        similarity_method=args.similarity_method,
        top_k_for_mean=args.top_k_for_mean,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim
    )
    
    processor = SemanticJoinQueryProcessor(
        query_config,
        sketches_path,
        embeddings_dir,
        datalake_dir=datalake_path,
    )
    
    all_results = []
    successful_queries = 0
    
    print(f"\nProcessing {len(query_specs)} queries with similarity_method={args.similarity_method}...")
    
    for i, query_spec in enumerate(query_specs):
        table_name = query_spec['target_ds']
        column_name = query_spec['target_attr']
        
        print(f"Processing query {i+1}/{len(query_specs)}: {table_name}.{column_name}")
        
        query_values = load_query_values_from_datalake(datalake_path, table_name, column_name)
        
        if not query_values:
            print(f"  Skipping: No values found for {table_name}.{column_name}")
            continue
        
        print(f"  Loaded {len(query_values)} values")
        
        query = QueryColumn(
            table_name=table_name,
            column_name=column_name,
            values=query_values
        )
        
        results = processor.process_query(query)
        
        if results:
            successful_queries += 1
            print(f"  Found {len(results)} similar columns")
            
            query_output_file = output_dir / f"query_{i+1:03d}_{table_name}_{column_name}.csv"
            save_query_results(results, query_output_file,
                               query_sample_values=query_values,
                               datalake_dir=datalake_path)
            
            
            query_sample = query_values[:5] if query_values else []
            query_sample_str = ", ".join(str(v) for v in query_sample) if query_sample else ""
            
            for result in results:
                # Skip self-matches
                if result.candidate_table == table_name and result.candidate_column == column_name:
                    continue
                candidate_values = load_query_values_from_datalake(
                    datalake_path, result.candidate_table, result.candidate_column
                )
                candidate_sample = candidate_values[:5] if candidate_values else []
                candidate_sample_str = ", ".join(str(v) for v in candidate_sample) if candidate_sample else ""
                
                all_results.append({
                    'query_table': table_name,
                    'query_column': column_name,
                    'query_index': i + 1,
                    'candidate_table': result.candidate_table,
                    'candidate_column': result.candidate_column,
                    'similarity_score': result.similarity_score,
                    'query_sample_values': query_sample_str,
                    'candidate_sample_values': candidate_sample_str
                })
        else:
            print(f"  No similar columns found")
    
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_output_file = output_dir / "all_query_results.csv"
        combined_df.to_csv(combined_output_file, index=False)
        print(f"\nCombined results saved to: {combined_output_file}")
    
    summary_stats = {
        "total_queries": len(query_specs),
        "successful_queries": successful_queries,
        "failed_queries": len(query_specs) - successful_queries,
        "total_results": len(all_results),
        "avg_results_per_query": len(all_results) / max(1, successful_queries),
        "query_config": {
            "top_k_return": args.top_k_return,
            "similarity_threshold": args.similarity_threshold,
            "similarity_method": args.similarity_method,
            "sketch_size": args.sketch_size,
            "device": args.device
        }
    }
    
    summary_file = output_dir / "query_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    processor.print_stats()
    
    print(f"\nQuery processing completed!")
    print(f"Successful queries: {successful_queries}/{len(query_specs)}")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
