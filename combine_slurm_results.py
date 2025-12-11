#!/usr/bin/env python3
"""
Combine Results from Slurm Query Processing
Combines query results from multiple Slurm job outputs into a single consolidated file.
"""

import pandas as pd
import json
from pathlib import Path
import glob
from typing import List, Dict, Any, Tuple

def load_all_results(output_dir: Path, num_jobs: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and combine results from all job directories."""
    all_results = []
    summary_stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "total_results": 0
    }
    
    print(f"Loading results from {num_jobs} job directories...")
    
    for job_id in range(num_jobs):
        job_dir = output_dir / f"job_{job_id}"
        print(f"  Loading results from {job_dir}...")
        if not job_dir.exists():
            print(f"  Warning: Job directory {job_dir} does not exist")
            continue
        
        # Load combined results if they exist
        combined_file = job_dir / "all_query_results.csv"
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            all_results.append(df)
            print(f"  Loaded {len(df)} results from job_{job_id}")
        else:
            # Try loading individual query files
            query_files = sorted(glob.glob(str(job_dir / "query_*.csv")))
            if query_files:
                print(f"  Loading individual query files from job_{job_id}...")
                for query_file in query_files:
                    df = pd.read_csv(query_file)
                    all_results.append(df)
        
        # Load summary stats
        summary_file = job_dir / "query_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                job_stats = json.load(f)
                summary_stats['total_queries'] += job_stats.get('total_queries', 0)
                summary_stats['successful_queries'] += job_stats.get('successful_queries', 0)
                summary_stats['failed_queries'] += job_stats.get('failed_queries', 0)
                summary_stats['total_results'] += job_stats.get('total_results', 0)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_stats['avg_results_per_query'] = summary_stats['total_results'] / max(1, summary_stats['successful_queries'])
        return combined_df, summary_stats
    else:
        print("  No results found in any job directory")
        return pd.DataFrame(), summary_stats

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine results from Slurm query processing")
    parser.add_argument('output_dir', type=str, help='Output directory containing job_* subdirectories')
    parser.add_argument('--num-jobs', type=int, required=True, help='Number of Slurm jobs')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    num_jobs = args.num_jobs
    
    if not output_dir.exists():
        print(f"❌ Error: Output directory does not exist: {output_dir}")
        return 1
    
    # Load and combine results
    combined_df, summary_stats = load_all_results(output_dir, num_jobs)
    
    if combined_df.empty:
        print("❌ No results to combine")
        return 1
    
    # Save combined results
    output_file = output_dir / "all_query_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Combined results saved to: {output_file}")
    print(f"   Total results: {len(combined_df)}")
    
    # Save summary statistics
    summary_file = output_dir / "query_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✅ Summary statistics saved to: {summary_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("QUERY PROCESSING SUMMARY")
    print("="*60)
    print(f"Total queries processed: {summary_stats['total_queries']}")
    print(f"Successful queries: {summary_stats['successful_queries']}")
    print(f"Failed queries: {summary_stats['failed_queries']}")
    print(f"Total results: {summary_stats['total_results']}")
    if 'avg_results_per_query' in summary_stats:
        print(f"Average results per query: {summary_stats['avg_results_per_query']:.2f}")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())

