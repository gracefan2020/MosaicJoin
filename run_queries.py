#!/usr/bin/env python3
"""
Query Processing Runner Script
Simple script to run query processing with common configurations.

DeepJoin Integration:
To enable DeepJoin integration during query processing, set use_deepjoin_index = True and provide:
- deepjoin_embeddings_path: Path to DeepJoin lake embeddings
- deepjoin_query_embeddings_path: Path to DeepJoin query embeddings
- deepjoin_index_path: Path to DeepJoin HNSW index (optional)

⚠️ IMPORTANT: DeepJoin filtering should NOT be used during sketch building as it can skip useful columns.

Example DeepJoin configuration:
use_deepjoin_index = True
deepjoin_embeddings_path = "Deepjoin/output/lake_embeddings.pkl"
deepjoin_query_embeddings_path = "Deepjoin/output/query_embeddings.pkl"
deepjoin_index_path = "Deepjoin/output/hnsw_index.bin"
"""

import os
import shutil
import subprocess
import re
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any

def cleanup_query_data(output_dir: str):
    """Clean up previous query data."""
    print("🧹 Cleaning up previous query data...")
    
    # Clean query results directory
    query_results_dir = Path(output_dir)
    if query_results_dir.exists():
        print(f"Removing previous query results: {query_results_dir}")
        shutil.rmtree(query_results_dir)
        print("✅ Cleaned previous query results")

def parse_query_stats(output: str) -> Dict[str, Any]:
    """Parse query processing statistics from output."""
    stats = {}
    
    # Extract statistics from the output
    lines = output.split('\n')
    
    for line in lines:
        # Parse total queries
        if "Total queries:" in line:
            stats['total_queries'] = int(re.search(r'Total queries: (\d+)', line).group(1))
        
        # Parse successful queries
        elif "Successful queries:" in line:
            stats['successful_queries'] = int(re.search(r'Successful queries: (\d+)', line).group(1))
        
        # Parse failed queries
        elif "Failed queries:" in line:
            stats['failed_queries'] = int(re.search(r'Failed queries: (\d+)', line).group(1))
        
        # Parse total processing time
        elif "Total processing time:" in line:
            stats['total_processing_time'] = float(re.search(r'Total processing time: ([\d.]+)s', line).group(1))
        
        # Parse total candidates found
        elif "Total candidates found:" in line:
            stats['total_candidates_found'] = int(re.search(r'Total candidates found: (\d+)', line).group(1))
        
        # Parse average time per query
        elif "Average time per query:" in line:
            stats['avg_time_per_query'] = float(re.search(r'Average time per query: ([\d.]+)s', line).group(1))
        
        # Parse average candidates per query
        elif "Average candidates per query:" in line:
            stats['avg_candidates_per_query'] = float(re.search(r'Average candidates per query: ([\d.]+)', line).group(1))
        
        # Parse high-quality candidates statistics
        elif "Total high-quality candidates:" in line:
            stats['total_high_quality_candidates'] = int(re.search(r'Total high-quality candidates: (\d+)', line).group(1))
        
        elif "Average high-quality candidates per query:" in line:
            stats['avg_high_quality_candidates'] = float(re.search(r'Average high-quality candidates per query: ([\d.]+)', line).group(1))
        
        # Parse DeepJoin statistics
        elif "DeepJoin queries processed:" in line:
            stats['deepjoin_queries_processed'] = int(re.search(r'DeepJoin queries processed: (\d+)', line).group(1))
        
        elif "Total DeepJoin candidates:" in line:
            stats['total_deepjoin_candidates'] = int(re.search(r'Total DeepJoin candidates: (\d+)', line).group(1))
        
        elif "Average DeepJoin candidates per query:" in line:
            stats['avg_deepjoin_candidates'] = float(re.search(r'Average DeepJoin candidates per query: ([\d.]+)', line).group(1))
    
    return stats

def save_timing_stats(stats: Dict[str, Any], output_dir: str):
    """Save timing statistics to a JSON file."""
    output_path = Path(output_dir)
    timing_file = output_path / "timing_stats.json"
    
    with open(timing_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Timing statistics saved to: {timing_file}")

def run_with_progress(cmd: str, max_queries: int = None):
    """Run command with progress indicators."""
    
    # Show expected progress
    if max_queries is None:
        print("📊 Processing all queries (estimated 50 queries)")
        print("⏳ This may take several minutes...")
    else:
        print(f"📊 Processing {max_queries} queries")
        print("⏳ Progress will be shown below...")
    
    print("-" * 60)
    
    # Start the process
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Track progress
    output_lines = []
    query_count = 0
    start_time = time.time()
    
    try:
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line)
                print(line.rstrip())  # Print without extra newline
                
                # Track query progress with simple indicators
                if "Processing query" in line and "/" in line:
                    # Extract query number from "Processing query X/Y:"
                    match = re.search(r'Processing query (\d+)/(\d+)', line)
                    if match:
                        current_query = int(match.group(1))
                        total_queries = int(match.group(2))
                        progress = (current_query / total_queries) * 100
                        elapsed = time.time() - start_time
                        
                        # Show progress every 25% or at completion
                        if current_query % max(1, total_queries // 4) == 0 or current_query == total_queries:
                            # Create a simple progress bar
                            bar_length = 20
                            filled_length = int(bar_length * current_query // total_queries)
                            bar = '█' * filled_length + '░' * (bar_length - filled_length)
                            print(f"📈 Progress: [{bar}] {current_query}/{total_queries} ({progress:.0f}%) - Elapsed: {elapsed:.1f}s")
                
                # Track successful queries
                if "Found" in line and "similar columns" in line:
                    query_count += 1
                    elapsed = time.time() - start_time
                    avg_time = elapsed / query_count if query_count > 0 else 0
                    print(f"✅ Query {query_count} completed - Avg time: {avg_time:.1f}s")
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        return '\n'.join(output_lines)
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        process.terminate()
        process.wait()
        raise
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        process.terminate()
        process.wait()
        raise

def main():    
    # Query parameters
    top_k_return = 50
    similarity_threshold = 0.7
    sketch_size = 1024
    device = "auto"
    max_queries = None  # Set to a number to limit queries for testing (e.g., 5)
    
    # DeepJoin integration options
    # Use DeepJoin during query processing to filter candidates before sketch comparison
    # This provides faster query processing by reducing the number of sketches to compare
    use_deepjoin_index = False  # Set to True to enable DeepJoin filtering during query processing
    deepjoin_embeddings_path = "Deepjoin/output/freyja_lake_embeddings_frequent.pkl"  # Path to DeepJoin embeddings pickle file (e.g., "Deepjoin/output/lake_embeddings.pkl")
    deepjoin_query_embeddings_path = "Deepjoin/output/freyja_queries_embeddings_frequent.pkl"  # Path to DeepJoin query embeddings pickle file (e.g., "Deepjoin/output/query_embeddings.pkl")
    deepjoin_index_path = None  # Path to DeepJoin HNSW index file (optional, will create if not exists)
    deepjoin_scale = 1.0  # Scale factor for DeepJoin dataset (0.0-1.0)
    deepjoin_encoder = "sherlock"  # DeepJoin encoder type ("sherlock" or "sato")
    deepjoin_candidate_limit = 100  # Number of candidates from DeepJoin index (N parameter)
    deepjoin_top_k = 500  # Number of top results from DeepJoin (for candidate filtering)
    deepjoin_threshold = 0.6  # DeepJoin similarity threshold for verification
    
    # Progress display options
    progress_mode = "detailed"
    
    # Configuration
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    sketches_dir = f"offline_data/sketches_k{sketch_size}"
    query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    
    # Generate configuration-based output directory name
    deepjoin_suffix = f"_deepjoin_N{deepjoin_candidate_limit}_K{deepjoin_top_k}_T{deepjoin_threshold}" if use_deepjoin_index else ""
    output_dir = f"query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}{deepjoin_suffix}"
    
    # Print DeepJoin status
    print("🔧 Configuration:")
    print(f"  - Sketch size: {sketch_size}")
    print(f"  - Similarity threshold: {similarity_threshold}")
    print(f"  - Top-k return: {top_k_return}")
    print(f"  - Device: {device}")
    if use_deepjoin_index:
        print(f"  - DeepJoin integration: ENABLED")
        print(f"    * Embeddings path: {deepjoin_embeddings_path}")
        print(f"    * Query embeddings path: {deepjoin_query_embeddings_path}")
        print(f"    * Index path: {deepjoin_index_path}")
        print(f"    * Scale: {deepjoin_scale}")
        print(f"    * Encoder: {deepjoin_encoder}")
        print(f"    * Candidate limit: {deepjoin_candidate_limit}")
        print(f"    * Top-k: {deepjoin_top_k}")
        print(f"    * Threshold: {deepjoin_threshold}")
    else:
        print(f"  - DeepJoin integration: DISABLED (standard sketch processing)")
    print()
    
    # Clean up previous query data
    # cleanup_query_data(output_dir)
    
    # Optional embeddings directory (for query sketch building)
    embeddings_dir = "offline_data/embeddings"
    
    # Check if required files exist
    if not Path(datalake_dir).exists():
        print(f"Error: Datalake directory not found: {datalake_dir}")
        print("Please run offline processing first:")
        print("  python run_offline_embeddings_parallel.py")
        print("  python run_offline_sketch_parallel.py")
        return 1
    
    if not Path(sketches_dir).exists():
        print(f"Error: Sketches directory not found: {sketches_dir}")
        print("Please run sketch building first:")
        print("  python run_offline_sketch_parallel.py")
        return 1
    
    if not Path(query_file).exists():
        print(f"Error: Query file not found: {query_file}")
        return 1
    
    # Check DeepJoin files if integration is enabled
    if use_deepjoin_index:
        if deepjoin_embeddings_path and not Path(deepjoin_embeddings_path).exists():
            print(f"Error: DeepJoin embeddings file not found: {deepjoin_embeddings_path}")
            print("Please generate DeepJoin embeddings first:")
            print("  python Deepjoin/run_deepjoin.py --mode build_index")
            return 1
        
        if deepjoin_query_embeddings_path and not Path(deepjoin_query_embeddings_path).exists():
            print(f"Error: DeepJoin query embeddings file not found: {deepjoin_query_embeddings_path}")
            print("Please generate DeepJoin query embeddings first")
            return 1
        
        if deepjoin_index_path and not Path(deepjoin_index_path).exists():
            print(f"Warning: DeepJoin index file not found: {deepjoin_index_path}")
            print("Index will be created automatically during processing")
    
    # Build command
    cmd = f"""python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold {similarity_threshold} \\
    --sketch-size {sketch_size} \\
    --device "{device}" \\
    --embeddings-dir "{embeddings_dir}" """
    
    # Add DeepJoin parameters if enabled
    if use_deepjoin_index:
        cmd += f" \\\n    --use-deepjoin-index"
        if deepjoin_embeddings_path:
            cmd += f" \\\n    --deepjoin-embeddings-path \"{deepjoin_embeddings_path}\""
        if deepjoin_query_embeddings_path:
            cmd += f" \\\n    --deepjoin-query-embeddings-path \"{deepjoin_query_embeddings_path}\""
        if deepjoin_index_path:
            cmd += f" \\\n    --deepjoin-index-path \"{deepjoin_index_path}\""
        cmd += f" \\\n    --deepjoin-scale {deepjoin_scale}"
        cmd += f" \\\n    --deepjoin-encoder {deepjoin_encoder}"
        cmd += f" \\\n    --deepjoin-candidate-limit {deepjoin_candidate_limit}"
        cmd += f" \\\n    --deepjoin-top-k {deepjoin_top_k}"
        cmd += f" \\\n    --deepjoin-threshold {deepjoin_threshold}"
    
    # Add max-queries parameter if specified
    if max_queries is not None:
        cmd += f" --max-queries {max_queries}"
    
    # Execute command with progress tracking
    try:
        if progress_mode == "simple":
            print("Running query processing...")
            print(f"Command: {cmd}")
            print()
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout
            print(output)
        else:
            output = run_with_progress(cmd, max_queries)
        
        # Parse statistics from output
        stats = parse_query_stats(output)
        
        if stats:
            print("\n" + "="*60)
            print("QUERY PROCESSING TIMING SUMMARY")
            print("="*60)
            
            if 'total_queries' in stats:
                print(f"Total queries processed: {stats['total_queries']}")
            
            if 'successful_queries' in stats:
                print(f"Successful queries: {stats['successful_queries']}")
            
            if 'failed_queries' in stats:
                print(f"Failed queries: {stats['failed_queries']}")
            
            if 'total_processing_time' in stats:
                print(f"Total processing time: {stats['total_processing_time']:.2f} seconds")
            
            if 'avg_time_per_query' in stats:
                print(f"Average time per query: {stats['avg_time_per_query']:.2f} seconds")
            
            if 'total_candidates_found' in stats:
                print(f"Total candidates found: {stats['total_candidates_found']}")
            
            if 'avg_candidates_per_query' in stats:
                print(f"Average candidates per query: {stats['avg_candidates_per_query']:.2f}")
            
            if 'total_high_quality_candidates' in stats:
                print(f"Total high-quality candidates: {stats['total_high_quality_candidates']}")
            
            if 'avg_high_quality_candidates' in stats:
                print(f"Average high-quality candidates per query: {stats['avg_high_quality_candidates']:.2f}")
            
            # DeepJoin statistics
            if 'deepjoin_queries_processed' in stats and stats['deepjoin_queries_processed'] > 0:
                print(f"\nDeepJoin Statistics:")
                print(f"  DeepJoin queries processed: {stats['deepjoin_queries_processed']}")
                print(f"  Total DeepJoin candidates: {stats['total_deepjoin_candidates']}")
                print(f"  Average DeepJoin candidates per query: {stats['avg_deepjoin_candidates']:.2f}")
            elif 'deepjoin_queries_processed' in stats:
                print(f"\nDeepJoin integration: Not used")
            
            # Calculate additional metrics
            if 'total_processing_time' in stats and 'successful_queries' in stats and stats['successful_queries'] > 0:
                calculated_avg = stats['total_processing_time'] / stats['successful_queries']
                print(f"Calculated average time per query: {calculated_avg:.2f} seconds")
            
            print("="*60)
            
            # Save timing statistics
            save_timing_stats(stats, output_dir)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error running query processing: {e}")
        print(f"Error output: {e.stderr}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
