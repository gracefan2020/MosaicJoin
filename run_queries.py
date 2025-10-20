#!/usr/bin/env python3
"""
Query Processing Runner Script
Simple script to run query processing with common configurations.
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
    print("🚀 Starting query processing...")
    print(f"Command: {cmd}")
    print()
    
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
    print("🚀 Starting query processing with timing analysis...")
    
    # Query parameters
    top_k_return = 50
    similarity_threshold = 0.7
    sketch_size = 1024
    device = "auto"
    max_queries = None  # Set to a number to limit queries for testing (e.g., 5)
    
    # Progress display options
    progress_mode = "detailed"
    
    # Configuration
    datalake_dir = "datasets/freyja-semantic-join/datalake"
    sketches_dir = f"offline_data/sketches_k{sketch_size}"
    query_file = "datasets/freyja-semantic-join/freyja_query_columns.csv"
    
    # Generate configuration-based output directory name
    output_dir = f"query_results_k{sketch_size}_t{similarity_threshold}_top{top_k_return}"
    
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
    
    # Build command
    cmd = f"""python run_query_processing.py "{datalake_dir}" "{sketches_dir}" "{query_file}" \\
    --output-dir "{output_dir}" \\
    --top-k-return {top_k_return} \\
    --similarity-threshold {similarity_threshold} \\
    --sketch-size {sketch_size} \\
    --device "{device}" \\
    --embeddings-dir "{embeddings_dir}" """
    
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
