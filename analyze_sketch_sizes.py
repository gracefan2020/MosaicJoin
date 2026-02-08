#!/usr/bin/env python3
"""
Analyze actual sketch sizes in a sketches directory.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np


def analyze_sketch_sizes(sketches_dir: Path):
    """Analyze sketch sizes from metadata files."""
    
    metadata_files = list(sketches_dir.rglob("*_metadata.json"))
    
    if not metadata_files:
        print(f"No metadata files found in {sketches_dir}")
        return
    
    print(f"Found {len(metadata_files)} sketch metadata files\n")
    
    sketch_sizes = []
    total_embeddings = []
    processing_times = []
    size_distribution = Counter()
    
    small_sketches = []  # Sketches smaller than requested
    
    for meta_file in metadata_files:
        with open(meta_file) as f:
            meta = json.load(f)
        
        k = meta.get("sketch_k", meta.get("k", 0))
        total = meta.get("total_embeddings", 0)
        proc_time = meta.get("processing_time", 0.0)
        table = meta.get("table_name", "unknown")
        column = meta.get("column_name", "unknown")
        
        sketch_sizes.append(k)
        total_embeddings.append(total)
        processing_times.append(proc_time)
        size_distribution[k] += 1
        
        # Track sketches that are smaller than max
        if k < max(sketch_sizes) if sketch_sizes else 0:
            small_sketches.append((table, column, k, total))
    
    # Statistics
    sizes = np.array(sketch_sizes)
    totals = np.array(total_embeddings)
    times = np.array(processing_times)
    
    print("=" * 60)
    print("SKETCH SIZE STATISTICS")
    print("=" * 60)
    print(f"Min sketch size:    {sizes.min()}")
    print(f"Max sketch size:    {sizes.max()}")
    print(f"Mean sketch size:   {sizes.mean():.1f}")
    print(f"Median sketch size: {np.median(sizes):.0f}")
    print(f"Std dev:            {sizes.std():.1f}")
    
    print("\n" + "=" * 60)
    print("ORIGINAL EMBEDDING COUNTS")
    print("=" * 60)
    print("Statistics for the total number of original embeddings (before sketching) per column:")
    print("This measures how many vectors were available in each column, i.e., the number of original values that could be sketched.")
    print(f"Min embeddings:     {totals.min()}    # The fewest original embeddings for any column")
    print(f"Max embeddings:     {totals.max()}    # The most original embeddings for any column")
    print(f"Mean embeddings:    {totals.mean():.1f} # The average number of embeddings per column")
    
    print("\n" + "=" * 60)
    print("SKETCH BUILD TIME")
    print("=" * 60)
    if times.sum() > 0:
        print(f"Min build time:     {times.min():.4f}s")
        print(f"Max build time:     {times.max():.4f}s")
        print(f"Mean build time:    {times.mean():.4f}s")
        print(f"Median build time:  {np.median(times):.4f}s")
        print(f"Total build time:   {times.sum():.2f}s ({times.sum()/60:.2f} min)")
    else:
        print("No processing time data available")
    

def main(sketches_dir):
    sketches_path = Path(sketches_dir)
    if not sketches_path.exists():
        print(f"Error: {sketches_path} does not exist")
        return 1
    
    analyze_sketch_sizes(sketches_path)
    return 0


if __name__ == "__main__":
    main("wt-experiments/wt_offline_data_embeddinggemma_no_column_names/sketches_k64")
