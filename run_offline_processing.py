"""
Offline Processing Script

This script handles the offline stages of the semantic join pipeline:
1. Building embeddings for all tables in the datalake
2. Creating semantic sketches from embeddings

This is the offline component that should be run once to prepare
the datalake for query processing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from offline_embedding import OfflineEmbeddingBuilder, EmbeddingConfig
from offline_sketch import OfflineSketchBuilder, SketchConfig

def main():
    """Main function for offline processing."""
    parser = argparse.ArgumentParser(description="Offline processing for semantic join pipeline")
    
    # Required arguments
    parser.add_argument("datalake_dir", type=str, help="Path to datalake directory")
    parser.add_argument("--output-dir", type=str, default="offline_data", 
                       help="Output directory for embeddings and sketches")
    
    # Embedding parameters
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for MPNet model (auto, cpu, cuda, mps)")
    parser.add_argument("--tables", type=str, nargs="*",
                       help="Specific tables to process (default: all tables)")
    
    # Sketch parameters
    parser.add_argument("--sketch-size", type=int, default=1024,
                       help="Number of representative vectors per sketch")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                       help="Similarity threshold for validation")
    parser.add_argument("--use-centered-distance", action="store_true",
                       help="Use centroid distance instead of origin distance")
    
    # Processing options
    parser.add_argument("--embeddings-only", action="store_true",
                       help="Only build embeddings, skip sketch creation")
    parser.add_argument("--sketches-only", action="store_true",
                       help="Only build sketches, skip embedding creation")
    parser.add_argument("--embeddings-dir", type=str,
                       help="Path to existing embeddings directory (for sketches-only mode)")
    
    args = parser.parse_args()
    
    # Setup paths with sketch size in directory names
    datalake_path = Path(args.datalake_dir)
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    sketches_dir = output_dir / f"sketches_k{args.sketch_size}"
    
    if not datalake_path.exists():
        print(f"Error: Datalake directory {datalake_path} does not exist")
        return 1
    
    # Create output directories
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    sketches_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Build embeddings
    if not args.sketches_only:
        print("Building embeddings...")
        
        embedding_config = EmbeddingConfig(
            device=args.device,
            output_dir=str(embeddings_dir)
        )
        
        embedding_builder = OfflineEmbeddingBuilder(embedding_config)
        embedding_stats = embedding_builder.build_embeddings_for_datalake(datalake_path, args.tables)
        
        print(f"Embeddings completed: {embedding_stats.processed_columns} columns processed")
    
    # Stage 2: Create sketches
    if not args.embeddings_only:
        print("Creating sketches...")
        
        # Use provided embeddings dir or default
        if args.embeddings_dir:
            embeddings_source_dir = Path(args.embeddings_dir)
        else:
            embeddings_source_dir = embeddings_dir
        
        if not embeddings_source_dir.exists():
            print(f"Error: Embeddings directory {embeddings_source_dir} does not exist")
            return 1
        
        sketch_config = SketchConfig(
            sketch_size=args.sketch_size,
            similarity_threshold=args.similarity_threshold,
            use_centered_distance_for_sampling=args.use_centered_distance,
            output_dir=str(sketches_dir)
        )
        
        sketch_builder = OfflineSketchBuilder(sketch_config)
        sketch_stats = sketch_builder.build_sketches_for_embeddings(embeddings_source_dir, args.tables)
        
        print(f"Sketches completed: {sketch_stats.processed_columns} columns processed")
    
    # Save configuration
    config_file = output_dir / "offline_config.json"
    config_data = {
        "datalake_dir": str(datalake_path),
        "embeddings_dir": str(embeddings_dir),
        "sketches_dir": str(sketches_dir),
        "device": args.device,
        "sketch_size": args.sketch_size,
        "similarity_threshold": args.similarity_threshold,
        "tables_processed": args.tables,
        "use_centered_distance": args.use_centered_distance
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\nOffline processing completed!")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Sketches (k={args.sketch_size}): {sketches_dir}")
    print(f"Config: {config_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
