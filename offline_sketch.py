"""
Offline Sketch Creation Module

Creates semantic sketches using k-closest embeddings to origin for all columns in a datalake.
This module handles the second stage of the semantic join pipeline.

Key features:
- Uses pre-built embeddings from offline_embedding.py
- Creates sketches using k-closest points to origin
- Memory-efficient processing for large datalakes
- Progress tracking
- Configurable sketch sizes and similarity thresholds
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class SketchConfig:
    """Configuration for offline sketch creation."""
    # Core parameters
    sketch_size: int = 1024  # k for k-closest selection
    similarity_threshold: float = 0.7  # for debugging/validation
    use_centered_distance_for_sampling: bool = False  # centroid vs origin distance
    
    # Processing options
    skip_empty_columns: bool = True
    skip_single_value_columns: bool = True
    min_values_per_column: int = 2
    
    # Output options
    output_dir: str = "offline_sketches"
    save_metadata: bool = True

@dataclass
class SketchStats:
    """Statistics for sketch creation process."""
    total_tables: int = 0
    total_columns: int = 0
    processed_columns: int = 0
    skipped_columns: int = 0
    failed_columns: int = 0
    total_processing_time: float = 0.0
    total_embeddings_processed: int = 0
    total_sketches_generated: int = 0

@dataclass
class SemanticSketch:
    """Compact semantic signature for a column."""
    representative_vectors: np.ndarray
    representative_ids: List[int]
    distances_to_origin: np.ndarray
    embedding_dim: int
    k: int
    centroid: np.ndarray
    representative_names: Optional[List[str]] = None

@dataclass
class SketchMetadata:
    """Metadata for a processed sketch."""
    table_name: str
    column_name: str
    column_index: int
    total_embeddings: int
    sketch_k: int
    processing_time: float
    file_path: str

# =============================================================================
# Core Classes
# =============================================================================

class OfflineSketchBuilder:
    """Creates semantic sketches for all columns in a datalake offline."""
    
    def __init__(self, config: SketchConfig):
        self.config = config
        self.stats = SketchStats()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.processed_columns: Set[Tuple[str, str, int]] = set()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the sketch builder."""
        logger = logging.getLogger("OfflineSketchBuilder")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / "sketch_building.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def build_sketches_for_embeddings(self, embeddings_dir: Path, tables_to_process: Optional[List[str]] = None) -> SketchStats:
        """Build sketches for all embeddings in the embeddings directory."""
        self.logger.info("Starting offline sketch building")
        
        # Discover all embedding files
        all_columns = self._discover_embedding_files(embeddings_dir, tables_to_process)
        
        # No checkpointing: always process all columns
        remaining_columns = all_columns
        
        self.logger.info(f"Processing {len(remaining_columns)} columns")
        
        # Process columns with tqdm progress bar
        for table_name, column_name, column_index in tqdm(remaining_columns, desc="Creating sketches"):
            # Check if already processed
            column_key = (table_name, column_name, column_index)
            if column_key in self.processed_columns:
                continue

            # Create sketch
            result = self._build_sketch_for_column(embeddings_dir, table_name, column_name, column_index)

            if result is not None:
                sketch, metadata = result

                # Save sketch and metadata
                self._save_sketch_and_metadata(table_name, column_name, column_index, sketch, metadata)

                # Update stats
                self.stats.processed_columns += 1
                self.stats.total_embeddings_processed += metadata.total_embeddings
                self.stats.total_sketches_generated += 1
                self.stats.total_processing_time += metadata.processing_time

                # Mark as processed
                self.processed_columns.add(column_key)
            else:
                self.stats.skipped_columns += 1
        
        # Save final statistics
        self._save_final_stats()
        
        self.logger.info("Offline sketch building completed")
        return self.stats
    
    def _discover_embedding_files(self, embeddings_dir: Path, tables_to_process: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
        """Discover all embedding files in the embeddings directory."""
        columns = []
        
        self.logger.info(f"Discovering embedding files in: {embeddings_dir}")
        
        # Find all table directories
        table_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
        if tables_to_process:
            # Filter to only specified tables
            table_dirs = [d for d in table_dirs if d.name in tables_to_process]
            self.logger.info(f"Filtering to specified tables: {tables_to_process}")
        
        self.logger.info(f"Found {len(table_dirs)} table directories")
        
        for table_dir in table_dirs:
            table_name = table_dir.name
            
            # Find all embedding files in this table directory
            embedding_files = list(table_dir.glob("*.npz"))
            
            for embedding_file in embedding_files:
                try:
                    # Extract column name and index from filename
                    # Format: column_name_column_index.npz
                    filename = embedding_file.stem
                    if '_' in filename:
                        # Find the last underscore to split column name and index
                        parts = filename.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            column_name = parts[0]
                            column_index = int(parts[1])
                            columns.append((table_name, column_name, column_index))
                        else:
                            self.logger.warning(f"Could not parse filename: {filename}")
                    else:
                        self.logger.warning(f"Could not parse filename: {filename}")
                        
                except Exception as e:
                    self.logger.warning(f"Could not process {embedding_file}: {e}")
                    continue
        
        self.stats.total_tables = len(set(table for table, _, _ in columns))
        self.stats.total_columns = len(columns)
        
        self.logger.info(f"Discovered {len(columns)} embedding files across {self.stats.total_tables} tables")
        return columns
    
    
    def _build_sketch_for_column(self, embeddings_dir: Path, table_name: str, 
                                column_name: str, column_index: int) -> Optional[Tuple[SemanticSketch, SketchMetadata]]:
        """Build semantic sketch for a single column."""
        start_time = time.time()
        
        try:
            # Load embeddings
            embeddings = self._load_column_embeddings(embeddings_dir, table_name, column_name, column_index)
            if embeddings is None:
                return None
            
            # Check if should skip
            if self._should_skip_column(embeddings):
                self.stats.skipped_columns += 1
                return None
            
            # Build semantic sketch
            sketch = self._build_semantic_sketch_from_embeddings(embeddings)
            
            if sketch is None:
                return None
            
            # Create metadata
            processing_time = time.time() - start_time
            metadata = SketchMetadata(
                table_name=table_name,
                column_name=column_name,
                column_index=column_index,
                total_embeddings=len(embeddings),
                sketch_k=sketch.k,
                processing_time=processing_time,
                file_path=str(embeddings_dir / table_name / f"{column_name}_{column_index}.npz")
            )
            
            return sketch, metadata
            
        except Exception as e:
            self.logger.error(f"Error building sketch for {table_name}.{column_name}: {e}")
            self.stats.failed_columns += 1
            return None
    
    def _load_column_embeddings(self, embeddings_dir: Path, table_name: str, column_name: str, column_index: int) -> Optional[np.ndarray]:
        """Load embeddings for a specific column."""
        try:
            embedding_file = embeddings_dir / table_name / f"{column_name}_{column_index}.npz"
            if not embedding_file.exists():
                return None
            
            data = np.load(embedding_file)
            return data["embeddings"]
            
        except Exception as e:
            self.logger.warning(f"Could not load embeddings from {table_name}.{column_name}: {e}")
            return None
    
    def _should_skip_column(self, embeddings: np.ndarray) -> bool:
        """Determine if a column should be skipped."""
        if len(embeddings) == 0:
            return True
        
        if self.config.skip_empty_columns and len(embeddings) == 0:
            return True
        
        if self.config.skip_single_value_columns and len(embeddings) == 1:
            return True
        
        if len(embeddings) < self.config.min_values_per_column:
            return True
        
        return False
    
    def _build_semantic_sketch_from_embeddings(self, embeddings: np.ndarray) -> Optional[SemanticSketch]:
        """Build a semantic sketch using k-closest points to origin."""
        if len(embeddings) == 0:
            return SemanticSketch(
                representative_vectors=np.zeros((0, 0)),
                representative_ids=[],
                distances_to_origin=np.zeros(0),
                embedding_dim=0,
                k=0,
                centroid=np.zeros(0)
            )
        
        # Calculate centroid of all embeddings
        centroid = np.mean(embeddings, axis=0)
        
        # Select representatives: origin or centroid distance
        if self.config.use_centered_distance_for_sampling:
            centroid_distances = np.linalg.norm(embeddings - centroid, axis=1)
            k = min(self.config.sketch_size, len(centroid_distances))
            idx = np.argpartition(centroid_distances, k - 1)[:k]
            idx_sorted = idx[np.argsort(centroid_distances[idx])]
            distances = centroid_distances[idx_sorted]
        else:
            closest_indices, distances = self._k_closest_to_origin(embeddings, self.config.sketch_size)
            idx_sorted = closest_indices
        
        # Extract representative vectors and calculate distances to origin
        representative_vectors = embeddings[idx_sorted]
        representative_ids = list(range(len(idx_sorted)))  # Simple ID assignment
        
        # Calculate distances to origin for all representatives
        distances_to_origin = np.linalg.norm(representative_vectors, axis=1)
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=representative_ids,
            distances_to_origin=distances,
            embedding_dim=embeddings.shape[1],
            k=len(idx_sorted),
            centroid=centroid,
            representative_names=None
        )
    
    def _k_closest_to_origin(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k vectors closest to origin by L2 norm"""
        norms = np.linalg.norm(X, axis=1)
        k = min(k, len(norms))
        idx = np.argpartition(norms, k - 1)[:k]
        idx_sorted = idx[np.argsort(norms[idx])]
        return idx_sorted, norms[idx_sorted]
    
    def _save_sketch_and_metadata(self, table_name: str, column_name: str, 
                                 column_index: int, sketch: SemanticSketch, 
                                 metadata: SketchMetadata) -> None:
        """Save sketch and metadata to disk."""
        try:
            # Create table directory
            table_dir = self.output_dir / table_name
            table_dir.mkdir(exist_ok=True)
            
            # Save sketch as NPZ file
            sketch_file = table_dir / f"{column_name}_{column_index}.npz"
            np.savez_compressed(
                sketch_file,
                representative_vectors=sketch.representative_vectors,
                representative_ids=np.array(sketch.representative_ids),
                distances_to_origin=sketch.distances_to_origin,
                embedding_dim=sketch.embedding_dim,
                k=sketch.k,
                centroid=sketch.centroid,
                representative_names=np.array(sketch.representative_names, dtype=object) 
                    if sketch.representative_names is not None else np.array([], dtype=object)
            )
            
            # Save metadata as JSON
            if self.config.save_metadata:
                metadata_file = table_dir / f"{column_name}_{column_index}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving sketch for {table_name}.{column_name}: {e}")
    
    
    def _save_final_stats(self) -> None:
        """Save final statistics."""
        stats_file = self.output_dir / "build_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)
        
        # Print summary
        self.logger.info("=== BUILD STATISTICS ===")
        self.logger.info(f"Total tables: {self.stats.total_tables}")
        self.logger.info(f"Total columns: {self.stats.total_columns}")
        self.logger.info(f"Processed columns: {self.stats.processed_columns}")
        self.logger.info(f"Skipped columns: {self.stats.skipped_columns}")
        self.logger.info(f"Failed columns: {self.stats.failed_columns}")
        self.logger.info(f"Total processing time: {self.stats.total_processing_time:.2f}s")
        self.logger.info(f"Total embeddings processed: {self.stats.total_embeddings_processed}")
        self.logger.info(f"Total sketches generated: {self.stats.total_sketches_generated}")
        
        if self.stats.processed_columns > 0:
            avg_time_per_column = self.stats.total_processing_time / self.stats.processed_columns
            self.logger.info(f"Average time per column: {avg_time_per_column:.4f}s")

# =============================================================================
# Utility Functions
# =============================================================================

def load_offline_sketch(table_name: str, column_name: str, column_index: int, 
                       sketches_dir: Path) -> Optional[SemanticSketch]:
    """Load a pre-built sketch from disk."""
    try:
        sketch_file = sketches_dir / table_name / f"{column_name}_{column_index}.npz"
        if not sketch_file.exists():
            return None
        
        data = np.load(sketch_file, allow_pickle=True)
        sketch = SemanticSketch(
            representative_vectors=data["representative_vectors"],
            representative_ids=data["representative_ids"].tolist(),
            distances_to_origin=data["distances_to_origin"],
            embedding_dim=int(data["embedding_dim"]),
            k=int(data["k"]),
            centroid=data["centroid"],
            representative_names=data["representative_names"].tolist() 
                if "representative_names" in data else None,
        )
        return sketch
        
    except Exception as e:
        print(f"Error loading sketch for {table_name}.{column_name}: {e}")
        return None

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create offline semantic sketches from embeddings")
    
    parser.add_argument("embeddings_dir", type=str, help="Path to embeddings directory")
    parser.add_argument("--output-dir", type=str, default="offline_sketches", 
                       help="Output directory for sketches")
    parser.add_argument("--sketch-size", type=int, default=1024,
                       help="Number of representative vectors per sketch (k)")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                       help="Similarity threshold for validation")
    parser.add_argument("--tables", type=str, nargs="*",
                       help="Specific tables to process (default: all tables)")
    parser.add_argument("--use-centered-distance", action="store_true",
                       help="Use centroid distance instead of origin distance")
    parser.add_argument("--save-metadata", action="store_true", default=True,
                       help="Save metadata for each sketch")
    
    args = parser.parse_args()
    
    # Create config
    config = SketchConfig(
        sketch_size=args.sketch_size,
        similarity_threshold=args.similarity_threshold,
        use_centered_distance_for_sampling=args.use_centered_distance,
        save_metadata=args.save_metadata,
        output_dir=args.output_dir
    )
    
    # Build sketches
    builder = OfflineSketchBuilder(config)
    embeddings_path = Path(args.embeddings_dir)
    
    if not embeddings_path.exists():
        print(f"Error: Embeddings directory {embeddings_path} does not exist")
        return 1
    
    stats = builder.build_sketches_for_embeddings(embeddings_path, args.tables)
    
    print(f"\nCompleted! Processed {stats.processed_columns} columns")
    print(f"Sketches saved to: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
