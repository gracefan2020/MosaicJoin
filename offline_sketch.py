"""
Offline Sketch Creation Module

Creates semantic sketches for all columns in a datalake.
Supports multiple selection methods:
- k_closest: k vectors closest to origin (original)
- farthest_point: Farthest point sampling for diversity (NEW)
- centroid_distance: k vectors closest to centroid
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class SketchConfig:
    """Configuration for offline sketch creation."""
    sketch_size: int = 1024
    similarity_threshold: float = 0.7
    selection_method: str = "k_closest"  # "k_closest", "farthest_point", "centroid_distance"
    skip_empty_columns: bool = True
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


class OfflineSketchBuilder:
    """Creates semantic sketches for all columns in a datalake offline."""
    
    def __init__(self, config: SketchConfig):
        self.config = config
        self.stats = SketchStats()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        self.processed_columns: Set[Tuple[str, str, int]] = set()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("OfflineSketchBuilder")
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(console_handler)
        return logger
    
    def build_sketches_for_embeddings(self, embeddings_dir: Path, 
                                      tables_to_process: Optional[List[str]] = None,
                                      datalake_dir: Optional[Path] = None) -> SketchStats:
        """Build sketches for all embeddings in the embeddings directory."""
        self.logger.info(f"Starting offline sketch building with method: {self.config.selection_method}")
        
        all_columns = self._discover_embedding_files(embeddings_dir, tables_to_process)
        remaining_columns = all_columns
        
        self.logger.info(f"Processing {len(remaining_columns)} columns")
        
        for table_name, column_name, column_index in tqdm(remaining_columns, desc="Creating sketches"):
            column_key = (table_name, column_name, column_index)
            if column_key in self.processed_columns:
                continue

            result = self._build_sketch_for_column(embeddings_dir, table_name, column_name, column_index, datalake_dir)

            if result is not None:
                sketch, metadata = result
                self._save_sketch_and_metadata(table_name, column_name, column_index, sketch, metadata)
                self.stats.processed_columns += 1
                self.stats.total_embeddings_processed += metadata.total_embeddings
                self.stats.total_sketches_generated += 1
                self.stats.total_processing_time += metadata.processing_time
                self.processed_columns.add(column_key)
            else:
                self.stats.skipped_columns += 1
        
        self._save_final_stats()
        self.logger.info("Offline sketch building completed")
        return self.stats
    
    def _discover_embedding_files(self, embeddings_dir: Path, 
                                  tables_to_process: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
        columns = []
        table_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
        
        if tables_to_process:
            table_dirs = [d for d in table_dirs if d.name in tables_to_process]
        
        for table_dir in table_dirs:
            table_name = table_dir.name
            embedding_files = list(table_dir.glob("*.pkl"))
            
            for embedding_file in embedding_files:
                try:
                    filename = embedding_file.stem
                    if '_' in filename:
                        parts = filename.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            column_name = parts[0]
                            column_index = int(parts[1])
                            columns.append((table_name, column_name, column_index))
                except Exception as e:
                    self.logger.warning(f"Could not process {embedding_file}: {e}")
        
        self.stats.total_tables = len(set(t for t, _, _ in columns))
        self.stats.total_columns = len(columns)
        self.logger.info(f"Discovered {len(columns)} embedding files across {self.stats.total_tables} tables")
        return columns
    
    def _build_sketch_for_column(self, embeddings_dir: Path, table_name: str,
                                 column_name: str, column_index: int,
                                 datalake_dir: Optional[Path] = None) -> Optional[Tuple[SemanticSketch, SketchMetadata]]:
        start_time = time.time()
        
        try:
            embeddings = self._load_column_embeddings(embeddings_dir, table_name, column_name, column_index)
            if embeddings is None or len(embeddings) == 0:
                return None
            
            values = None
            if datalake_dir is not None:
                values = self._load_column_values_for_sketch(datalake_dir, table_name, column_name)
            
            sketch = self._build_semantic_sketch_from_embeddings(embeddings, values)
            
            if sketch is None:
                return None
            
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
    
    def _load_column_embeddings(self, embeddings_dir: Path, table_name: str, 
                                column_name: str, column_index: int) -> Optional[np.ndarray]:
        try:
            embedding_file = embeddings_dir / table_name / f"{column_name}_{column_index}.pkl"
            if not embedding_file.exists():
                return None
            with open(embedding_file, 'rb') as f:
                data = pickle.load(f)
            return data["embeddings"]
        except Exception as e:
            self.logger.warning(f"Could not load embeddings from {table_name}.{column_name}: {e}")
            return None
    
    def _load_column_values_for_sketch(self, datalake_dir: Path, table_name: str, 
                                       column_name: str) -> Optional[List[str]]:
        try:
            csv_file = datalake_dir / f"{table_name}.csv"
            if not csv_file.exists():
                return None
            df = pd.read_csv(csv_file)
            if column_name not in df.columns:
                return None
            values = df[column_name].dropna().astype(str).tolist()
            return [v.strip().lower() for v in values if v.strip()]
        except Exception as e:
            return None
    
    def _build_semantic_sketch_from_embeddings(self, embeddings: np.ndarray,
                                               values: Optional[List[str]] = None) -> Optional[SemanticSketch]:
        """Build a semantic sketch using the configured selection method."""
        if len(embeddings) == 0:
            return SemanticSketch(
                representative_vectors=np.zeros((0, 0)),
                representative_ids=[],
                distances_to_origin=np.zeros(0),
                embedding_dim=0,
                k=0,
                centroid=np.zeros(0),
                representative_names=None
            )
        
        centroid = np.mean(embeddings, axis=0)
        k = min(self.config.sketch_size, len(embeddings))
        method = self.config.selection_method
        
        if method == "farthest_point":
            idx_sorted = self._farthest_point_sampling(embeddings, k)
        elif method == "centroid_distance":
            centroid_distances = np.linalg.norm(embeddings - centroid, axis=1)
            idx = np.argpartition(centroid_distances, k - 1)[:k]
            idx_sorted = idx[np.argsort(centroid_distances[idx])]
        else:  # k_closest
            idx_sorted, _ = self._k_closest_to_origin(embeddings, k)
        
        representative_vectors = embeddings[idx_sorted]
        representative_ids = list(range(len(idx_sorted)))
        distances_to_origin = np.linalg.norm(representative_vectors, axis=1)
        
        representative_names = None
        if values is not None and len(values) == len(embeddings):
            representative_names = [values[i] for i in idx_sorted]
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=representative_ids,
            distances_to_origin=distances_to_origin,
            embedding_dim=embeddings.shape[1],
            k=len(idx_sorted),
            centroid=centroid,
            representative_names=representative_names
        )
    
    def _k_closest_to_origin(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k vectors closest to origin by L2 norm."""
        norms = np.linalg.norm(X, axis=1)
        k = min(k, len(norms))
        idx = np.argpartition(norms, k - 1)[:k]
        idx_sorted = idx[np.argsort(norms[idx])]
        return idx_sorted, norms[idx_sorted]
    
    def _farthest_point_sampling(self, X: np.ndarray, k: int) -> np.ndarray:
        """Select k diverse points using farthest point sampling.
        
        Algorithm:
        1. Start with the point closest to centroid (most representative)
        2. Iteratively add the point that is farthest from all selected points
        
        This ensures diverse, well-distributed representatives that cover
        the full semantic space of the column.
        
        This is a greedy approximation to the k-center problem.
        
        Time complexity: O(n * k) where n = number of embeddings
        """
        n = len(X)
        k = min(k, n)
        
        if k == 0:
            return np.array([], dtype=int)
        
        # Start with the point closest to centroid
        centroid = np.mean(X, axis=0)
        distances_to_centroid = np.linalg.norm(X - centroid, axis=1)
        first_idx = np.argmin(distances_to_centroid)
        
        selected = [first_idx]
        
        # Track minimum distance to any selected point for each candidate
        min_distances = np.linalg.norm(X - X[first_idx], axis=1)
        min_distances[first_idx] = -np.inf  # Exclude already selected
        
        # Iteratively select farthest point
        for _ in range(k - 1):
            # Select point with maximum minimum distance to selected set
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)
            
            # Update minimum distances
            new_distances = np.linalg.norm(X - X[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
            min_distances[next_idx] = -np.inf  # Exclude already selected
        
        return np.array(selected, dtype=int)
    
    def _save_sketch_and_metadata(self, table_name: str, column_name: str,
                                  column_index: int, sketch: SemanticSketch,
                                  metadata: SketchMetadata) -> None:
        try:
            table_dir = self.output_dir / table_name
            table_dir.mkdir(exist_ok=True)
            
            sketch_file = table_dir / f"{column_name}_{column_index}.pkl"
            sketch_data = {
                'representative_vectors': sketch.representative_vectors,
                'representative_ids': sketch.representative_ids,
                'distances_to_origin': sketch.distances_to_origin,
                'embedding_dim': sketch.embedding_dim,
                'k': sketch.k,
                'centroid': sketch.centroid,
                'representative_names': sketch.representative_names
            }
            with open(sketch_file, 'wb') as f:
                pickle.dump(sketch_data, f)
            
            if self.config.save_metadata:
                metadata_file = table_dir / f"{column_name}_{column_index}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"Error saving sketch for {table_name}.{column_name}: {e}")
    
    def _save_final_stats(self) -> None:
        stats_file = self.output_dir / "build_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)


def load_offline_sketch(table_name: str, column_name: str, column_index: int,
                        sketches_dir: Path) -> Optional[SemanticSketch]:
    """Load a pre-built sketch from disk."""
    try:
        sketch_file = sketches_dir / table_name / f"{column_name}_{column_index}.pkl"
        if not sketch_file.exists():
            return None
        
        with open(sketch_file, 'rb') as f:
            data = pickle.load(f)
        
        return SemanticSketch(
            representative_vectors=data["representative_vectors"],
            representative_ids=data["representative_ids"],
            distances_to_origin=data["distances_to_origin"],
            embedding_dim=data["embedding_dim"],
            k=data["k"],
            centroid=data["centroid"],
            representative_names=data.get("representative_names", None)
        )
    except Exception as e:
        print(f"Error loading sketch for {table_name}.{column_name}: {e}")
        return None


def main():
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
    parser.add_argument("--selection-method", type=str, default="k_closest",
                        choices=["k_closest", "farthest_point", "centroid_distance"],
                        help="Sketch selection method")
    parser.add_argument("--save-metadata", action="store_true", default=True,
                        help="Save metadata for each sketch")
    
    args = parser.parse_args()
    
    config = SketchConfig(
        sketch_size=args.sketch_size,
        similarity_threshold=args.similarity_threshold,
        selection_method=args.selection_method,
        save_metadata=args.save_metadata,
        output_dir=args.output_dir
    )
    
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
