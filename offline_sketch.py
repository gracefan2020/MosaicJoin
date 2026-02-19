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
    selection_method: str = "k_closest"  # "k_closest", "farthest_point", "centroid_distance"
    skip_empty_columns: bool = True
    output_dir: str = "offline_sketches"
    save_metadata: bool = False  # Disabled by default to save disk space
    save_representative_names: bool = False  # Disabled by default - saves significant disk space


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


def farthest_point_sampling(X: np.ndarray, k: int) -> np.ndarray:
    """Select k diverse points using farthest point sampling.
    
    Algorithm:
    1. Start with the point closest to centroid (most representative)
    2. Iteratively add the point that is farthest from all selected points
    
    This ensures diverse, well-distributed representatives that cover
    the full semantic space of the column.
    
    This is a greedy approximation to the k-center problem.
    
    Time complexity: O(n * k) where n = number of embeddings
    
    Args:
        X: (n, d) array of embeddings
        k: Number of points to select
        
    Returns:
        Array of selected indices
    """
    n = len(X)
    k = min(k, n)
    
    if k == 0:
        return np.array([], dtype=int)
    
    # Start with the point closest to centroid
    centroid = np.mean(X, axis=0)
    distances_to_centroid = np.linalg.norm(X - centroid, axis=1)
    first_idx = int(np.argmin(distances_to_centroid))
    
    selected = [first_idx]
    
    # Track minimum distance to any selected point for each candidate
    min_distances = np.linalg.norm(X - X[first_idx], axis=1)
    min_distances[first_idx] = -np.inf  # Exclude already selected
    
    # Iteratively select farthest point
    for _ in range(k - 1):
        # Select point with maximum minimum distance to selected set
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        
        # Update minimum distances
        new_distances = np.linalg.norm(X - X[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[next_idx] = -np.inf  # Exclude already selected
    
    return np.array(selected, dtype=int)


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
        representative_ids = idx_sorted.tolist()  # Store actual indices for on-demand value loading
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
        """Wrapper for standalone farthest_point_sampling function."""
        return farthest_point_sampling(X, k)
    
    def _save_sketch_and_metadata(self, table_name: str, column_name: str,
                                  column_index: int, sketch: SemanticSketch,
                                  metadata: SketchMetadata) -> None:
        try:
            table_dir = self.output_dir / table_name
            table_dir.mkdir(exist_ok=True)
            
            sketch_file = table_dir / f"{column_name}_{column_index}.pkl"
            # Minimal sketch data - distances_to_origin can be recomputed from vectors
            sketch_data = {
                'representative_vectors': sketch.representative_vectors,
                'representative_ids': sketch.representative_ids,
                'embedding_dim': sketch.embedding_dim,
                'k': sketch.k,
                'centroid': sketch.centroid,
            }
            # Only save representative_names if explicitly enabled (uses significant disk space)
            if self.config.save_representative_names and sketch.representative_names is not None:
                sketch_data['representative_names'] = sketch.representative_names
            
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
        
        representative_vectors = data["representative_vectors"]
        # Compute distances_to_origin on-demand if not stored (saves disk space)
        distances_to_origin = data.get("distances_to_origin")
        if distances_to_origin is None and len(representative_vectors) > 0:
            distances_to_origin = np.linalg.norm(representative_vectors, axis=1)
        elif distances_to_origin is None:
            distances_to_origin = np.zeros(0)
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=data["representative_ids"],
            distances_to_origin=distances_to_origin,
            embedding_dim=data["embedding_dim"],
            k=data["k"],
            centroid=data["centroid"],
            representative_names=data.get("representative_names", None)
        )
    except Exception as e:
        print(f"Error loading sketch for {table_name}.{column_name}: {e}")
        return None


class ConsolidatedSketchStore:
    """
    Memory-efficient consolidated sketch storage.
    
    Instead of loading thousands of individual pickle files, this stores all sketches
    in a single memory-mapped numpy file for fast bulk loading.
    
    Structure:
    - sketches_consolidated.npy: All representative vectors concatenated (memory-mappable)
    - sketches_index.pkl: Index mapping (table, column, idx) -> (start_row, end_row, metadata)
    """
    
    def __init__(self, store_path: Path, mode: str = 'r'):
        """
        Args:
            store_path: Directory containing consolidated sketch files
            mode: 'r' for read-only, 'w' for write
        """
        self.store_path = Path(store_path)
        self.mode = mode
        self.vectors_file = self.store_path / "sketches_consolidated.npy"
        self.index_file = self.store_path / "sketches_index.pkl"
        self.centroids_file = self.store_path / "sketches_centroids.npy"
        
        self._vectors: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._index: Dict[Tuple[str, str, int], Dict] = {}
        self._loaded = False
    
    def load(self, mmap_mode: Optional[str] = 'r') -> None:
        """
        Load the consolidated sketch store.
        
        Args:
            mmap_mode: Memory-map mode ('r', 'r+', 'c', None). 
                       'r' = read-only mmap (fastest, lowest memory)
                       None = load fully into RAM (faster access but more memory)
        """
        if not self.vectors_file.exists() or not self.index_file.exists():
            raise FileNotFoundError(f"Consolidated sketch store not found at {self.store_path}")
        
        # Load index (small, always fully loaded)
        with open(self.index_file, 'rb') as f:
            self._index = pickle.load(f)
        
        # Load vectors (potentially memory-mapped)
        self._vectors = np.load(self.vectors_file, mmap_mode=mmap_mode)
        
        # Load centroids if they exist
        if self.centroids_file.exists():
            self._centroids = np.load(self.centroids_file, mmap_mode=mmap_mode)
        
        self._loaded = True
    
    def get_sketch(self, table_name: str, column_name: str, column_index: int) -> Optional[SemanticSketch]:
        """Get a sketch by key."""
        if not self._loaded:
            raise RuntimeError("Store not loaded. Call load() first.")
        
        key = (table_name, column_name, column_index)
        if key not in self._index:
            return None
        
        meta = self._index[key]
        start_row = meta['start_row']
        end_row = meta['end_row']
        centroid_idx = meta.get('centroid_idx', -1)
        
        representative_vectors = np.array(self._vectors[start_row:end_row])
        
        if self._centroids is not None and centroid_idx >= 0:
            centroid = np.array(self._centroids[centroid_idx])
        else:
            centroid = meta.get('centroid', np.mean(representative_vectors, axis=0))
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=meta.get('representative_ids', list(range(end_row - start_row))),
            distances_to_origin=np.array(meta.get('distances_to_origin', np.linalg.norm(representative_vectors, axis=1))),
            embedding_dim=meta['embedding_dim'],
            k=meta['k'],
            centroid=centroid,
            representative_names=meta.get('representative_names', None)
        )
    
    def get_all_vectors_matrix(self) -> np.ndarray:
        """Get the full vectors matrix (useful for batch operations)."""
        if not self._loaded:
            raise RuntimeError("Store not loaded. Call load() first.")
        return self._vectors
    
    def get_index(self) -> Dict[Tuple[str, str, int], Dict]:
        """Get the full index."""
        if not self._loaded:
            raise RuntimeError("Store not loaded. Call load() first.")
        return self._index
    
    def keys(self):
        """Iterate over all sketch keys."""
        if not self._loaded:
            raise RuntimeError("Store not loaded. Call load() first.")
        return self._index.keys()
    
    def __len__(self):
        return len(self._index) if self._loaded else 0
    
    def __contains__(self, key):
        return key in self._index if self._loaded else False
    
    @classmethod
    def consolidate_from_directory(cls, sketches_dir: Path, output_dir: Path, 
                                   sketch_size: int = None,
                                   show_progress: bool = True) -> 'ConsolidatedSketchStore':
        """
        Convert a directory of individual sketch pickle files to consolidated format.
        
        Args:
            sketches_dir: Directory containing table subdirs with .pkl sketch files
            output_dir: Output directory for consolidated files
            sketch_size: If set, truncate/pad all sketches to this size (for uniform matrix)
            show_progress: Show progress bar
            
        Returns:
            ConsolidatedSketchStore instance (not yet loaded)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover all sketch files
        sketch_files = []
        for table_dir in sketches_dir.iterdir():
            if not table_dir.is_dir():
                continue
            table_name = table_dir.name
            for sketch_file in table_dir.glob("*.pkl"):
                if "_metadata" in sketch_file.name:
                    continue
                try:
                    filename = sketch_file.stem
                    parts = filename.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        column_name = parts[0]
                        column_index = int(parts[1])
                        sketch_files.append((table_name, column_name, column_index, sketch_file))
                except:
                    continue
        
        print(f"Found {len(sketch_files)} sketch files to consolidate")
        
        # First pass: determine total size and embedding dim
        total_vectors = 0
        embedding_dim = None
        iterator = tqdm(sketch_files, desc="Scanning sketches") if show_progress else sketch_files
        
        for table_name, column_name, column_index, sketch_file in iterator:
            try:
                with open(sketch_file, 'rb') as f:
                    data = pickle.load(f)
                vectors = data['representative_vectors']
                if len(vectors) > 0:
                    if embedding_dim is None:
                        embedding_dim = vectors.shape[1]
                    k = sketch_size if sketch_size else vectors.shape[0]
                    total_vectors += k
            except:
                continue
        
        if embedding_dim is None:
            raise ValueError("No valid sketches found")
        
        print(f"Total vectors: {total_vectors}, Embedding dim: {embedding_dim}")
        
        # Allocate arrays
        all_vectors = np.zeros((total_vectors, embedding_dim), dtype=np.float32)
        all_centroids = []
        index = {}
        
        # Second pass: populate arrays
        current_row = 0
        iterator = tqdm(sketch_files, desc="Consolidating sketches") if show_progress else sketch_files
        
        for table_name, column_name, column_index, sketch_file in iterator:
            try:
                with open(sketch_file, 'rb') as f:
                    data = pickle.load(f)
                
                vectors = data['representative_vectors']
                if len(vectors) == 0:
                    continue
                
                k = vectors.shape[0]
                if sketch_size:
                    if k > sketch_size:
                        vectors = vectors[:sketch_size]
                        k = sketch_size
                    elif k < sketch_size:
                        # Pad with zeros
                        padded = np.zeros((sketch_size, embedding_dim), dtype=np.float32)
                        padded[:k] = vectors
                        vectors = padded
                        k = sketch_size
                
                end_row = current_row + k
                all_vectors[current_row:end_row] = vectors
                
                centroid = data.get('centroid', np.mean(vectors, axis=0))
                centroid_idx = len(all_centroids)
                all_centroids.append(centroid)
                
                key = (table_name, column_name, column_index)
                index[key] = {
                    'start_row': current_row,
                    'end_row': end_row,
                    'k': k,
                    'embedding_dim': embedding_dim,
                    'centroid_idx': centroid_idx
                }
                
                current_row = end_row
                
            except Exception as e:
                print(f"Error processing {sketch_file}: {e}")
                continue
        
        # Trim if we allocated too much
        all_vectors = all_vectors[:current_row]
        all_centroids = np.array(all_centroids, dtype=np.float32)
        
        # Save
        vectors_file = output_dir / "sketches_consolidated.npy"
        index_file = output_dir / "sketches_index.pkl"
        centroids_file = output_dir / "sketches_centroids.npy"
        
        print(f"Saving {len(index)} sketches ({all_vectors.shape[0]} vectors)...")
        np.save(vectors_file, all_vectors)
        np.save(centroids_file, all_centroids)
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        
        # Save stats
        stats = {
            'num_sketches': len(index),
            'total_vectors': all_vectors.shape[0],
            'embedding_dim': embedding_dim,
            'vectors_file_size_mb': vectors_file.stat().st_size / (1024 * 1024),
            'centroids_file_size_mb': centroids_file.stat().st_size / (1024 * 1024)
        }
        with open(output_dir / "consolidation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Consolidated store saved to {output_dir}")
        print(f"  Vectors: {stats['vectors_file_size_mb']:.1f} MB")
        print(f"  Centroids: {stats['centroids_file_size_mb']:.1f} MB")
        
        return cls(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Create offline semantic sketches from embeddings")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Build command (original functionality)
    build_parser = subparsers.add_parser('build', help='Build sketches from embeddings')
    build_parser.add_argument("embeddings_dir", type=str, help="Path to embeddings directory")
    build_parser.add_argument("--output-dir", type=str, default="offline_sketches",
                              help="Output directory for sketches")
    build_parser.add_argument("--sketch-size", type=int, default=1024,
                              help="Number of representative vectors per sketch (k)")
    build_parser.add_argument("--tables", type=str, nargs="*",
                              help="Specific tables to process (default: all tables)")
    build_parser.add_argument("--selection-method", type=str, default="k_closest",
                              choices=["k_closest", "farthest_point", "centroid_distance"],
                              help="Sketch selection method")
    build_parser.add_argument("--save-metadata", action="store_true", default=True,
                              help="Save metadata for each sketch")
    build_parser.add_argument("--save-representative-names", action="store_true", default=False,
                              help="Save representative names for each sketch")
    build_parser.add_argument("--datalake-dir", type=str, default=None,
                              help="Path to datalake directory")
    
    # Consolidate command (new functionality)
    consolidate_parser = subparsers.add_parser('consolidate', 
                                                help='Consolidate individual sketch files into a single file for faster loading')
    consolidate_parser.add_argument("sketches_dir", type=str, 
                                    help="Path to directory containing individual sketch files")
    consolidate_parser.add_argument("--output-dir", type=str,
                                    help="Output directory for consolidated files (default: same as sketches_dir)")
    consolidate_parser.add_argument("--sketch-size", type=int,
                                    help="Truncate/pad all sketches to this size for uniform matrix")
    consolidate_parser.add_argument("--remove-originals", action="store_true",
                                    help="Remove original sketch folders after consolidation to save disk space")
    
    args = parser.parse_args()
    
    if args.command == 'build' or args.command is None:
        # Handle legacy usage (no subcommand)
        if args.command is None:
            # Re-parse with positional argument
            parser = argparse.ArgumentParser(description="Create offline semantic sketches from embeddings")
            parser.add_argument("embeddings_dir", type=str, help="Path to embeddings directory")
            parser.add_argument("--output-dir", type=str, default="offline_sketches",
                                help="Output directory for sketches")
            parser.add_argument("--sketch-size", type=int, default=1024,
                                help="Number of representative vectors per sketch (k)")
            parser.add_argument("--tables", type=str, nargs="*",
                                help="Specific tables to process (default: all tables)")
            parser.add_argument("--selection-method", type=str, default="k_closest",
                                choices=["k_closest", "farthest_point", "centroid_distance"],
                                help="Sketch selection method")
            parser.add_argument("--save-metadata", action="store_true", default=True,
                                help="Save metadata for each sketch")
            parser.add_argument("--save-representative-names", action="store_true", default=False,
                                help="Save representative names for each sketch")
            parser.add_argument("--datalake-dir", type=str, default=None,
                                help="Path to datalake directory")
            args = parser.parse_args()
        
        config = SketchConfig(
            sketch_size=args.sketch_size,
            selection_method=args.selection_method,
            save_metadata=args.save_metadata,
            save_representative_names=getattr(args, 'save_representative_names', False),
            output_dir=args.output_dir
        )
        
        builder = OfflineSketchBuilder(config)
        embeddings_path = Path(args.embeddings_dir)
        
        if not embeddings_path.exists():
            print(f"Error: Embeddings directory {embeddings_path} does not exist")
            return 1
        
        datalake_path = Path(args.datalake_dir) if args.datalake_dir else None
        stats = builder.build_sketches_for_embeddings(embeddings_path, args.tables, datalake_path)
        
        print(f"\nCompleted! Processed {stats.processed_columns} columns")
        print(f"Sketches saved to: {args.output_dir}")
        
    elif args.command == 'consolidate':
        import shutil
        
        sketches_dir = Path(args.sketches_dir)
        output_dir = Path(args.output_dir) if args.output_dir else sketches_dir
        
        if not sketches_dir.exists():
            print(f"Error: Sketches directory {sketches_dir} does not exist")
            return 1
        
        print(f"Consolidating sketches from {sketches_dir}")
        print(f"Output directory: {output_dir}")
        
        store = ConsolidatedSketchStore.consolidate_from_directory(
            sketches_dir, 
            output_dir,
            sketch_size=args.sketch_size
        )
        
        print(f"\nConsolidation complete!")
        print(f"To use the consolidated store, the query processor will automatically detect it.")
        
        # Remove original sketch folders to save disk space
        print(f"\nRemoving original sketch folders to save disk space...")
        removed_count = 0
        removed_bytes = 0
        for table_dir in sketches_dir.iterdir():
            if not table_dir.is_dir():
                continue
            # Skip the consolidated files themselves
            if table_dir.name in ("sketches_consolidated.npy", "sketches_index.pkl", "sketches_centroids.npy"):
                continue
            # Calculate size before removal
            for f in table_dir.rglob("*"):
                if f.is_file():
                    removed_bytes += f.stat().st_size
            # Remove the directory
            try:
                shutil.rmtree(table_dir)
                removed_count += 1
            except Exception as e:
                print(f"  Warning: Could not remove {table_dir}: {e}")
        
        removed_mb = removed_bytes / (1024 * 1024)
        print(f"Removed {removed_count} table folders, freed ~{removed_mb:.1f} MB")
    
    return 0


if __name__ == "__main__":
    exit(main())
