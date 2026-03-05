"""
Offline Sketch Module

Builds k-representative sketches from pre-computed embeddings. Selection methods:
farthest_point (FPS, default), random, first_k, kmeans, k_closest. Consolidate
merges individual .pkl sketches into one .npy + index for fast bulk loading at
query time. Expects embeddings at {table}/{column}_{index}.pkl.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

# =============================================================================
# Config & Data Types
# =============================================================================

@dataclass
class SketchConfig:
    """Configuration for sketch creation: k representatives, selection method, paths."""
    sketch_size: int = 1024
    selection_method: str = "farthest_point"
    skip_empty_columns: bool = True
    output_dir: str = "offline_sketches"
    save_metadata: bool = False
    save_representative_names: bool = False


@dataclass
class SketchStats:
    """Statistics for sketch building."""
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
    """Compact semantic signature for a column: k representative vectors, centroid, optional names."""
    representative_vectors: np.ndarray
    representative_ids: List[int]
    distances_to_origin: np.ndarray
    embedding_dim: int
    k: int
    centroid: np.ndarray
    representative_names: Optional[List[str]] = None


@dataclass
class SketchMetadata:
    """Per-column metadata (table, column, counts, timing, file path) for a processed sketch."""
    table_name: str
    column_name: str
    column_index: int
    total_embeddings: int
    sketch_k: int
    processing_time: float
    file_path: str

# =============================================================================
# K-center Sampling (AKA Farthest Point Sampling)
# =============================================================================

def k_center_sampling(X: np.ndarray, k: int) -> np.ndarray:
    """Select k diverse points via farthest-point sampling (greedy k-center approximation).
    Start near centroid, then iteratively add the point farthest from the current set.
    Returns indices into X."""
    n, k = len(X), min(k, len(X))
    if k == 0:
        return np.array([], dtype=int)
    centroid = np.mean(X, axis=0)
    first_idx = int(np.argmin(np.linalg.norm(X - centroid, axis=1)))
    selected = [first_idx]
    min_dists = np.linalg.norm(X - X[first_idx], axis=1)
    min_dists[first_idx] = -np.inf
    for _ in range(k - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, np.linalg.norm(X - X[next_idx], axis=1))
        min_dists[next_idx] = -np.inf
    return np.array(selected, dtype=int)


farthest_point_sampling = k_center_sampling


def load_offline_sketch(table_name: str, column_name: str, column_index: int,
                        sketches_dir: Path) -> Optional[SemanticSketch]:
    """Load a single sketch from table/column_index.pkl."""
    try:
        path = Path(sketches_dir) / table_name / f"{column_name}_{column_index}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            d = pickle.load(f)
        v = d["representative_vectors"]
        if len(v) == 0:
            return None
        return SemanticSketch(
            representative_vectors=v,
            representative_ids=d.get("representative_ids", list(range(len(v)))),
            distances_to_origin=np.linalg.norm(v, axis=1),
            embedding_dim=d["embedding_dim"],
            k=d["k"],
            centroid=d.get("centroid", v.mean(axis=0)),
            representative_names=d.get("representative_names"),
        )
    except Exception:
        return None


# =============================================================================
# Offline Sketch Builder
# =============================================================================

class OfflineSketchBuilder:
    """Creates semantic sketches for all columns in an embeddings directory.
    Each sketch selects k representative vectors via FPS, random, kmeans, or k_closest."""

    def __init__(self, config: SketchConfig):
        self.config = config
        self.stats = SketchStats()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("OfflineSketchBuilder")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        self.processed_columns: Set[Tuple[str, str, int]] = set()

    def build_sketches_for_embeddings(self, embeddings_dir: Path,
                                      tables_to_process: Optional[List[str]] = None,
                                      datalake_dir: Optional[Path] = None) -> SketchStats:
        """Build sketches for all embedding files. Optionally load values from datalake for names."""
        all_columns = self.discover_embedding_files(embeddings_dir, tables_to_process)

        for table_name, column_name, column_index in tqdm(all_columns, desc="Creating sketches"):
            key = (table_name, column_name, column_index)
            if key in self.processed_columns:
                continue
            result = self.build_sketch_for_column(embeddings_dir, table_name, column_name, column_index, datalake_dir)
            if result is not None:
                sketch, metadata = result
                self.save_sketch_and_metadata(table_name, column_name, column_index, sketch, metadata)
                self.stats.processed_columns += 1
                self.stats.total_embeddings_processed += metadata.total_embeddings
                self.stats.total_sketches_generated += 1
                self.stats.total_processing_time += metadata.processing_time
                self.processed_columns.add(key)
            else:
                self.stats.skipped_columns += 1

        if self.config.save_metadata:
            (self.output_dir / "build_stats.json").write_text(json.dumps(asdict(self.stats), indent=2))
        return self.stats

    def discover_embedding_files(self, embeddings_dir: Path,
                                tables_to_process: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
        """Scan embeddings dir for (table, column, index) from .pkl filenames (column_N.pkl)."""
        columns = []
        for table_dir in embeddings_dir.iterdir():
            if not table_dir.is_dir() or (tables_to_process and table_dir.name not in tables_to_process):
                continue
            for f in table_dir.glob("*.pkl"):
                if "_metadata" in f.name:
                    continue
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    columns.append((table_dir.name, parts[0], int(parts[1])))
        self.stats.total_tables = len(set(c[0] for c in columns))
        self.stats.total_columns = len(columns)
        return columns

    def build_sketch_for_column(self, embeddings_dir: Path, table_name: str, column_name: str,
                                column_index: int,
                                datalake_dir: Optional[Path] = None) -> Optional[Tuple[SemanticSketch, SketchMetadata]]:
        """Load embeddings from .pkl, select k representatives using configured method, return sketch + metadata."""
        start = time.time()
        try:
            embeddings = self.load_column_embeddings(embeddings_dir, table_name, column_name, column_index)
            if embeddings is None or len(embeddings) == 0:
                return None

            values = None
            if datalake_dir and self.config.save_representative_names:
                values = self.load_column_values(datalake_dir, table_name, column_name)

            centroid = np.mean(embeddings, axis=0)
            k = self.config.sketch_size
            method = self.config.selection_method

            if k <= 0 or k >= len(embeddings):
                idx_sorted = np.arange(len(embeddings))
            else:
                k = min(k, len(embeddings))
                if method == "farthest_point":
                    idx_sorted = k_center_sampling(embeddings, k)
                elif method == "random":
                    idx_sorted = np.random.choice(len(embeddings), k, replace=False)
                elif method == "first_k":
                    idx_sorted = np.arange(k)
                elif method == "kmeans":
                    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(embeddings)
                    idx_sorted, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
                    idx_sorted = np.array(idx_sorted)

            vecs = embeddings[idx_sorted]
            names = [values[i] for i in idx_sorted] if values is not None and len(values) == len(embeddings) else None

            sketch = SemanticSketch(
                representative_vectors=vecs,
                representative_ids=idx_sorted.tolist(),
                distances_to_origin=np.linalg.norm(vecs, axis=1),
                embedding_dim=embeddings.shape[1],
                k=len(idx_sorted),
                centroid=centroid,
                representative_names=names
            )
            metadata = SketchMetadata(
                table_name=table_name, column_name=column_name, column_index=column_index,
                total_embeddings=len(embeddings), sketch_k=sketch.k,
                processing_time=time.time() - start,
                file_path=str(embeddings_dir / table_name / f"{column_name}_{column_index}.pkl")
            )
            return sketch, metadata
        except Exception as e:
            self.logger.error(f"Error building sketch for {table_name}.{column_name}: {e}")
            self.stats.failed_columns += 1
            return None

    def load_column_embeddings(self, embeddings_dir: Path, table_name: str,
                              column_name: str, column_index: int) -> Optional[np.ndarray]:
        """Load embeddings array from table/column_index.pkl (expects 'embeddings' key)."""
        try:
            path = embeddings_dir / table_name / f"{column_name}_{column_index}.pkl"
            if not path.exists():
                return None
            with open(path, "rb") as f:
                return pickle.load(f)["embeddings"]
        except Exception:
            return None

    def load_column_values(self, datalake_dir: Path, table_name: str, column_name: str) -> Optional[List[str]]:
        """Load column values from CSV for representative names."""
        try:
            df = pd.read_csv(datalake_dir / f"{table_name}.csv")
            if column_name not in df.columns:
                return None
            vals = df[column_name].dropna().astype(str).tolist()
            return [v.strip().lower() for v in vals if v.strip()]
        except Exception:
            return None

    def save_sketch_and_metadata(self, table_name: str, column_name: str, column_index: int,
                                 sketch: SemanticSketch, metadata: SketchMetadata) -> None:
        """Write .pkl sketch and optional .json metadata."""
        try:
            (self.output_dir / table_name).mkdir(exist_ok=True)
            data = {
                "representative_vectors": sketch.representative_vectors,
                "representative_ids": sketch.representative_ids,
                "embedding_dim": sketch.embedding_dim,
                "k": sketch.k,
                "centroid": sketch.centroid,
            }
            if self.config.save_representative_names and sketch.representative_names:
                data["representative_names"] = sketch.representative_names
            with open(self.output_dir / table_name / f"{column_name}_{column_index}.pkl", "wb") as f:
                pickle.dump(data, f)
            if self.config.save_metadata:
                (self.output_dir / table_name / f"{column_name}_{column_index}_metadata.json").write_text(
                    json.dumps(asdict(metadata), indent=2))
        except Exception as e:
            self.logger.error(f"Error saving {table_name}.{column_name}: {e}")

# =============================================================================
# Consolidated Store (memory-mapped bulk load)
# =============================================================================

class ConsolidatedSketchStore:
    """Memory-mapped consolidated sketch storage. Merges individual .pkl sketches into one .npy + index
    for fast bulk loading at query time without reopening many files."""

    def __init__(self, store_path: Path, mode: str = "r"):
        self.store_path = Path(store_path)
        self.vectors_file = self.store_path / "sketches_consolidated.npy"
        self.index_file = self.store_path / "sketches_index.pkl"
        self.centroids_file = self.store_path / "sketches_centroids.npy"
        self._vectors: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._index: Dict[Tuple[str, str, int], Dict] = {}
        self._loaded = False

    def load(self, mmap_mode: Optional[str] = "r") -> None:
        """Load index (pkl) and vectors/centroids (npy). mmap_mode='r' for read-only without full load."""
        if not self.vectors_file.exists() or not self.index_file.exists():
            raise FileNotFoundError(f"Consolidated store not found at {self.store_path}")
        with open(self.index_file, "rb") as f:
            self._index = pickle.load(f)
        self._vectors = np.load(self.vectors_file, mmap_mode=mmap_mode)
        if self.centroids_file.exists():
            self._centroids = np.load(self.centroids_file, mmap_mode=mmap_mode)
        self._loaded = True

    def get_sketch(self, table_name: str, column_name: str, column_index: int) -> Optional[SemanticSketch]:
        """Retrieve SemanticSketch by (table, column, index); slice vectors from consolidated array."""
        if not self._loaded:
            raise RuntimeError("Store not loaded. Call load() first.")
        key = (table_name, column_name, column_index)
        if key not in self._index:
            return None
        m = self._index[key]
        vecs = np.array(self._vectors[m["start_row"]:m["end_row"]])
        centroid = np.array(self._centroids[m["centroid_idx"]]) if self._centroids is not None else vecs.mean(axis=0)
        return SemanticSketch(
            representative_vectors=vecs,
            representative_ids=m.get("representative_ids", list(range(len(vecs)))),
            distances_to_origin=np.linalg.norm(vecs, axis=1),
            embedding_dim=m["embedding_dim"],
            k=m["k"],
            centroid=centroid,
            representative_names=m.get("representative_names")
        )

    def get_all_vectors_matrix(self) -> np.ndarray:
        """Return full consolidated vectors matrix (all sketches stacked)."""
        if not self._loaded:
            raise RuntimeError("Store not loaded.")
        return self._vectors

    def get_index(self) -> Dict:
        """Return index mapping (table,col,idx) -> {start_row, end_row, k, ...}."""
        if not self._loaded:
            raise RuntimeError("Store not loaded.")
        return self._index

    def keys(self):
        return self._index.keys() if self._loaded else iter(())

    def __len__(self):
        return len(self._index) if self._loaded else 0

    def __contains__(self, key):
        return key in self._index if self._loaded else False

    @classmethod
    def consolidate_from_directory(cls, sketches_dir: Path, output_dir: Path,
                                   sketch_size: Optional[int] = None,
                                   show_progress: bool = True) -> "ConsolidatedSketchStore":
        """Two-pass: (1) scan all .pkl to compute total vectors; (2) concatenate into .npy + index .pkl."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sketch_files = []
        for table_dir in sketches_dir.iterdir():
            if not table_dir.is_dir():
                continue
            for f in table_dir.glob("*.pkl"):
                if "_metadata" in f.name:
                    continue
                parts = f.stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    sketch_files.append((table_dir.name, parts[0], int(parts[1]), f))

        # Pass 1: compute total size and embedding dimension
        total_vectors, embedding_dim = 0, None
        for tn, cn, ci, pf in (tqdm(sketch_files, desc="Scanning") if show_progress else sketch_files):
            try:
                d = pickle.load(open(pf, "rb"))
                v = d["representative_vectors"]
                if len(v) > 0:
                    embedding_dim = embedding_dim or v.shape[1]
                    k = sketch_size or v.shape[0]
                    total_vectors += k
            except Exception:
                continue

        if embedding_dim is None:
            raise ValueError("No valid sketches found")

        # Pass 2: concatenate vectors, centroids, build index
        all_vecs = np.zeros((total_vectors, embedding_dim), dtype=np.float32)
        centroids, index, current_row = [], {}, 0

        for tn, cn, ci, pf in (tqdm(sketch_files, desc="Consolidating") if show_progress else sketch_files):
            try:
                d = pickle.load(open(pf, "rb"))
                v = d["representative_vectors"]
                if len(v) == 0:
                    continue
                k = min(v.shape[0], sketch_size) if sketch_size else v.shape[0]
                if sketch_size and v.shape[0] < sketch_size:
                    padded = np.zeros((sketch_size, embedding_dim), dtype=np.float32)
                    padded[:k] = v[:k]
                    v = padded
                    k = sketch_size
                else:
                    v = v[:k]
                end_row = current_row + k
                all_vecs[current_row:end_row] = v
                c = d.get("centroid", v.mean(axis=0))
                centroids.append(c)
                index[(tn, cn, ci)] = {"start_row": current_row, "end_row": end_row, "k": k,
                                       "embedding_dim": embedding_dim, "centroid_idx": len(centroids) - 1}
                current_row = end_row
            except Exception:
                continue

        all_vecs = all_vecs[:current_row]
        centroids = np.array(centroids, dtype=np.float32)
        np.save(output_dir / "sketches_consolidated.npy", all_vecs)
        np.save(output_dir / "sketches_centroids.npy", centroids)
        with open(output_dir / "sketches_index.pkl", "wb") as f:
            pickle.dump(index, f)
        return cls(output_dir)

# =============================================================================
# CLI: build (embeddings → sketches) | consolidate (sketches → .npy + index)
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description="Build or consolidate offline sketches")
    sub = p.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build sketches from embeddings")
    build_p.add_argument("embeddings_dir", help="Embeddings directory")
    build_p.add_argument("--output-dir", default="offline_sketches")
    build_p.add_argument("--sketch-size", type=int, default=64)
    build_p.add_argument("--selection-method", default="farthest_point",
                         choices=["farthest_point", "random", "first_k", "kmeans", "k_closest"])
    build_p.add_argument("--tables", nargs="*")
    build_p.add_argument("--datalake-dir")
    build_p.add_argument("--consolidate", action="store_true", default=True,
                         help="Consolidate after build (default)")
    build_p.add_argument("--no-consolidate", dest="consolidate", action="store_false")

    cons_p = sub.add_parser("consolidate", help="Merge sketches into consolidated store")
    cons_p.add_argument("sketches_dir", help="Directory with sketch .pkl files")
    cons_p.add_argument("--output-dir", help="Output dir (default: same as sketches_dir)")
    cons_p.add_argument("--remove-originals", action="store_true", help="Remove original folders after consolidate")

    args = p.parse_args()

    if args.command == "consolidate":
        # Merge individual .pkl sketches into sketches_consolidated.npy + sketches_index.pkl
        import shutil
        sketches_dir = Path(args.sketches_dir)
        output_dir = Path(args.output_dir) if args.output_dir else sketches_dir
        if not sketches_dir.exists():
            print(f"Error: {sketches_dir} does not exist")
            return 1
        ConsolidatedSketchStore.consolidate_from_directory(sketches_dir, output_dir)
        if args.remove_originals:
            for d in sketches_dir.iterdir():
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
        print(f"Consolidated → {output_dir}")
        return 0

    # build: create per-column sketches from embeddings, optionally consolidate
    emb_path = Path(args.embeddings_dir)
    if not emb_path.exists():
        print(f"Error: {emb_path} does not exist")
        return 1
    config = SketchConfig(sketch_size=args.sketch_size, selection_method=args.selection_method, output_dir=args.output_dir)
    builder = OfflineSketchBuilder(config)
    datalake = Path(args.datalake_dir) if args.datalake_dir else None
    stats = builder.build_sketches_for_embeddings(emb_path, args.tables, datalake)
    if args.consolidate:
        ConsolidatedSketchStore.consolidate_from_directory(Path(args.output_dir), Path(args.output_dir))
        print(f"Built {stats.processed_columns} sketches, consolidated → {args.output_dir}")
    else:
        print(f"Built {stats.processed_columns} sketches → {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
