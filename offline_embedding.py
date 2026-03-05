"""
Offline Embedding Module

Embeds all column values in a datalake using MPNet, EmbeddingGemma, or BGE.
Outputs .pkl files per column (table/column_index.pkl). Use run_offline_embeddings_parallel
to distribute work across SLURM nodes.
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
from sentence_transformers import SentenceTransformer
import torch

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for offline embedding building."""
    skip_empty_columns: bool = True
    output_dir: str = "offline_embeddings"
    save_metadata: bool = True
    device: str = "auto"
    embedding_model: str = "mpnet"
    embedding_dim: int = 128  # For embeddinggemma MRL: 128, 256, 512, 768

@dataclass
class EmbeddingStats:
    """Statistics for embedding building process."""
    total_tables: int = 0
    total_columns: int = 0
    processed_columns: int = 0
    skipped_columns: int = 0
    failed_columns: int = 0
    total_processing_time: float = 0.0
    total_values_processed: int = 0
    total_embeddings_generated: int = 0

@dataclass
class ColumnEmbeddingMetadata:
    """Metadata for a processed column embedding."""
    table_name: str
    column_name: str
    column_index: int
    total_values: int
    unique_values: int
    processed_values: int
    embedding_dim: int
    processing_time: float
    file_path: str

# =============================================================================
# Embedder (MPNet / EmbeddingGemma / BGE)
# =============================================================================

def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device."""
    if device.lower() != "auto":
        return device
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _prepare_texts(values: List[str], column_name: str = "") -> List[str]:
    """Prepare value strings for embedding."""
    clean_col = str(column_name).strip().lower().replace('_', ' ').replace('-', ' ') if column_name else ""
    texts = [f"{str(v)}" for v in values]
    return texts


class BaseEmbedder:
    """Embedder for MPNet, EmbeddingGemma, and BGE."""

    # Model configs: name and output dim (None = use output_dim arg)
    _MODELS = {
        "mpnet": {"name": "sentence-transformers/all-mpnet-base-v2", "dim": 768},
        "embeddinggemma": {"name": "google/embeddinggemma-300m", "dim": None},  # uses output_dim
        "bge": {"name": "BAAI/bge-base-en-v1.5", "dim": 768},
        "bge384": {"name": "BAAI/bge-small-en-v1.5", "dim": 384},
    }

    def __init__(self, model: str = "mpnet", device: str = "cpu", output_dim: int = 128, mode: str = "document"):
        self.device = _resolve_device(device)
        self.mode = mode
        model_key = model.lower()
        # EmbeddingGemma uses encode_query/encode_document; others use standard encode
        if model_key in ("embeddinggemma", "gemma"):
            cfg = self._MODELS["embeddinggemma"]
            self.dimension = output_dim
            self.model = SentenceTransformer(cfg["name"], device=self.device)
            self._encode_fn = self._encode_gemma
        elif model_key == "bge":
            model_key = "bge384" if output_dim == 384 else "bge"
            cfg = self._MODELS[model_key]
            self.dimension = cfg["dim"]
            self.model = SentenceTransformer(cfg["name"], device=self.device)
            self._encode_fn = self._encode_standard
        else:
            cfg = self._MODELS.get("mpnet", self._MODELS["mpnet"])
            self.dimension = cfg["dim"]
            self.model = SentenceTransformer(cfg["name"], device=self.device)
            self._encode_fn = self._encode_standard

    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _encode_gemma(self, texts: List[str]) -> np.ndarray:
        # Query mode for query-time; document mode for offline
        enc = self.model.encode_query if self.mode == "query" else self.model.encode_document
        return enc(texts, convert_to_numpy=True, truncate_dim=self.dimension, normalize_embeddings=True)

    def embed_values(self, values: List[str], column_name: str = "", use_context: bool = False) -> np.ndarray:
        """Embed values, optionally with column context."""
        if not values:
            return np.zeros((0, self.dimension), dtype=np.float32)
        texts = _prepare_texts(values, column_name)
        return self._encode_fn(texts).astype(np.float32)

# Embedders
MPNetEmbedder = BaseEmbedder
EmbeddingGemmaEmbedder = BaseEmbedder
BgeEmbedder = BaseEmbedder


def create_mpnet_embedder(device: str = "auto") -> BaseEmbedder:
    """Create MPNet embedder."""
    return BaseEmbedder("mpnet", device=device)


def create_embedder(model: str = "mpnet", device: str = "auto", embedding_dim: int = 128,
                    mode: str = "document") -> BaseEmbedder:
    """Create an embedder for the given model."""
    return BaseEmbedder(
        model=model, device=device, output_dim=embedding_dim, mode=mode
    )


def load_column_embedding(table_name: str, column_name: str, column_index: int,
                         embeddings_dir: Path) -> Optional[np.ndarray]:
    """Load a pre-built embedding from disk."""
    try:
        path = embeddings_dir / table_name / f"{column_name}_{column_index}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)["embeddings"]
    except Exception:
        return None

# =============================================================================
# Offline Embedding Builder
# =============================================================================

class OfflineEmbeddingBuilder:
    """Scans datalake CSVs, embeds each column with use_context=True (column: value), saves as .pkl."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = EmbeddingStats()
        self.embedder: Optional[BaseEmbedder] = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("OfflineEmbeddingBuilder")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(h)
        self.processed_columns: Set[Tuple[str, str, int]] = set()

    def build_embeddings_for_datalake(self, datalake_dir: Path,
                                      tables_to_process: Optional[List[str]] = None) -> EmbeddingStats:
        """Discover columns, embed values with context, save .pkl + optional metadata per column."""
        self.embedder = create_embedder(
            model=self.config.embedding_model,
            device=self.config.device,
            embedding_dim=self.config.embedding_dim
        )
        all_columns = self._discover_columns(datalake_dir, tables_to_process)

        # Process each column: load values → embed → save
        for table_name, column_name, column_index in tqdm(all_columns, desc="Building embeddings"):
            key = (table_name, column_name, column_index)
            if key in self.processed_columns:
                continue
            result = self._build_embedding_for_column(datalake_dir, table_name, column_name, column_index)
            if result is not None:
                embedding, metadata = result
                self._save_embedding_and_metadata(table_name, column_name, column_index, embedding, metadata)
                self.stats.processed_columns += 1
                self.stats.total_values_processed += metadata.total_values
                self.stats.total_embeddings_generated += metadata.processed_values
                self.stats.total_processing_time += metadata.processing_time
                self.processed_columns.add(key)
            else:
                self.stats.skipped_columns += 1

        if self.config.save_metadata:
            (self.output_dir / "build_stats.json").write_text(json.dumps(asdict(self.stats), indent=2))
        return self.stats

    def _discover_columns(self, datalake_dir: Path,
                         tables_to_process: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
        """Discover all columns in the datalake."""
        csv_files = list(datalake_dir.glob("*.csv"))
        if tables_to_process:
            stems = {t[:-4] if t.endswith(".csv") else t for t in tables_to_process}
            csv_files = [f for f in csv_files if f.stem in stems]
        columns = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, nrows=0)
                for i, col in enumerate(df.columns):
                    columns.append((f.stem, col, i))
            except Exception as e:
                self.logger.warning(f"Could not read {f}: {e}")
        self.stats.total_tables = len(set(c[0] for c in columns))
        self.stats.total_columns = len(columns)
        return columns

    def _build_embedding_for_column(self, datalake_dir: Path, table_name: str,
                                   column_name: str, column_index: int
                                   ) -> Optional[Tuple[np.ndarray, ColumnEmbeddingMetadata]]:
        """Build embedding for a single column."""
        start = time.time()
        try:
            values = self._load_column_values(datalake_dir, table_name, column_name)
            if not values or self._should_skip_column(values):
                return None
            embeddings = self.embedder.embed_values(values, column_name, use_context=True)
            if len(embeddings) == 0:
                return None
            return embeddings, ColumnEmbeddingMetadata(
                table_name=table_name, column_name=column_name, column_index=column_index,
                total_values=len(values), unique_values=len(set(values)),
                processed_values=len(values), embedding_dim=embeddings.shape[1],
                processing_time=time.time() - start, file_path=str(datalake_dir / f"{table_name}.csv")
            )
        except Exception as e:
            self.logger.error(f"Error building {table_name}.{column_name}: {e}")
            self.stats.failed_columns += 1
            return None

    def _load_column_values(self, datalake_dir: Path, table_name: str, column_name: str) -> Optional[List[str]]:
        """Load values for a specific column."""
        try:
            path = datalake_dir / f"{table_name}.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path)
            if column_name not in df.columns:
                return None
            values = df[column_name].dropna().astype(str).tolist()
            return [v.strip().lower() for v in values if v.strip()]
        except Exception as e:
            self.logger.warning(f"Could not load {table_name}.{column_name}: {e}")
            return None

    def _should_skip_column(self, values: List[str]) -> bool:
        return (self.config.skip_empty_columns and not values) or not values

    def _save_embedding_and_metadata(self, table_name: str, column_name: str, column_index: int,
                                    embedding: np.ndarray, metadata: ColumnEmbeddingMetadata) -> None:
        """Save embedding and metadata to disk."""
        try:
            (self.output_dir / table_name).mkdir(exist_ok=True)
            pkl_path = self.output_dir / table_name / f"{column_name}_{column_index}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump({"embeddings": embedding, "embedding_dim": embedding.shape[1],
                             "num_values": embedding.shape[0]}, f)
            if self.config.save_metadata:
                meta_path = self.output_dir / table_name / f"{column_name}_{column_index}_metadata.json"
                meta_path.write_text(json.dumps(asdict(metadata), indent=2))
        except Exception as e:
            self.logger.error(f"Error saving {table_name}.{column_name}: {e}")

# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Build offline embeddings for datalake columns")
    parser.add_argument("datalake_dir", type=str, help="Path to datalake directory")
    parser.add_argument("--output-dir", type=str, default="offline_embeddings", help="Output directory")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tables", type=str, nargs="*", help="Specific tables to process")
    parser.add_argument("--save-metadata", action="store_true", default=True)
    parser.add_argument("--embedding-model", type=str, default="mpnet",
                        choices=["mpnet", "embeddinggemma", "bge"])
    parser.add_argument("--embedding-dim", type=int, default=768, choices=[128, 256, 384, 512, 768])

    args = parser.parse_args()
    config = EmbeddingConfig(
        device=args.device, save_metadata=args.save_metadata, output_dir=args.output_dir,
        embedding_model=args.embedding_model, embedding_dim=args.embedding_dim
    )
    builder = OfflineEmbeddingBuilder(config)
    datalake_path = Path(args.datalake_dir)
    if not datalake_path.exists():
        print(f"Error: Datalake directory {datalake_path} does not exist")
        return 1
    stats = builder.build_embeddings_for_datalake(datalake_path, args.tables)
    print(f"Processed {stats.processed_columns} columns → {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
