"""
Offline Embedding Module

Builds and saves MPNet embeddings for all columns in a datalake.
This module handles the first stage of the semantic join pipeline.

Key features:
- Sequential processing of tables and columns
- Memory-efficient processing for large datalakes
- Progress tracking and resumable processing
- Direct processing from CSV files
- Configurable batch sizes and device selection
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
# Configuration Classes
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for offline embedding building."""
    # Processing options
    skip_empty_columns: bool = True

    # Output options
    output_dir: str = "offline_embeddings"
    save_metadata: bool = True

    # Device settings
    device: str = "auto"  # auto | cpu | cuda | mps

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
# Core Classes
# =============================================================================

class MPNetEmbedder:
    """MPNet embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = 768  # MPNet base v2 dimension
        self.device = device
        self.model_name = model_name
        
        # Timing statistics
        self.timing_stats = {
            'embed_calls': 0,
            'total_embed_time': 0.0,
            'total_tokens': 0,
            'avg_time_per_token': 0.0,
            'avg_time_per_call': 0.0,
            'min_call_time': float('inf'),
            'max_call_time': 0.0
        }
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using MPNet."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        # Start timing
        start_time = time.time()
        
        # Get embeddings without normalization to enable meaningful distance-based selection
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        embeddings = embeddings.astype(np.float32)
        
        # Check if we got unit-normalized vectors despite requesting non-normalized
        norms = np.linalg.norm(embeddings, axis=1)
        if np.allclose(norms, 1.0, atol=1e-3):
            # Model still outputs unit vectors, use token-level pooling to get non-normalized embeddings
            with torch.no_grad():
                # Get token embeddings (may return a list; convert to tensor)
                token_embeddings = self.model.encode(
                    texts,
                    output_value='token_embeddings',
                    convert_to_tensor=True,
                    normalize_embeddings=False
                )  # shape: (batch, seq_len, hidden)
                if isinstance(token_embeddings, (list, tuple)):
                    # Pad variable-length sequences to the same length, then stack
                    elems = []
                    # Infer hidden size and device
                    sample = token_embeddings[0]
                    sample_tensor = sample if isinstance(sample, torch.Tensor) else torch.tensor(sample)
                    hidden = sample_tensor.size(-1)
                    device = sample_tensor.device
                    dtype = sample_tensor.dtype
                    max_len = max((
                        (t.size(0) if isinstance(t, torch.Tensor) else torch.tensor(t).size(0))
                        for t in token_embeddings
                    ))
                    for t in token_embeddings:
                        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=dtype)
                        seq_len = t_tensor.size(0)
                        if seq_len < max_len:
                            pad = torch.zeros((max_len - seq_len, hidden), dtype=dtype, device=device)
                            t_tensor = torch.cat([t_tensor, pad], dim=0)
                        elems.append(t_tensor)
                    token_embeddings = torch.stack(elems, dim=0)
                # Build a padding mask by detecting zero-padding rows
                # Any token vector with all zeros is considered padding
                token_nonzero = token_embeddings.abs().sum(dim=2) > 0  # (batch, seq_len)
                mask = token_nonzero.unsqueeze(-1).to(token_embeddings.dtype)  # (batch, seq_len, 1)
                summed = (token_embeddings * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / lengths
                embeddings = pooled.cpu().numpy().astype(np.float32)
        
        # End timing and update statistics
        end_time = time.time()
        call_time = end_time - start_time
        
        # Update timing statistics
        self.timing_stats['embed_calls'] += 1
        self.timing_stats['total_embed_time'] += call_time
        self.timing_stats['total_tokens'] += len(texts)
        self.timing_stats['avg_time_per_call'] = self.timing_stats['total_embed_time'] / self.timing_stats['embed_calls']
        self.timing_stats['avg_time_per_token'] = self.timing_stats['total_embed_time'] / max(1, self.timing_stats['total_tokens'])
        self.timing_stats['min_call_time'] = min(self.timing_stats['min_call_time'], call_time)
        self.timing_stats['max_call_time'] = max(self.timing_stats['max_call_time'], call_time)
        
        return embeddings
    
    def embed_values(self, values: List[str], column_name: str) -> np.ndarray:
        """Embed values with column context."""
        if not values:
            return np.zeros((0, self.dimension), dtype=np.float32)
        
        # Clean column name for better embedding
        clean_column_name = str(column_name).strip().lower().replace('_', ' ').replace('-', ' ')
        
        # Combine column name and value for each value
        contextual_values = [f"{clean_column_name}: {str(v).strip() or 'unknown'}" for v in values]
        
        return self.embed_texts(contextual_values)
    
    def get_timing_stats(self) -> Dict[str, Any]:
        """Get current timing statistics."""
        return self.timing_stats.copy()

class OfflineEmbeddingBuilder:
    """Builds MPNet embeddings for all columns in a datalake offline."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.stats = EmbeddingStats()
        self.embedder = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.processed_columns: Set[Tuple[str, str, int]] = set()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the embedding builder."""
        logger = logging.getLogger("OfflineEmbeddingBuilder")
        logger.setLevel(logging.INFO)

        # Create file handler
        log_file = self.output_dir / "embedding_building.log"
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

    def build_embeddings_for_datalake(self, datalake_dir: Path, tables_to_process: Optional[List[str]] = None) -> EmbeddingStats:
        """Build embeddings for all columns in a datalake."""

        # Initialize embedder
        self.embedder = create_mpnet_embedder(self.config.device)

        # Discover all columns
        all_columns = self._discover_columns(datalake_dir, tables_to_process)

        # Process columns with tqdm progress bar
        for table_name, column_name, column_index in tqdm(all_columns, desc="Building embeddings"):
            # Check if already processed
            column_key = (table_name, column_name, column_index)
            if column_key in self.processed_columns:
                continue

            # Build embedding
            result = self._build_embedding_for_column(datalake_dir, table_name, column_name, column_index)

            if result is not None:
                embedding, metadata = result

                # Save embedding and metadata
                self._save_embedding_and_metadata(table_name, column_name, column_index, embedding, metadata)

                # Update stats
                self.stats.processed_columns += 1
                self.stats.total_values_processed += metadata.total_values
                self.stats.total_embeddings_generated += metadata.processed_values
                self.stats.total_processing_time += metadata.processing_time

                # Mark as processed
                self.processed_columns.add(column_key)
            else:
                self.stats.skipped_columns += 1

        # Save final statistics
        self._save_final_stats()

        return self.stats

    def _discover_columns(self, datalake_dir: Path, tables_to_process: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
        """Discover all columns in the datalake."""
        columns = []

        self.logger.info(f"Discovering columns in datalake: {datalake_dir}")

        csv_files = list(datalake_dir.glob("*.csv"))
        if tables_to_process:
            # Filter to only specified tables
            # Handle both .csv and non-.csv table names
            table_stems = set()
            for table in tables_to_process:
                if table.endswith('.csv'):
                    table_stems.add(table[:-4])  # Remove .csv extension
                else:
                    table_stems.add(table)
            
            filtered_csv_files = [f for f in csv_files if f.stem in table_stems]
            filtered_table_names = [f.stem for f in filtered_csv_files]
            self.logger.info(f"Filtering to process tables: {sorted(filtered_table_names)}")
            csv_files = filtered_csv_files

        self.logger.info(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                # Read just the header to get column names
                df_header = pd.read_csv(csv_file, nrows=0)
                table_name = csv_file.stem

                for col_idx, col_name in enumerate(df_header.columns):
                    columns.append((table_name, col_name, col_idx))

            except Exception as e:
                self.logger.warning(f"Could not read header from {csv_file}: {e}")
                continue

        self.stats.total_tables = len(set(table for table, _, _ in columns))
        self.stats.total_columns = len(columns)

        self.logger.info(f"Discovered {len(columns)} columns across {self.stats.total_tables} tables")
        return columns


    def _build_embedding_for_column(self, datalake_dir: Path, table_name: str,
                                   column_name: str, column_index: int) -> Optional[Tuple[np.ndarray, ColumnEmbeddingMetadata]]:
        """Build embedding for a single column."""
        start_time = time.time()

        try:
            # Load column values
            values = self._load_column_values(datalake_dir, table_name, column_name)
            if values is None:
                return None

            # Check if should skip
            if self._should_skip_column(values, column_name):
                self.stats.skipped_columns += 1
                return None

            # No sampling: process all values
            processed_values = values

            # Generate embeddings with column context
            embeddings = self._embed_values_with_context(processed_values, column_name)

            if len(embeddings) == 0:
                return None

            # Create metadata
            processing_time = time.time() - start_time
            metadata = ColumnEmbeddingMetadata(
                table_name=table_name,
                column_name=column_name,
                column_index=column_index,
                total_values=len(values),
                unique_values=len(set(values)),
                processed_values=len(processed_values),
                embedding_dim=embeddings.shape[1],
                processing_time=processing_time,
                file_path=str(datalake_dir / f"{table_name}.csv")
            )

            return embeddings, metadata

        except Exception as e:
            self.logger.error(f"Error building embedding for {table_name}.{column_name}: {e}")
            self.stats.failed_columns += 1
            return None

    def _load_column_values(self, datalake_dir: Path, table_name: str, column_name: str) -> Optional[List[str]]:
        """Load values for a specific column."""
        try:
            csv_file = datalake_dir / f"{table_name}.csv"
            if not csv_file.exists():
                return None

            df = pd.read_csv(csv_file)
            if column_name not in df.columns:
                return None

            # Extract values and normalize
            values = df[column_name].dropna().astype(str).tolist()
            normalized_values = [self._normalize_value(v) for v in values]
            clean_values = [v for v in normalized_values if v]

            return clean_values

        except Exception as e:
            self.logger.warning(f"Could not load values from {table_name}.{column_name}: {e}")
            return None

    def _normalize_value(self, value: Any) -> str:
        """Normalize a raw cell value for embedding."""
        s = str(value) if value is not None else ""
        return s.strip().lower()

    def _should_skip_column(self, values: List[str], column_name: str) -> bool:
        """Determine if a column should be skipped."""
        if not values:
            self.logger.warning(f"Skipping column {column_name} because it has no values ({values})")
            return True

        if self.config.skip_empty_columns and len(values) == 0:
            self.logger.warning(f"Skipping column {column_name} because it has no values ({values})")
            return True

        return False

    # def _sample_values_if_needed(self, values: List[str]) -> List[str]:
    #     """Sample values if there are too many for memory efficiency."""
    #     if not self.config.enable_value_sampling:
    #         return values
    #     if len(values) <= self.config.max_values_per_column:
    #         return values
    #     # ... sampling logic ...
    #     return some_sample

    def _embed_values_with_context(self, values: List[str], column_name: str) -> np.ndarray:
        """Embed values with column context."""
        if not values:
            return np.zeros((0, self.embedder.dimension), dtype=np.float32)

        # Clean column name for better embedding
        clean_column_name = str(column_name).strip().lower().replace('_', ' ').replace('-', ' ')

        # Normalize and prepare values with column context
        contextual_values = []

        for v in values:
            # Trim only
            clean_v = self._normalize_value(v)
            if not clean_v:
                clean_v = "unknown"

            # Combine column name and value using template
            contextual_value = f"{clean_column_name}: {clean_v}"
            contextual_values.append(contextual_value)

        # Get embeddings from MPNet embedder
        embeddings = self.embedder.embed_texts(contextual_values)

        # Convert to float32 for consistency
        embeddings = embeddings.astype(np.float32)

        return embeddings

    def _save_embedding_and_metadata(self, table_name: str, column_name: str,
                                    column_index: int, embedding: np.ndarray,
                                    metadata: ColumnEmbeddingMetadata) -> None:
        """Save embedding and metadata to disk."""
        try:
            # Create table directory
            table_dir = self.output_dir / table_name
            table_dir.mkdir(exist_ok=True)

            # Save embedding as pickle file
            embedding_file = table_dir / f"{column_name}_{column_index}.pkl"
            embedding_data = {
                'embeddings': embedding,
                'embedding_dim': embedding.shape[1],
                'num_values': embedding.shape[0]
            }
            with open(embedding_file, 'wb') as f:
                pickle.dump(embedding_data, f)

            # Save metadata as JSON
            if self.config.save_metadata:
                metadata_file = table_dir / f"{column_name}_{column_index}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving embedding for {table_name}.{column_name}: {e}")


    def _save_final_stats(self) -> None:
        """Save final statistics."""
        stats_file = self.output_dir / "build_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)
        if self.stats.processed_columns > 0:
            avg_time_per_column = self.stats.total_processing_time / self.stats.processed_columns
            avg_size_per_column = self.stats.total_values_processed / self.stats.processed_columns
            self.logger.info(f"Total embedding build time: {self.stats.total_processing_time:.4f}s, Average embedding build time per column: {avg_time_per_column:.4f}s")
            self.logger.info(f"Created {self.stats.processed_columns} embeddings with an average of {int(avg_size_per_column)} values per column")

# =============================================================================
# Utility Functions
# =============================================================================

def create_mpnet_embedder(device: str = "auto") -> MPNetEmbedder:
    """Create MPNet embedder with specified device."""
    if device.lower() == "auto":
        device = "cpu"
        try:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"
    print(f"Device selected: {device}")
    return MPNetEmbedder(device=device)

def load_column_embedding(table_name: str, column_name: str, column_index: int,
                         embeddings_dir: Path) -> Optional[np.ndarray]:
    """Load a pre-built embedding from disk."""
    try:
        embedding_file = embeddings_dir / table_name / f"{column_name}_{column_index}.pkl"
        if not embedding_file.exists():
            return None

        with open(embedding_file, 'rb') as f:
            data = pickle.load(f)
        return data["embeddings"]

    except Exception as e:
        print(f"Error loading embedding for {table_name}.{column_name}: {e}")
        return None

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Build offline MPNet embeddings for datalake columns")

    parser.add_argument("datalake_dir", type=str, help="Path to datalake directory")
    parser.add_argument("--output-dir", type=str, default="offline_embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for MPNet model (auto, cpu, cuda, mps)")
    parser.add_argument("--tables", type=str, nargs="*",
                       help="Specific tables to process (default: all tables)")
    parser.add_argument("--save-metadata", action="store_true", default=True,
                       help="Save metadata for each embedding")

    args = parser.parse_args()

    # Create config
    config = EmbeddingConfig(
        device=args.device,
        save_metadata=args.save_metadata,
        output_dir=args.output_dir
    )

    # Build embeddings
    builder = OfflineEmbeddingBuilder(config)
    datalake_path = Path(args.datalake_dir)

    if not datalake_path.exists():
        print(f"Error: Datalake directory {datalake_path} does not exist")
        return 1

    stats = builder.build_embeddings_for_datalake(datalake_path, args.tables)

    print(f"\nCompleted! Processed {stats.processed_columns} columns")
    print(f"Embeddings saved to: {args.output_dir}")

    return 0

if __name__ == "__main__":
    exit(main())
