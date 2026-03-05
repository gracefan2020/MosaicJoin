"""
Query Time Module

Embeds query column values and compares against pre-computed datalake sketches
or full embeddings to find top-k joinable tables. Supports chamfer, inverse_chamfer,
symmetric_chamfer (avg chamfer), harmonic_chamfer similarity methods.
"""

from __future__ import annotations

import logging
import heapq
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from offline_embedding import create_embedder, load_column_embedding
from offline_sketch import SemanticSketch, load_offline_sketch, ConsolidatedSketchStore, farthest_point_sampling

# =============================================================================
# Config & Data Types
# =============================================================================

@dataclass
class QueryConfig:
    top_k_return: int = 50
    similarity_threshold: float = 0.7
    query_sketch_size: int = 0
    d_sketch_size: int = 64
    similarity_method: str = "symmetric_chamfer"
    device: str = "auto"
    embedding_model: str = "embeddinggemma"
    embedding_dim: int = 128
    large_table_sample_size: int = 0

@dataclass
class QueryColumn:
    """Represents a query column."""
    table_name: str
    column_name: str
    values: List[str]

@dataclass
class QueryResult:
    candidate_table: str
    candidate_column: str
    similarity_score: float

@dataclass
class QueryStats:
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    total_embedding_time: float = 0.0
    total_search_time: float = 0.0
    total_candidates_found: int = 0

# =============================================================================
# Query Processor
# =============================================================================

def discover_columns(base_dir: Path, ext: str = "*.pkl") -> Dict[Tuple[str, str, int], Path]:
    """Discover (table, column, index) from dir structure."""
    out = {}
    if not base_dir.exists():
        return out
    for table_dir in base_dir.iterdir():
        if not table_dir.is_dir():
            continue
        for f in table_dir.glob(ext):
            if "_metadata" in f.name:
                continue
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                out[(table_dir.name, parts[0], int(parts[1]))] = f
    return out


def embeddings_to_sketch(emb: np.ndarray) -> SemanticSketch:
    return SemanticSketch(
        representative_vectors=emb,
        representative_ids=list(range(len(emb))),
        distances_to_origin=np.array([]),
        embedding_dim=emb.shape[1] if len(emb) > 0 else 0,
        k=len(emb),
        centroid=np.array([]),
        representative_names=None
    )


class SemanticJoinQueryProcessor:
    """Processes semantic join queries. Uses sketches or full embeddings (d_sketch_size=0)."""

    def __init__(self, config: QueryConfig, sketches_dir: Path,
                 embeddings_dir: Optional[Path] = None,
                 datalake_dir: Optional[Path] = None,
                 preload: bool = True,
                 use_consolidated: bool = True,
                 mmap_mode: Optional[str] = "r"):
        self.config = config
        self.sketches_dir = Path(sketches_dir)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.datalake_dir = datalake_dir
        self.stats = QueryStats()
        self.embedder = None
        self.logger = logging.getLogger(__name__)
        self.use_full_embeddings = config.d_sketch_size == 0

        if self.use_full_embeddings:
            if not self.embeddings_dir or not self.embeddings_dir.exists():
                raise ValueError("embeddings_dir required when d_sketch_size=0")
            self.available_columns = discover_columns(self.embeddings_dir)
            self.consolidated_store = None
            self.available_sketches = {}
            self.sketches_cache = {}
            self.embeddings_cache: Dict[Tuple[str, str, int], np.ndarray] = {}
            if preload and len(self.available_columns) > 0:
                self._preload_all_embeddings()
        else:
            self.available_columns = {}
            self.embeddings_cache = {}
            consolidated_path = self.sketches_dir / "sketches_consolidated.npy"
            if use_consolidated and consolidated_path.exists():
                self.consolidated_store = ConsolidatedSketchStore(self.sketches_dir)
                self.consolidated_store.load(mmap_mode=mmap_mode)
                self.available_sketches = {k: None for k in self.consolidated_store.keys()}
            else:
                self.consolidated_store = None
                self.available_sketches = discover_columns(self.sketches_dir)
            self.sketches_cache = {}
            if preload and self.consolidated_store is None and self.available_sketches:
                for (tn, cn, ci) in self.available_sketches:
                    sk = load_offline_sketch(tn, cn, ci, self.sketches_dir)
                    if sk is not None:
                        self.sketches_cache[(tn, cn, ci)] = sk

    def process_query(self, query: QueryColumn) -> List[QueryResult]:
        """Embed query, find similar columns, return top-k."""
        start = time.time()
        self.stats.total_queries += 1
        try:
            if self.config.large_table_sample_size > 0 and len(query.values) > self.config.large_table_sample_size:
                query.values = random.sample(query.values, self.config.large_table_sample_size)

            # Query Embedding Time
            t0 = time.time()
            query_emb = self._build_query_embeddings(query)
            self.stats.total_embedding_time += time.time() - t0
            if query_emb is None:
                self.stats.failed_queries += 1
                return []

            # Search Time
            t1 = time.time()
            results = self._find_similar_columns(query_emb, query)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            final = results[:self.config.top_k_return]
            self.stats.total_search_time += time.time() - t1

            self.stats.successful_queries += 1
            self.stats.total_processing_time += time.time() - start
            self.stats.total_candidates_found += len(final)
            return final
        except Exception:
            self.stats.failed_queries += 1
            return []
    
    def _build_query_embeddings(self, query: QueryColumn) -> Optional[SemanticSketch]:
        """Embed query values and optionally sketch them.
        
        Returns query embeddings (optionally sketched) to compare against datalake sketches.
        """
        if self.embedder is None:
            self.embedder = create_embedder(
                model=self.config.embedding_model,
                device=self.config.device,
                embedding_dim=self.config.embedding_dim,
                mode="query"
            )
        
        values = query.values
        # if len(values) > self.config.large_table_sample_size:
        #     import random
        #     values = random.sample(values, self.config.large_table_sample_size)
        
        embeddings = self.embedder.embed_values(values, query.column_name)
        if embeddings is None or len(embeddings) == 0:
            return None
        
        # Optionally sketch the query embeddings
        if self.config.query_sketch_size > 0 and len(embeddings) > self.config.query_sketch_size:
            original_count = len(embeddings)
            embeddings, sketch_ids = self._sketch_embeddings(embeddings, self.config.query_sketch_size)
            sketch_values = [values[i] for i in sketch_ids] if sketch_ids else []
            self.logger.info(f"Sketched query embeddings: {original_count} → {len(embeddings)}")
        else:
            sketch_values = values
        
        return SemanticSketch(
            representative_vectors=embeddings,
            representative_ids=[],
            distances_to_origin=np.array([]),
            embedding_dim=embeddings.shape[1],
            k=len(embeddings),
            centroid=np.array([]),
            representative_names=sketch_values
        )
    
    def _sketch_embeddings(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, List[int]]:
        """Sketch embeddings using Farthest Point Sampling (FPS).
        
        Same algorithm used for datalake sketches - ensures diverse, well-distributed
        representatives that cover the full semantic space.
        """
        n_samples = len(embeddings)
        if k >= n_samples:
            return embeddings, list(range(n_samples))
        
        selected_indices = farthest_point_sampling(embeddings, k)
        return embeddings[selected_indices], selected_indices.tolist()
    
    def _find_similar_columns(self, query_embeddings: SemanticSketch, 
                              query: QueryColumn) -> List[QueryResult]:
        """Find similar columns by comparing query embeddings against datalake sketches or full embeddings."""
        
        # Determine data source based on mode
        if self.use_full_embeddings:
            num_candidates = len(self.available_columns)
            candidate_items = self.available_columns.items()
            self.logger.info(f"Comparing {query_embeddings.k} query embeddings against {num_candidates} datalake columns (full embeddings)")
        else:
            num_candidates = len(self.available_sketches)
            candidate_items = self.available_sketches.items()
            self.logger.info(f"Comparing {query_embeddings.k} query embeddings against {num_candidates} datalake sketches")
        
        # Use a min-heap to track top-k results (heap stores negative scores for max behavior)
        # Format: (negative_score, table_name, column_name)
        top_k_heap: List[Tuple[float, str, str]] = []
        min_score_threshold = 0.0  # Minimum score to consider (updates as heap fills)
        processed = 0
        skipped_early = 0
        
        # Determine the sketch source (only used when not in full embeddings mode)
        use_consolidated = self.consolidated_store is not None
        use_sketch_cache = len(self.sketches_cache) > 0
        use_embedding_cache = len(self.embeddings_cache) > 0
        
        if self.use_full_embeddings:
            if use_embedding_cache:
                self.logger.info(f"Using pre-loaded embeddings cache ({len(self.embeddings_cache)} embeddings)")
        else:
            if use_consolidated:
                self.logger.info(f"Using consolidated sketch store ({len(self.consolidated_store)} sketches)")
            elif use_sketch_cache:
                self.logger.info(f"Using pre-loaded sketches cache ({len(self.sketches_cache)} sketches)")
        
        # Normalize query table name once (remove .csv extension)
        query_table_normalized = query.table_name.replace('.csv', '')
        
        for (table_name, column_name, column_index), file_path in candidate_items:
            try:
                # Skip the query table itself
                table_name_normalized = table_name.replace('.csv', '')
                if table_name_normalized == query_table_normalized:
                    continue
                
                # Get candidate data from the appropriate source
                cache_key = (table_name, column_name, column_index)
                
                if self.use_full_embeddings:
                    # Full embeddings mode
                    if use_embedding_cache and cache_key in self.embeddings_cache:
                        embeddings = self.embeddings_cache[cache_key]
                    else:
                        embeddings = load_column_embedding(table_name, column_name, column_index, self.embeddings_dir)
                    
                    if embeddings is None:
                        continue
                    candidate_sketch = embeddings_to_sketch(embeddings)
                else:
                    # Sketch mode
                    if use_consolidated:
                        candidate_sketch = self.consolidated_store.get_sketch(table_name, column_name, column_index)
                    elif use_sketch_cache and cache_key in self.sketches_cache:
                        candidate_sketch = self.sketches_cache[cache_key]
                    else:
                        candidate_sketch = load_offline_sketch(table_name, column_name, column_index, self.sketches_dir)
                
                if candidate_sketch is None:
                    continue
                
                semantic_matches, semantic_density = self._estimate_semantic_joinability(
                    query_embeddings, candidate_sketch, min_score_threshold
                )
                
                processed += 1
                
                # Skip if below threshold
                if self.config.similarity_threshold > 0.1 and semantic_density < self.config.similarity_threshold:
                    continue
                
                # Early termination: skip if score can't make it to top-k
                if len(top_k_heap) >= self.config.top_k_return and semantic_density <= min_score_threshold:
                    skipped_early += 1
                    continue
                
                # Add to heap
                if len(top_k_heap) < self.config.top_k_return:
                    heapq.heappush(top_k_heap, (semantic_density, table_name, column_name))
                    if len(top_k_heap) == self.config.top_k_return:
                        min_score_threshold = top_k_heap[0][0]  # Update threshold
                elif semantic_density > min_score_threshold:
                    heapq.heapreplace(top_k_heap, (semantic_density, table_name, column_name))
                    min_score_threshold = top_k_heap[0][0]  # Update threshold
                
                # Log progress every 10000 candidates
                if processed % 10000 == 0:
                    self.logger.info(f"Processed {processed} candidates, current min threshold: {min_score_threshold:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Error processing candidate {table_name}.{column_name}: {e}")
                continue
        
        self.logger.info(f"Processed {processed} candidates total, skipped {skipped_early} via early termination")
        
        # Convert heap to results
        results = [
            QueryResult(
                candidate_table=table_name,
                candidate_column=column_name,
                similarity_score=float(score)
            )
            for score, table_name, column_name in top_k_heap
        ]
        
        return results
    
    def _load_column_values_for_debug(self, table_name: str, column_name: str, 
                                       representative_ids: List[int]) -> Optional[List[str]]:
        """Load column values on-demand for debugging.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            representative_ids: Indices into the unique values to select
            
        Returns:
            List of representative value strings, or None if loading fails
        """
        if self.datalake_dir is None:
            return None
        
        try:
            csv_file = self.datalake_dir / f"{table_name}.csv"
            if not csv_file.exists():
                return None
            
            df = pd.read_csv(csv_file)
            if column_name not in df.columns:
                return None
            
            # Get unique values in order (same as during embedding)
            unique_values = df[column_name].dropna().astype(str).str.strip().str.lower()
            unique_values = unique_values[unique_values != ''].drop_duplicates().tolist()
            
            # Use representative_ids to get the selected values
            selected_values = []
            for idx in representative_ids:
                if 0 <= idx < len(unique_values):
                    selected_values.append(unique_values[idx])
                else:
                    selected_values.append(f"d_{idx}")
            
            return selected_values
        except Exception as e:
            self.logger.debug(f"Could not load values for {table_name}.{column_name}: {e}")
            return None
    
    def _estimate_semantic_joinability(self, query_emb: SemanticSketch, datalake_sketch: SemanticSketch, 
                                        min_score_threshold: float = 0.0,
                                        candidate_info: Optional[Tuple[str, str]] = None) -> Tuple[int, float]:
        """Estimate semantic joinability between query embeddings and datalake sketch.
        
        Compares all query embeddings against datalake sketch vectors (no query sketching).
        
        Similarity methods:
        - "chamfer": Chamfer similarity (MaxSim) - iterates over query embeddings
                     Chamfer(Q, D) = mean over q in Q of max over d in D of sim(q, d)
                     Reference: https://arxiv.org/abs/2405.19504
        - "inverse_chamfer": Inverse Chamfer - iterates over datalake sketch
                     Chamfer(D, Q) = mean over d in D of max over q in Q of sim(d, q)
        - "symmetric_chamfer": Arithmetic mean of chamfer and inverse_chamfer.
                     Balanced combination of both directions.
        - "harmonic_chamfer": Harmonic mean of both directions.
                     More conservative - penalizes when either direction has low similarity.
                     Good for reducing false positives across different benchmark sizes.
        
        Args:
            query_emb: All query embeddings (not sketched)
            datalake_sketch: Pre-computed datalake sketch
            min_score_threshold: Minimum score needed to be considered (for early termination)
        """
        if query_emb.k == 0 or datalake_sketch.k == 0:
            return 0, 0.0
        
        # Normalize vectors for cosine similarity
        q_norm = query_emb.representative_vectors / (np.linalg.norm(query_emb.representative_vectors, axis=1, keepdims=True) + 1e-12)
        d_norm = datalake_sketch.representative_vectors / (np.linalg.norm(datalake_sketch.representative_vectors, axis=1, keepdims=True) + 1e-12)
        
        # Compute similarity matrix: (num_query_embeddings, datalake_sketch_size)
        similarity_matrix = np.dot(q_norm, d_norm.T)
        
        method = self.config.similarity_method
        
        if method == "chamfer":
            # Chamfer similarity (MaxSim): for each query embedding, find max sim to datalake sketch
            # Chamfer(Q, D) = mean over q in Q of max over d in D of sim(q, d)
            
            # Early stopping: check first few query vectors to estimate upper bound
            early_check_size = min(10, similarity_matrix.shape[0])
            early_max_sims = np.max(similarity_matrix[:early_check_size, :], axis=1)
            early_estimate = float(np.mean(early_max_sims))
            
            # If early estimate is way below threshold, skip full computation
            if early_estimate < self.config.similarity_threshold * 0.8:
                return 0, 0.0
            
            # Also skip if all early similarities are very low
            if np.all(early_max_sims < self.config.similarity_threshold):
                return 0, 0.0
            
            use_gpu = (TORCH_AVAILABLE and 
                       self.config.device in ("cuda", "auto") and 
                       torch.cuda.is_available())
            if use_gpu:
                sim_tensor = torch.from_numpy(similarity_matrix).cuda()
                max_sims_per_query = torch.max(sim_tensor, dim=1).values
                semantic_density = float(torch.mean(max_sims_per_query).cpu())
            else:
                max_sims_per_query = np.max(similarity_matrix, axis=1)
                semantic_density = float(np.mean(max_sims_per_query))
            semantic_matches = query_emb.k
            
        
        elif method == "inverse_chamfer":
            # Inverse Chamfer: for each datalake sketch vector, find max sim to query embeddings
            # Chamfer(D, Q) = mean over d in D of max over q in Q of sim(d, q)
            
            # Early stopping: check first few datalake vectors to estimate upper bound
            early_check_size = min(10, similarity_matrix.shape[1])
            early_max_sims = np.max(similarity_matrix[:, :early_check_size], axis=0)
            early_estimate = float(np.mean(early_max_sims))
            
            # If early estimate is way below threshold, skip full computation
            if early_estimate < self.config.similarity_threshold * 0.8:
                return 0, 0.0
            
            # Also skip if all early similarities are very low
            if np.all(early_max_sims < self.config.similarity_threshold):
                return 0, 0.0
            
            use_gpu = (TORCH_AVAILABLE and 
                       self.config.device in ("cuda", "auto") and 
                       torch.cuda.is_available())
            if use_gpu:
                sim_tensor = torch.from_numpy(similarity_matrix).cuda()
                max_sims_per_datalake = torch.max(sim_tensor, dim=0).values
                semantic_density = float(torch.mean(max_sims_per_datalake).cpu())
            else:
                max_sims_per_datalake = np.max(similarity_matrix, axis=0)
            semantic_matches = datalake_sketch.k

        
        elif method in ("symmetric_chamfer", "harmonic_chamfer"):
            # Combined Chamfer methods: compute both directions and combine
            # This reduces false positives by requiring good coverage in BOTH directions
            
            # Early stopping: quick check on both directions
            early_check_size = min(10, similarity_matrix.shape[0], similarity_matrix.shape[1])
            early_max_query = np.max(similarity_matrix[:early_check_size, :], axis=1)
            early_max_datalake = np.max(similarity_matrix[:, :early_check_size], axis=0)
            early_chamfer = float(np.mean(early_max_query))
            early_inv_chamfer = float(np.mean(early_max_datalake))
            
            if early_chamfer < self.config.similarity_threshold * 0.8 or early_inv_chamfer < self.config.similarity_threshold * 0.8:
                return 0, 0.0

            # Also skip if all early similarities are very low
            if np.all(early_max_query < self.config.similarity_threshold) or np.all(early_max_datalake < self.config.similarity_threshold):
                return 0, 0.0
            
            use_gpu = (TORCH_AVAILABLE and 
                       self.config.device in ("cuda", "auto") and 
                       torch.cuda.is_available())
            
            # Traditional chamfer: max similarity
            if use_gpu:
                sim_tensor = torch.from_numpy(similarity_matrix).cuda()
                max_sims_per_query = torch.max(sim_tensor, dim=1).values
                chamfer_score = float(torch.mean(max_sims_per_query).cpu())
                max_sims_per_datalake = torch.max(sim_tensor, dim=0).values
                inv_chamfer_score = float(torch.mean(max_sims_per_datalake).cpu())
            else:
                max_sims_per_query = np.max(similarity_matrix, axis=1)
                chamfer_score = float(np.mean(max_sims_per_query))
                max_sims_per_datalake = np.max(similarity_matrix, axis=0)
                inv_chamfer_score = float(np.mean(max_sims_per_datalake))
            
            # Combine the two scores
            if method == "symmetric_chamfer":
                # Arithmetic mean: balanced combination
                semantic_density = (chamfer_score + inv_chamfer_score) / 2.0
            elif method == "harmonic_chamfer":
                # Harmonic mean: penalizes when either score is low
                # More conservative - good for reducing false positives
                if chamfer_score > 0 and inv_chamfer_score > 0:
                    semantic_density = 2.0 * chamfer_score * inv_chamfer_score / (chamfer_score + inv_chamfer_score)
                else:
                    semantic_density = 0.0
            else:  # traditional chamfer
                semantic_density = chamfer_score
            
            semantic_matches = min(query_emb.k, datalake_sketch.k)
            
        return int(semantic_matches), float(semantic_density)
    
    
    def print_stats(self) -> None:
        """Print query processing statistics."""
        stats = self.stats
        print(f"\n=== QUERY PROCESSING STATISTICS ===")
        print(f"Total queries: {stats.total_queries}")
        print(f"Successful queries: {stats.successful_queries}")
        print(f"Failed queries: {stats.failed_queries}")
        print(f"Total processing time: {stats.total_processing_time:.2f}s")
        print(f"Total query time (embedding + search): {stats.total_embedding_time + stats.total_search_time:.2f}s")
        print(f"  - Query embedding time: {stats.total_embedding_time:.2f}s")
        print(f"  - Semantic search time: {stats.total_search_time:.2f}s")
        print(f"Total candidates found: {stats.total_candidates_found}")
        
        if stats.successful_queries > 0:
            num_queries = stats.successful_queries
            avg_time = (stats.total_embedding_time + stats.total_search_time) / num_queries
            avg_candidates = stats.total_candidates_found / num_queries
            print(f"Average time per query (embedding + search): {avg_time:.4f}s")
            print(f"  - Avg embedding time per query: {stats.total_embedding_time / num_queries:.4f}s")
            print(f"  - Avg search time per query: {stats.total_search_time / num_queries:.4f}s")
            print(f"Average candidates per query: {avg_candidates:.2f}")


def save_query_results(results: List[QueryResult], output_file: Path,
                       query_sample_values: Optional[List[str]] = None,
                       datalake_dir: Optional[Path] = None) -> None:
    """Save query results to CSV."""
    rows = []
    for result in results:
        row = {
            'candidate_table': result.candidate_table,
            'candidate_column': result.candidate_column,
            'similarity_score': result.similarity_score
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
