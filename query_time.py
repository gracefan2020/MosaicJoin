"""
Query Time Module

Given a query table, embeds all query column values and compares against pre-computed
datalake sketches to find top-k joinable tables. No sketching is performed on the query
side - all query embeddings are used directly.

Supports multiple similarity computation methods:
- mean: Average of all pairwise similarities
- greedy_match: Greedy 1-to-1 bipartite matching
- top_k_mean: Average of top-k highest similarities
- max: Maximum similarity only
- chamfer: Chamfer similarity (MaxSim) - for each query embedding, find max similarity
           to datalake sketch vectors. Reference: https://arxiv.org/abs/2405.19504
- inverse_chamfer: Inverse Chamfer - for each datalake sketch vector, find max similarity
           to query embeddings.
- symmetric_chamfer: Arithmetic mean of chamfer and inverse_chamfer. Balanced combination.
- harmonic_chamfer: Harmonic mean of both directions. More conservative - penalizes when
           either direction has low similarity. Good for reducing false positives.
- min_chamfer: Minimum of both directions. Most conservative - requires both directions
           to have high similarity.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from offline_embedding import MPNetEmbedder, create_mpnet_embedder, create_embedder, load_column_embedding
from offline_sketch import SemanticSketch, load_offline_sketch, ConsolidatedSketchStore, farthest_point_sampling


@dataclass
class QueryConfig:
    """Configuration for query processing."""
    top_k_return: int = 50
    similarity_threshold: float = 0.7
    sketch_size: int = 1024
    query_sketch_size: int = 0  # 0 = no sketching (use all query embeddings), >0 = sketch to this size
    similarity_method: str = "mean"  # "mean", "greedy_match", "top_k_mean", "max", "chamfer", "inverse_chamfer", "symmetric_chamfer", "harmonic_chamfer", "min_chamfer"
    top_k_for_mean: int = 100
    enable_early_stopping: bool = True
    early_subset_size: int = 256
    enable_large_table_sampling: bool = True
    large_table_sample_size: int = 1000
    device: str = "auto"  # "auto", "cpu", "cuda" - used for embeddings AND chamfer similarity
    embedding_model: str = "mpnet"  # "mpnet" | "embeddinggemma"
    embedding_dim: int = 128  # Output dimension for embeddinggemma
    debug_matches: bool = False  # Print detailed match info for debugging
    debug_top_n: int = 10  # Number of top matches to print per query/datalake vector


@dataclass
class QueryColumn:
    """Represents a query column."""
    table_name: str
    column_name: str
    values: List[str]


@dataclass
class QueryResult:
    """Result of a semantic join query."""
    candidate_table: str
    candidate_column: str
    similarity_score: float


@dataclass
class QueryStats:
    """Statistics for query processing."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    total_embedding_time: float = 0.0
    total_search_time: float = 0.0
    total_candidates_found: int = 0
    total_high_quality_candidates: int = 0


class SemanticJoinQueryProcessor:
    """Processes semantic join queries using pre-built sketches."""
    
    def __init__(self, config: QueryConfig, sketches_dir: Path, 
                 embeddings_dir: Optional[Path] = None,
                 datalake_dir: Optional[Path] = None,
                 preload_sketches: bool = True,
                 use_consolidated: bool = True,
                 mmap_mode: Optional[str] = 'r'):
        """
        Args:
            config: Query configuration
            sketches_dir: Path to sketches directory (individual files or consolidated store)
            embeddings_dir: Optional path to embeddings directory
            datalake_dir: Optional path to datalake directory
            preload_sketches: Whether to preload all sketches into memory
            use_consolidated: Try to use consolidated sketch store if available (much faster)
            mmap_mode: Memory-map mode for consolidated store ('r', 'r+', 'c', None)
                       'r' = read-only mmap (fastest startup, lowest memory)
                       None = load fully into RAM (faster query access)
        """
        self.config = config
        self.sketches_dir = sketches_dir
        self.embeddings_dir = embeddings_dir
        self.datalake_dir = datalake_dir
        self.stats = QueryStats()
        self.embedder = None
        self.logger = self._setup_logging()
        self.mmap_mode = mmap_mode
        
        # Try to use consolidated store if available
        self.consolidated_store: Optional[ConsolidatedSketchStore] = None
        consolidated_path = sketches_dir / "sketches_consolidated.npy"
        
        if use_consolidated and consolidated_path.exists():
            self.logger.info("Found consolidated sketch store, using fast loading...")
            self._load_consolidated_store()
            self.available_sketches = {key: None for key in self.consolidated_store.keys()}
        else:
            if use_consolidated:
                self.logger.info("No consolidated store found. Consider running consolidate_sketches() for faster loading.")
            self.available_sketches = self._discover_available_sketches()
        
        # Pre-loaded sketches cache (only used when not using consolidated store)
        self.sketches_cache: Dict[Tuple[str, str, int], SemanticSketch] = {}
        if preload_sketches and self.consolidated_store is None and len(self.available_sketches) > 0:
            self._preload_all_sketches()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("SemanticJoinQueryProcessor")
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(console_handler)
        return logger
    
    def _load_consolidated_store(self) -> None:
        """Load the consolidated sketch store."""
        start_time = time.time()
        self.consolidated_store = ConsolidatedSketchStore(self.sketches_dir)
        self.consolidated_store.load(mmap_mode=self.mmap_mode)
        elapsed = time.time() - start_time
        self.logger.info(f"Loaded consolidated store with {len(self.consolidated_store)} sketches in {elapsed:.2f}s")
    
    def _discover_available_sketches(self) -> Dict[Tuple[str, str, int], Path]:
        sketches = {}
        if not self.sketches_dir.exists():
            return sketches
        
        for table_dir in self.sketches_dir.iterdir():
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
                        sketches[(table_name, column_name, column_index)] = sketch_file
                except Exception as e:
                    continue
        
        if len(sketches) > 0:
            self.logger.info(f"Found {len(sketches)} available sketches")
        return sketches
    
    def _preload_all_sketches(self) -> None:
        """Pre-load all sketches into memory for faster query processing."""
        self.logger.info(f"Pre-loading {len(self.available_sketches)} sketches into memory...")
        start_time = time.time()
        
        loaded = 0
        failed = 0
        
        for (table_name, column_name, column_index), sketch_path in self.available_sketches.items():
            try:
                sketch = load_offline_sketch(table_name, column_name, column_index, self.sketches_dir)
                if sketch is not None:
                    self.sketches_cache[(table_name, column_name, column_index)] = sketch
                    loaded += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                continue
            
            # Log progress every 50000 sketches
            if (loaded + failed) % 50000 == 0:
                self.logger.info(f"  Loaded {loaded} sketches ({failed} failed)...")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Pre-loaded {loaded} sketches in {elapsed:.2f}s ({failed} failed)")
        
    
    def process_query(self, query: QueryColumn) -> List[QueryResult]:
        """Process a semantic join query."""
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            # Time embedding
            embed_start = time.time()
            query_embeddings = self._build_query_embeddings(query)
            embed_time = time.time() - embed_start
            
            if query_embeddings is None:
                self.stats.failed_queries += 1
                return []
            
            # Time search
            search_start = time.time()
            results = self._find_similar_columns(query_embeddings, query)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = results[:self.config.top_k_return]
            search_time = time.time() - search_start
            
            processing_time = time.time() - start_time
            self.stats.successful_queries += 1
            self.stats.total_processing_time += processing_time
            self.stats.total_embedding_time += embed_time
            self.stats.total_search_time += search_time
            self.stats.total_candidates_found += len(final_results)
            
            self.logger.info(f"Query processed in {processing_time:.2f}s (embed: {embed_time:.2f}s, search: {search_time:.2f}s), found {len(final_results)} candidates")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            self.stats.failed_queries += 1
            return []
    
    def _build_query_embeddings(self, query: QueryColumn) -> Optional[SemanticSketch]:
        """Embed query values and optionally sketch them.
        
        If query_sketch_size > 0, applies K-means clustering to reduce query embeddings
        to a fixed-size sketch. Otherwise, returns all query embeddings.
        
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
        if self.config.enable_large_table_sampling and len(values) > self.config.large_table_sample_size:
            import random
            values = random.sample(values, self.config.large_table_sample_size)
        
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
        """Find similar columns by comparing query embeddings against datalake sketches."""
        self.logger.info(f"Comparing {query_embeddings.k} query embeddings against {len(self.available_sketches)} datalake sketches")
        
        # Use a min-heap to track top-k results (heap stores negative scores for max behavior)
        # Format: (negative_score, table_name, column_name)
        top_k_heap: List[Tuple[float, str, str]] = []
        min_score_threshold = 0.0  # Minimum score to consider (updates as heap fills)
        processed = 0
        skipped_early = 0
        
        # Determine the sketch source
        use_consolidated = self.consolidated_store is not None
        use_cache = len(self.sketches_cache) > 0
        
        if use_consolidated:
            self.logger.info(f"Using consolidated sketch store ({len(self.consolidated_store)} sketches)")
        elif use_cache:
            self.logger.info(f"Using pre-loaded sketches cache ({len(self.sketches_cache)} sketches)")
        
        # Normalize query table name once (remove .csv extension)
        query_table_normalized = query.table_name.replace('.csv', '')
        
        for (table_name, column_name, column_index), sketch_path in self.available_sketches.items():
            try:
                # Skip the query table itself
                table_name_normalized = table_name.replace('.csv', '')
                if table_name_normalized == query_table_normalized:
                    continue
                
                # Get candidate sketch from the appropriate source
                cache_key = (table_name, column_name, column_index)
                if use_consolidated:
                    candidate_sketch = self.consolidated_store.get_sketch(table_name, column_name, column_index)
                elif use_cache and cache_key in self.sketches_cache:
                    candidate_sketch = self.sketches_cache[cache_key]
                else:
                    candidate_sketch = load_offline_sketch(table_name, column_name, column_index, self.sketches_dir)
                
                if candidate_sketch is None:
                    continue
                
                semantic_matches, semantic_density = self._estimate_semantic_joinability(
                    query_embeddings, candidate_sketch, min_score_threshold,
                    candidate_info=(table_name, column_name) if self.config.debug_matches else None
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
    
    def _print_debug_matches(self, similarity_matrix: np.ndarray, 
                              query_emb: SemanticSketch, datalake_sketch: SemanticSketch,
                              direction: str = "chamfer",
                              candidate_info: Optional[Tuple[str, str]] = None) -> None:
        """Print detailed match information for debugging.
        
        Args:
            similarity_matrix: (num_query, num_datalake) similarity scores
            query_emb: Query embeddings with representative_names
            datalake_sketch: Datalake sketch with representative_names
            direction: "chamfer" (query->datalake) or "inverse" (datalake->query)
            candidate_info: Optional (table_name, column_name) tuple for context
        """
        top_n = self.config.debug_top_n
        q_names = query_emb.representative_names or [f"q_{i}" for i in range(query_emb.k)]
        
        # Try to get datalake names: from sketch, or load on-demand from CSV
        d_names = datalake_sketch.representative_names
        if d_names is None and candidate_info is not None:
            d_names = self._load_column_values_for_debug(
                candidate_info[0], candidate_info[1], datalake_sketch.representative_ids
            )
        if d_names is None:
            d_names = [f"d_{i}" for i in range(datalake_sketch.k)]
        
        if candidate_info:
            print(f"\n  [DEBUG] Candidate: {candidate_info[0]}.{candidate_info[1]}")
        
        if direction == "chamfer":
            # For each query embedding, show best datalake matches
            print(f"\n  === CHAMFER DEBUG: Query -> Datalake (top {top_n} per query) ===")
            max_sims = np.max(similarity_matrix, axis=1)
            best_indices = np.argmax(similarity_matrix, axis=1)
            
            # Sort by max similarity descending
            sorted_q_idx = np.argsort(max_sims)[::-1][:top_n]
            for q_idx in sorted_q_idx:
                d_idx = best_indices[q_idx]
                sim = max_sims[q_idx]
                q_val = q_names[q_idx] if q_idx < len(q_names) else f"q_{q_idx}"
                d_val = d_names[d_idx] if d_idx < len(d_names) else f"d_{d_idx}"
                print(f"    Q[{q_idx}] '{q_val[:50]}' -> D[{d_idx}] '{d_val[:50]}' | sim={sim:.4f}")
            
            # Also show worst matches (lowest max similarity)
            print(f"  --- Worst {top_n} query matches (lowest max sim) ---")
            worst_q_idx = np.argsort(max_sims)[:top_n]
            for q_idx in worst_q_idx:
                d_idx = best_indices[q_idx]
                sim = max_sims[q_idx]
                q_val = q_names[q_idx] if q_idx < len(q_names) else f"q_{q_idx}"
                d_val = d_names[d_idx] if d_idx < len(d_names) else f"d_{d_idx}"
                print(f"    Q[{q_idx}] '{q_val[:50]}' -> D[{d_idx}] '{d_val[:50]}' | sim={sim:.4f}")
                
        elif direction == "inverse":
            # For each datalake embedding, show best query matches
            print(f"\n  === INVERSE CHAMFER DEBUG: Datalake -> Query (top {top_n} per datalake) ===")
            max_sims = np.max(similarity_matrix, axis=0)
            best_indices = np.argmax(similarity_matrix, axis=0)
            
            # Sort by max similarity descending
            sorted_d_idx = np.argsort(max_sims)[::-1][:top_n]
            for d_idx in sorted_d_idx:
                q_idx = best_indices[d_idx]
                sim = max_sims[d_idx]
                q_val = q_names[q_idx] if q_idx < len(q_names) else f"q_{q_idx}"
                d_val = d_names[d_idx] if d_idx < len(d_names) else f"d_{d_idx}"
                print(f"    D[{d_idx}] '{d_val[:50]}' -> Q[{q_idx}] '{q_val[:50]}' | sim={sim:.4f}")
            
            # Also show worst matches
            print(f"  --- Worst {top_n} datalake matches (lowest max sim) ---")
            worst_d_idx = np.argsort(max_sims)[:top_n]
            for d_idx in worst_d_idx:
                q_idx = best_indices[d_idx]
                sim = max_sims[d_idx]
                q_val = q_names[q_idx] if q_idx < len(q_names) else f"q_{q_idx}"
                d_val = d_names[d_idx] if d_idx < len(d_names) else f"d_{d_idx}"
                print(f"    D[{d_idx}] '{d_val[:50]}' -> Q[{q_idx}] '{q_val[:50]}' | sim={sim:.4f}")

    def _estimate_semantic_joinability(self, query_emb: SemanticSketch, datalake_sketch: SemanticSketch, 
                                        min_score_threshold: float = 0.0,
                                        candidate_info: Optional[Tuple[str, str]] = None) -> Tuple[int, float]:
        """Estimate semantic joinability between query embeddings and datalake sketch.
        
        Compares all query embeddings against datalake sketch vectors (no query sketching).
        
        Similarity methods:
        - "mean": Average of all pairwise similarities
        - "greedy_match": Greedy 1-to-1 bipartite matching (principled, no double-counting)
        - "top_k_mean": Average of top-k highest similarities
        - "max": Maximum similarity only
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
        
        if method == "greedy_match":
            # Greedy 1-to-1 bipartite matching
            semantic_density = self._greedy_match_similarity(similarity_matrix)
            semantic_matches = min(query_emb.k, datalake_sketch.k)
        
        elif method == "chamfer":
            # Chamfer similarity (MaxSim): for each query embedding, find max sim to datalake sketch
            # Chamfer(Q, D) = mean over q in Q of max over d in D of sim(q, d)
            
            # Early stopping: check first few query vectors to estimate upper bound
            early_check_size = min(10, similarity_matrix.shape[0])
            early_max_sims = np.max(similarity_matrix[:early_check_size, :], axis=1)
            early_estimate = float(np.mean(early_max_sims))
            
            # If early estimate is way below threshold, skip full computation
            if early_estimate < min_score_threshold * 0.8:
                return 0, early_estimate
            
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
            
            if self.config.debug_matches:
                self._print_debug_matches(similarity_matrix, query_emb, datalake_sketch, "chamfer", candidate_info)
        
        elif method == "inverse_chamfer":
            # Inverse Chamfer: for each datalake sketch vector, find max sim to query embeddings
            # Chamfer(D, Q) = mean over d in D of max over q in Q of sim(d, q)
            
            # Early stopping: check first few datalake vectors to estimate upper bound
            early_check_size = min(10, similarity_matrix.shape[1])
            early_max_sims = np.max(similarity_matrix[:, :early_check_size], axis=0)
            early_estimate = float(np.mean(early_max_sims))
            
            # If early estimate is way below threshold, skip full computation
            if early_estimate < min_score_threshold * 0.8:
                return 0, early_estimate
            
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
                semantic_density = float(np.mean(max_sims_per_datalake))
            semantic_matches = datalake_sketch.k
            
            if self.config.debug_matches:
                self._print_debug_matches(similarity_matrix, query_emb, datalake_sketch, "inverse", candidate_info)
        
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
            
            if self.config.debug_matches:
                print(f"  [DEBUG] chamfer={chamfer_score:.4f}, inv_chamfer={inv_chamfer_score:.4f}")
                self._print_debug_matches(similarity_matrix, query_emb, datalake_sketch, "chamfer", candidate_info)
                self._print_debug_matches(similarity_matrix, query_emb, datalake_sketch, "inverse", candidate_info)
        
        elif method == "top_k_mean":
            flat = similarity_matrix.ravel()
            k = min(self.config.top_k_for_mean, len(flat))
            top_k_values = np.partition(flat, -k)[-k:]
            semantic_density = float(np.mean(top_k_values))
            semantic_matches = k
        
        elif method == "max":
            semantic_density = float(np.max(similarity_matrix))
            semantic_matches = 1
        
        else:  # "mean"
            if self.config.similarity_threshold <= 0.1:
                semantic_density = float(np.mean(similarity_matrix))
                semantic_matches = int(np.sum(similarity_matrix > 0))
            else:
                semantic_matches = np.sum(similarity_matrix > self.config.similarity_threshold)
                semantic_density = semantic_matches / min(query_emb.k, datalake_sketch.k)
        
        return int(semantic_matches), float(semantic_density)
    
    def _greedy_match_similarity(self, similarity_matrix: np.ndarray) -> float:
        """Compute similarity using greedy 1-to-1 bipartite matching.
        
        This is a greedy approximation to optimal bipartite matching (Hungarian algorithm).
        Each row (query vector) matches to at most one column (candidate vector).
        
        Algorithm:
        1. Sort all pairs by similarity (descending)
        2. Greedily select pairs, skipping if either endpoint already used
        3. Return mean of selected similarities
    
        
        The greedy approach typically achieves >95% of optimal matching quality.
        """
        if similarity_matrix.size == 0:
            return 0.0
        
        n_rows, n_cols = similarity_matrix.shape
        n_matches = min(n_rows, n_cols)
        
        # Flatten and get indices sorted by similarity (descending)
        flat = similarity_matrix.ravel()
        sorted_indices = np.argsort(flat)[::-1]
        
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        matched_similarities = []
        
        for flat_idx in sorted_indices:
            row = flat_idx // n_cols
            col = flat_idx % n_cols
            
            if row not in used_rows and col not in used_cols:
                matched_similarities.append(flat[flat_idx])
                used_rows.add(row)
                used_cols.add(col)
                
                if len(matched_similarities) >= n_matches:
                    break
        
        return float(np.mean(matched_similarities)) if matched_similarities else 0.0
    
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
