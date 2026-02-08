"""
Query Time Module

Given a query table, builds sketch for query column and finds top-k joinable tables.
Supports multiple similarity computation methods:
- mean: Average of all k² pairwise similarities (original)
- greedy_match: Greedy 1-to-1 bipartite matching
- top_k_mean: Average of top-k highest similarities
- max: Maximum similarity only
- chamfer: Chamfer similarity (MaxSim) from MUVERA paper - sum of max similarities
           Reference: https://arxiv.org/abs/2405.19504
"""

from __future__ import annotations

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
from offline_sketch import SemanticSketch, load_offline_sketch


@dataclass
class QueryConfig:
    """Configuration for query processing."""
    top_k_return: int = 50
    similarity_threshold: float = 0.7
    sketch_size: int = 1024
    similarity_method: str = "mean"  # "mean", "greedy_match", "top_k_mean", "max", "chamfer"
    top_k_for_mean: int = 100
    enable_early_stopping: bool = True
    early_subset_size: int = 256
    enable_large_table_sampling: bool = True
    large_table_sample_size: int = 1000
    device: str = "auto"  # "auto", "cpu", "cuda" - used for embeddings AND chamfer similarity
    embedding_model: str = "mpnet"  # "mpnet" | "embeddinggemma"
    embedding_dim: int = 128  # Output dimension for embeddinggemma


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
    contributing_entities: Optional[List[Tuple[str, str, float]]] = None


@dataclass
class QueryStats:
    """Statistics for query processing."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    total_candidates_found: int = 0
    total_high_quality_candidates: int = 0


class SemanticJoinQueryProcessor:
    """Processes semantic join queries using pre-built sketches."""
    
    def __init__(self, config: QueryConfig, sketches_dir: Path, 
                 embeddings_dir: Optional[Path] = None,
                 datalake_dir: Optional[Path] = None):
        self.config = config
        self.sketches_dir = sketches_dir
        self.embeddings_dir = embeddings_dir
        self.datalake_dir = datalake_dir
        self.stats = QueryStats()
        self.embedder = None
        self.logger = self._setup_logging()
        self.available_sketches = self._discover_available_sketches()
    
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
    
    def process_query(self, query: QueryColumn) -> List[QueryResult]:
        """Process a semantic join query."""
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            query_sketch = self._build_query_sketch(query)
            if query_sketch is None:
                self.stats.failed_queries += 1
                return []
            
            results = self._find_similar_columns(query_sketch, query)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = results[:self.config.top_k_return]
            
            processing_time = time.time() - start_time
            self.stats.successful_queries += 1
            self.stats.total_processing_time += processing_time
            self.stats.total_candidates_found += len(final_results)
            
            self.logger.info(f"Query processed in {processing_time:.2f}s, found {len(final_results)} candidates")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            self.stats.failed_queries += 1
            return []
    
    def _build_query_sketch(self, query: QueryColumn) -> Optional[SemanticSketch]:
        """Build a sketch for the query column."""
        if self.embedder is None:
            self.embedder = create_embedder(
                model=self.config.embedding_model,
                device=self.config.device,
                embedding_dim=self.config.embedding_dim
            )
        
        values = query.values
        if self.config.enable_large_table_sampling and len(values) > self.config.large_table_sample_size:
            import random
            values = random.sample(values, self.config.large_table_sample_size)
        
        embeddings = self.embedder.embed_values(values, query.column_name)
        if embeddings is None or len(embeddings) == 0:
            return None
        
        k = min(self.config.sketch_size, len(embeddings))
        norms = np.linalg.norm(embeddings, axis=1)
        idx = np.argpartition(norms, k - 1)[:k]
        idx_sorted = idx[np.argsort(norms[idx])]
        
        representative_vectors = embeddings[idx_sorted]
        representative_names = [values[i] for i in idx_sorted]
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=list(range(k)),
            distances_to_origin=norms[idx_sorted],
            embedding_dim=embeddings.shape[1],
            k=k,
            centroid=np.mean(embeddings, axis=0),
            representative_names=representative_names
        )
    
    def _find_similar_columns(self, query_sketch: SemanticSketch, 
                              query: QueryColumn) -> List[QueryResult]:
        """Find similar columns using sketch comparison."""
        results = []
        
        self.logger.info(f"Comparing query sketch against {len(self.available_sketches)} candidate sketches")
        
        for (table_name, column_name, column_index), sketch_path in self.available_sketches.items():
            try:
                # Skip the query table itself
                if table_name == query.table_name:
                    continue
                
                candidate_sketch = load_offline_sketch(table_name, column_name, column_index, self.sketches_dir)
                if candidate_sketch is None:
                    continue
                
                semantic_matches, semantic_density = self._estimate_semantic_joinability(
                    query_sketch, candidate_sketch
                )
                
                if self.config.similarity_threshold <= 0.1 or semantic_density >= self.config.similarity_threshold:
                    contributing_entities = self._get_contributing_entities(
                        query_sketch, candidate_sketch, top_n=200
                    )
                    
                    result = QueryResult(
                        candidate_table=table_name,
                        candidate_column=column_name,
                        similarity_score=float(semantic_density),
                        contributing_entities=contributing_entities if contributing_entities else None
                    )
                    results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error processing candidate {table_name}.{column_name}: {e}")
                continue
        
        return results
    
    def _estimate_semantic_joinability(self, a: SemanticSketch, b: SemanticSketch) -> Tuple[int, float]:
        """Estimate semantic joinability between two semantic sketches.
        
        Similarity methods:
        - "mean": Average of all k² pairwise similarities (default, fast but noisy)
        - "greedy_match": Greedy 1-to-1 bipartite matching (principled, no double-counting)
        - "top_k_mean": Average of top-k highest similarities
        - "max": Maximum similarity only
        - "chamfer": Chamfer similarity (MaxSim) from MUVERA paper
                     Chamfer(A, B) = mean over a in A of max over b in B of sim(a, b)
                     Reference: https://arxiv.org/abs/2405.19504
        """
        if a.k == 0 or b.k == 0:
            return 0, 0.0
        
        # Normalize vectors for cosine similarity
        a_norm = a.representative_vectors / (np.linalg.norm(a.representative_vectors, axis=1, keepdims=True) + 1e-12)
        b_norm = b.representative_vectors / (np.linalg.norm(b.representative_vectors, axis=1, keepdims=True) + 1e-12)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(a_norm, b_norm.T)  # shape (a.k, b.k)
        
        method = self.config.similarity_method
        
        if method == "greedy_match":
            # Greedy 1-to-1 bipartite matching
            semantic_density = self._greedy_match_similarity(similarity_matrix)
            semantic_matches = min(a.k, b.k)
        
        elif method == "chamfer":
            # Chamfer similarity (MaxSim) from MUVERA paper
            # Chamfer(A, B) = sum over a in A of max over b in B of sim(a, b)
            # We normalize by |A| to get a density score in [0, 1]
            use_gpu = (TORCH_AVAILABLE and 
                       self.config.device in ("cuda", "auto") and 
                       torch.cuda.is_available())
            if use_gpu:
                # GPU-accelerated computation using PyTorch
                sim_tensor = torch.from_numpy(similarity_matrix).cuda()
                max_sims_per_query = torch.max(sim_tensor, dim=1).values
                semantic_density = float(torch.mean(max_sims_per_query).cpu())
            else:
                # CPU computation using NumPy
                max_sims_per_query = np.max(similarity_matrix, axis=1)  # shape (a.k,)
                semantic_density = float(np.mean(max_sims_per_query))
            semantic_matches = a.k
        
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
                semantic_density = semantic_matches / min(a.k, b.k)
        
        return int(semantic_matches), float(semantic_density)
    
    def _greedy_match_similarity(self, similarity_matrix: np.ndarray) -> float:
        """Compute similarity using greedy 1-to-1 bipartite matching.
        
        This is a greedy approximation to optimal bipartite matching (Hungarian algorithm).
        Each row (query vector) matches to at most one column (candidate vector).
        
        Algorithm:
        1. Sort all pairs by similarity (descending)
        2. Greedily select pairs, skipping if either endpoint already used
        3. Return mean of selected similarities
        
        Time complexity: O(k² log k²) = O(k² log k)
        vs Hungarian: O(k³)
        
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
    
    def _get_contributing_entities(self, query_sketch: SemanticSketch, 
                                   candidate_sketch: SemanticSketch,
                                   top_n: int = 200) -> List[Tuple[str, str, float]]:
        """Get top contributing entity pairs between sketches."""
        if query_sketch.representative_names is None or candidate_sketch.representative_names is None:
            return []
        
        a_norm = query_sketch.representative_vectors / (np.linalg.norm(query_sketch.representative_vectors, axis=1, keepdims=True) + 1e-12)
        b_norm = candidate_sketch.representative_vectors / (np.linalg.norm(candidate_sketch.representative_vectors, axis=1, keepdims=True) + 1e-12)
        
        similarity_matrix = np.dot(a_norm, b_norm.T)
        
        flat = similarity_matrix.ravel()
        if flat.size == 0:
            return []
        
        target = min(top_n, min(query_sketch.k, candidate_sketch.k), flat.size)
        if target <= 0:
            return []
        
        used_i: Set[int] = set()
        used_j: Set[int] = set()
        selected: List[Tuple[int, int, float]] = []
        
        sorted_idx = np.argsort(flat)[::-1]
        for flat_idx in sorted_idx:
            i = flat_idx // similarity_matrix.shape[1]
            j = flat_idx % similarity_matrix.shape[1]
            
            if i in used_i or j in used_j:
                continue
            
            sim = float(similarity_matrix[i, j])
            used_i.add(i)
            used_j.add(j)
            selected.append((i, j, sim))
            
            if len(selected) >= target:
                break
        
        contributing_pairs = []
        for i, j, sim in selected:
            query_val = query_sketch.representative_names[i]
            cand_val = candidate_sketch.representative_names[j]
            contributing_pairs.append((query_val, cand_val, sim))
        
        return contributing_pairs
    
    def print_stats(self) -> None:
        """Print query processing statistics."""
        stats = self.stats
        print(f"\n=== QUERY PROCESSING STATISTICS ===")
        print(f"Total queries: {stats.total_queries}")
        print(f"Successful queries: {stats.successful_queries}")
        print(f"Failed queries: {stats.failed_queries}")
        print(f"Total processing time: {stats.total_processing_time:.2f}s")
        print(f"Total candidates found: {stats.total_candidates_found}")
        
        if stats.successful_queries > 0:
            avg_time = stats.total_processing_time / stats.successful_queries
            avg_candidates = stats.total_candidates_found / stats.successful_queries
            print(f"Average time per query: {avg_time:.4f}s")
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


def save_contributing_entities(entities: List[Tuple[str, str, float]], 
                               output_file: Path,
                               query_table: str, query_column: str,
                               candidate_table: str, candidate_column: str) -> None:
    """Save contributing entities to CSV."""
    rows = []
    for query_val, cand_val, sim in entities:
        rows.append({
            'query_table': query_table,
            'query_column': query_column,
            'query_value': query_val,
            'candidate_table': candidate_table,
            'candidate_column': candidate_column,
            'candidate_value': cand_val,
            'similarity': sim
        })
    
    df = pd.DataFrame(rows)
    output_path = Path(str(output_file) + "_contributing_entities.csv")
    df.to_csv(output_path, index=False)
