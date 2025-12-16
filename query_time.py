"""
Query Time Module

Given a query table, builds sketch for query column and finds top-k joinable tables.
This module handles the third stage of the semantic join pipeline.

Key features:
- Loads pre-built sketches from offline_sketch.py
- Builds query sketches on-demand
- Finds semantically similar columns using sketch comparison
- Configurable similarity thresholds and result counts
- Memory-efficient processing
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import logging

import numpy as np
import pandas as pd

# Import from other modules
from offline_embedding import MPNetEmbedder, create_mpnet_embedder, load_column_embedding
from offline_sketch import SemanticSketch, load_offline_sketch

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class QueryConfig:
    """Configuration for query processing."""
    # Core parameters
    top_k_return: int = 50  # final results to return
    similarity_threshold: float = 0.7  # threshold for semantic matching
    sketch_size: int = 1024  # k for k-closest selection
    
    # Processing options
    enable_early_stopping: bool = True
    early_subset_size: int = 256  # number of candidate values to probe before full processing
    
    # Large table sampling before sketching
    enable_large_table_sampling: bool = True
    large_table_sample_size: int = 1000
    
    # Device settings
    device: str = "auto"  # device for MPNet model

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
    contributing_entities: Optional[List[Tuple[str, str, float]]] = None  # (query_value, candidate_value, similarity)

@dataclass
class QueryStats:
    """Statistics for query processing."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    total_candidates_found: int = 0
    total_high_quality_candidates: int = 0

@dataclass
class ValueMatch:
    """Represents a match between a query value and candidate value."""
    query_value: str
    candidate_value: str
    similarity_score: float

@dataclass
class ValueMatchStats:
    """Statistics about value-level matching."""
    total_query_values: int
    total_candidate_values: int
    query_values_with_matches: int
    candidate_values_with_matches: int
    total_matches: int
    avg_matches_per_query_value: float
    max_matches_per_query_value: int
    min_matches_per_query_value: int
    one_to_one_matches: int  # Query values that match exactly 1 candidate value
    one_to_many_matches: int  # Query values that match >1 candidate values
    query_values_with_no_matches: int
    candidate_values_with_no_matches: int

# =============================================================================
# Core Classes
# =============================================================================

class SemanticJoinQueryProcessor:
    """Processes semantic join queries using pre-built sketches."""
    
    def __init__(self, config: QueryConfig, sketches_dir: Path, embeddings_dir: Optional[Path] = None,
                 datalake_dir: Optional[Path] = None):
        self.config = config
        self.sketches_dir = sketches_dir
        self.embeddings_dir = embeddings_dir
        self.datalake_dir = datalake_dir
        self.stats = QueryStats()
        self.embedder = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load all available sketches for fast lookup
        self.available_sketches = self._discover_available_sketches()
        # Logging is now handled inside _discover_available_sketches
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the query processor."""
        logger = logging.getLogger("SemanticJoinQueryProcessor")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(console_handler)
        
        return logger
    
    def _discover_available_sketches(self) -> Dict[Tuple[str, str, int], Path]:
        """Discover all available sketches."""
        sketches = {}
        
        if not self.sketches_dir.exists():
            # Don't warn if this looks like a temporary directory (for value-level matching only)
            if 'tmp' not in str(self.sketches_dir).lower() and 'temp' not in str(self.sketches_dir).lower():
                self.logger.warning(f"Sketches directory does not exist: {self.sketches_dir}")
            return sketches
        
        # Find all table directories
        table_dirs = [d for d in self.sketches_dir.iterdir() if d.is_dir()]
        
        for table_dir in table_dirs:
            table_name = table_dir.name
            
            # Find all sketch files in this table directory
            sketch_files = list(table_dir.glob("*.pkl"))
            
            for sketch_file in sketch_files:
                try:
                    # Extract column name and index from filename
                    # Format: column_name_column_index.pkl
                    filename = sketch_file.stem
                    if '_' in filename:
                        # Find the last underscore to split column name and index
                        parts = filename.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            column_name = parts[0]
                            column_index = int(parts[1])
                            sketches[(table_name, column_name, column_index)] = sketch_file
                        else:
                            self.logger.warning(f"Could not parse sketch filename: {filename}")
                    else:
                        self.logger.warning(f"Could not parse sketch filename: {filename}")
                        
                except Exception as e:
                    self.logger.warning(f"Could not process {sketch_file}: {e}")
                    continue
        
        # Only log if we have sketches or if this doesn't look like a temp directory
        if len(sketches) > 0 or ('tmp' not in str(self.sketches_dir).lower() and 'temp' not in str(self.sketches_dir).lower()):
            self.logger.info(f"Found {len(sketches)} available sketches")
        
        return sketches
    
    def process_query(self, query: QueryColumn) -> List[QueryResult]:
        """Process a semantic join query."""
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            # Build query sketch
            query_sketch = self._build_query_sketch(query)
            if query_sketch is None:
                self.stats.failed_queries += 1
                return []
            
            # Find similar columns using sketch comparison
            results = self._find_similar_columns(query_sketch, query)
            
            # Sort by similarity score and limit to exactly top_k_return results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            # Return exactly top_k_return results (or all available if fewer than top_k_return)
            final_results = results[:self.config.top_k_return]
            
            # Log if we couldn't return exactly k results
            if len(final_results) < self.config.top_k_return:
                self.logger.warning(f"Only {len(final_results)} results available (requested {self.config.top_k_return})")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats.successful_queries += 1
            self.stats.total_processing_time += processing_time
            self.stats.total_candidates_found += len(final_results)
            self.stats.total_high_quality_candidates += len(results)
            
            self.logger.info(f"Query processed in {processing_time:.2f}s, found {len(final_results)} candidates")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            self.stats.failed_queries += 1
            return []
    
    def _build_query_sketch(self, query: QueryColumn) -> Optional[SemanticSketch]:
        """Build semantic sketch for query column."""
        try:
            # Initialize embedder if needed
            if self.embedder is None:
                self.embedder = create_mpnet_embedder(self.config.device)
            
            # Normalize values
            normalized_values = [self._normalize_value(v) for v in query.values]
            clean_values = [v for v in normalized_values if v]
            
            if not clean_values:
                return None
            
            # Apply sampling if too many values for performance
            original_clean_values = clean_values.copy()
            if self.config.enable_large_table_sampling and len(clean_values) > self.config.large_table_sample_size:
                import random
                random.seed(42)
                clean_values = random.sample(clean_values, self.config.large_table_sample_size)
                self.logger.info(f"Sampled {len(clean_values)} values from {len(query.values)} for query sketch")
            
            # Generate embeddings with column context
            embeddings = self._embed_values_with_context(clean_values, query.column_name)
            
            if len(embeddings) == 0:
                return None
            
            # Build semantic sketch using k-closest to origin, passing values to store them
            sketch = self._build_semantic_sketch_from_embeddings(embeddings, clean_values)
            
            return sketch
            
        except Exception as e:
            self.logger.error(f"Error building query sketch: {e}")
            return None
    
    def _normalize_value(self, value: Any) -> str:
        """Normalize a raw cell value for embedding."""
        s = str(value) if value is not None else ""
        return s.strip().lower()
    
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
    
    def _build_semantic_sketch_from_embeddings(self, embeddings: np.ndarray, 
                                              values: Optional[List[str]] = None) -> Optional[SemanticSketch]:
        """Build a semantic sketch using k-closest points to origin.
        
        Args:
            embeddings: Array of embeddings
            values: Optional list of normalized values corresponding to embeddings (same order)
        """
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
        
        # Calculate centroid of all embeddings
        centroid = np.mean(embeddings, axis=0)
        
        # Select representatives using k-closest to origin
        closest_indices, distances = self._k_closest_to_origin(embeddings, self.config.sketch_size)
        
        # Extract representative vectors and calculate distances to origin
        representative_vectors = embeddings[closest_indices]
        representative_ids = list(range(len(closest_indices)))  # Simple ID assignment
        
        # Calculate distances to origin for all representatives
        distances_to_origin = np.linalg.norm(representative_vectors, axis=1)
        
        # Extract representative values if provided
        representative_names = None
        if values is not None and len(values) == len(embeddings):
            representative_names = [values[i] for i in closest_indices]
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=representative_ids,
            distances_to_origin=distances,
            embedding_dim=embeddings.shape[1],
            k=len(closest_indices),
            centroid=centroid,
            representative_names=representative_names
        )
    
    def _k_closest_to_origin(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k vectors closest to origin by L2 norm"""
        norms = np.linalg.norm(X, axis=1)
        k = min(k, len(norms))
        idx = np.argpartition(norms, k - 1)[:k]
        idx_sorted = idx[np.argsort(norms[idx])]
        return idx_sorted, norms[idx_sorted]
    
    def _find_similar_columns(self, query_sketch: SemanticSketch, query: QueryColumn) -> List[QueryResult]:
        """Find similar columns using sketch comparison."""
        results = []
        
        self.logger.info(f"Comparing query sketch against {len(self.available_sketches)} candidate sketches")
        
        for (table_name, column_name, column_index), sketch_path in self.available_sketches.items():
            try:
                # Skip the query table itself
                if table_name == query.table_name:
                    continue
                
                # Load candidate sketch
                candidate_sketch = load_offline_sketch(table_name, column_name, column_index, self.sketches_dir)
                if candidate_sketch is None:
                    continue

                # For older sketches, try to reconstruct representative values (best-effort)
                self._maybe_populate_sketch_values(table_name, column_name, column_index, candidate_sketch)
                
                # Compute semantic joinability
                semantic_matches, semantic_density = self._estimate_semantic_joinability(
                    query_sketch, candidate_sketch
                )
                
                # Include all results when threshold is 0.0 or very low (<= 0.1)
                # Otherwise, only include results above threshold
                if self.config.similarity_threshold <= 0.1 or semantic_density >= self.config.similarity_threshold:
                    # Extract retrieval evidence pairs (independent of thresholds)
                    contributing_entities = self._get_contributing_entities(
                        query_sketch, candidate_sketch, top_n=200
                    )
                    
                    result = QueryResult(
                        candidate_table=table_name,
                        candidate_column=column_name,
                        similarity_score=float(semantic_density),
                        contributing_entities=contributing_entities if contributing_entities else None
                    )
                    
                    # Keep all candidates
                    results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error processing candidate {table_name}.{column_name}: {e}")
                continue
        
        return results
    
    def _estimate_semantic_joinability(self, a: SemanticSketch, b: SemanticSketch) -> Tuple[int, float]:
        """Estimate semantic joinability between two semantic sketches."""
        if a.k == 0 or b.k == 0:
            return 0, 0.0
        
        # Calculate cosine similarity between all pairs of representative vectors
        # Normalize vectors for cosine similarity
        a_norm = a.representative_vectors / (np.linalg.norm(a.representative_vectors, axis=1, keepdims=True) + 1e-12)
        b_norm = b.representative_vectors / (np.linalg.norm(b.representative_vectors, axis=1, keepdims=True) + 1e-12)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(a_norm, b_norm.T)  # shape (a.k, b.k)
        
        # If threshold is 0.0 or very low (<= 0.1), use average similarity instead of threshold-based matching
        # This ensures we get meaningful scores even when threshold filtering is disabled
        if self.config.similarity_threshold <= 0.1:
            # Use average similarity as the semantic density when threshold is disabled
            # This gives a meaningful score for ranking while including all results
            semantic_density = float(np.mean(similarity_matrix))
            semantic_matches = int(np.sum(similarity_matrix > 0))  # Count positive similarities
        else:
            # Find semantic matches (similarity > threshold)
            semantic_matches = np.sum(similarity_matrix > self.config.similarity_threshold)
            # Calculate semantic density
            semantic_density = semantic_matches / min(a.k, b.k)
        
        return int(semantic_matches), float(semantic_density)
    
    def _maybe_populate_sketch_values(self, table_name: str, column_name: str, column_index: int,
                                     sketch: SemanticSketch) -> None:
        """Populate sketch.representative_names for older sketches (best-effort).
        
        Old offline sketches store only representative vectors, not the original values.
        If we have both offline embeddings and access to the datalake CSV, we can
        reconstruct representative values by matching representative vectors back
        to the full embedding matrix (exact byte match on float32 vectors).
        """
        if sketch.representative_names is not None:
            return
        if self.embeddings_dir is None or self.datalake_dir is None:
            return
        try:
            full_embeddings = load_column_embedding(table_name, column_name, column_index, self.embeddings_dir)
            if full_embeddings is None or len(full_embeddings) == 0:
                return
            raw_values = self._load_candidate_values(self.datalake_dir, table_name, column_name)
            if not raw_values:
                return
            # Normalize exactly like the embedding pipeline: strip/lower, drop empties, preserve order.
            clean_values = []
            for v in raw_values:
                norm = self._normalize_value(v)
                if norm:
                    clean_values.append(norm)
            if len(clean_values) != len(full_embeddings):
                # If alignment is off, we can't safely map embeddings -> values.
                return

            # Map embedding row bytes to indices (handle duplicates conservatively).
            by_bytes: Dict[bytes, List[int]] = {}
            for i in range(len(full_embeddings)):
                key = full_embeddings[i].tobytes()
                by_bytes.setdefault(key, []).append(i)

            rep_names: List[str] = []
            for rv in sketch.representative_vectors:
                k = rv.tobytes()
                idxs = by_bytes.get(k)
                if not idxs:
                    # Can't map this representative vector.
                    return
                idx = idxs.pop(0)
                rep_names.append(clean_values[idx])

            sketch.representative_names = rep_names
        except Exception:
            # Best-effort only; avoid breaking retrieval.
            return

    def _get_contributing_entities(self, query_sketch: SemanticSketch, candidate_sketch: SemanticSketch,
                                   top_n: int = 200) -> List[Tuple[str, str, float]]:
        """Get the value pairs that most strongly support the sketch match (retrieval evidence).
        
        Args:
            query_sketch: Query column sketch
            candidate_sketch: Candidate column sketch
            top_n: Number of highest-similarity sketch-vector pairs to return (default 200).
                   This is enforced as a 1-to-1 matching: each query value and each candidate
                   value appears in at most one returned pair.
            
        Returns:
            List of (query_value, candidate_value, similarity_score) tuples
        """
        if query_sketch.k == 0 or candidate_sketch.k == 0:
            return []
        
        # Need values to map back to entities
        # Query sketch should always have values (built on-the-fly), but candidate might not
        if query_sketch.representative_names is None:
            return []  # Can't map query values if not stored
        
        if candidate_sketch.representative_names is None:
            # Candidate sketch doesn't have values - can't extract contributing entities
            # This is expected for old sketches built before we added value storage
            return []
        
        # Calculate cosine similarity between all pairs of representative vectors
        a_norm = query_sketch.representative_vectors / (np.linalg.norm(query_sketch.representative_vectors, axis=1, keepdims=True) + 1e-12)
        b_norm = candidate_sketch.representative_vectors / (np.linalg.norm(candidate_sketch.representative_vectors, axis=1, keepdims=True) + 1e-12)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(a_norm, b_norm.T)  # shape (query.k, candidate.k)

        # Greedy 1-to-1 matching over high-similarity pairs.
        # We avoid sorting the full k*k matrix by progressively expanding a top-M candidate set.
        flat = similarity_matrix.ravel()
        if flat.size == 0:
            return []
        target = int(top_n) if top_n is not None else 0
        if target <= 0:
            return []
        target = min(target, min(query_sketch.k, candidate_sketch.k), flat.size)

        used_i: Set[int] = set()
        used_j: Set[int] = set()
        selected: List[Tuple[int, int, float]] = []

        m = min(flat.size, target * 50)  # initial candidate pool
        while True:
            idx = np.argpartition(flat, -m)[-m:]
            idx = idx[np.argsort(flat[idx])[::-1]]
            ii, jj = np.unravel_index(idx, similarity_matrix.shape)

            for i, j in zip(ii.tolist(), jj.tolist()):
                if i in used_i or j in used_j:
                    continue
                sim = float(similarity_matrix[i, j])
                used_i.add(i)
                used_j.add(j)
                selected.append((i, j, sim))
                if len(selected) >= target:
                    break

            if len(selected) >= target:
                break
            if m >= flat.size:
                break
            m = min(flat.size, m * 2)

        contributing_pairs: List[Tuple[str, str, float]] = []
        for i, j, sim in selected:
            contributing_pairs.append((
                query_sketch.representative_names[i],
                candidate_sketch.representative_names[j],
                sim
            ))
        return contributing_pairs
    
    def analyze_value_level_matches(self, query: QueryColumn, candidate_table: str, 
                                   candidate_column: str, datalake_dir: Optional[Path] = None,
                                   max_query_values: Optional[int] = None,
                                   max_candidate_values: Optional[int] = None,
                                   use_sketch_values: bool = True,
                                   match_threshold: Optional[float] = None) -> Tuple[List[ValueMatch], ValueMatchStats]:
        """Analyze value-level matches between query and candidate columns.
        
        Args:
            query: Query column to analyze
            candidate_table: Name of candidate table
            candidate_column: Name of candidate column
            datalake_dir: Path to datalake directory (required if use_sketch_values=False or for fallback)
            max_query_values: Maximum number of query values to analyze (None = all, ignored if use_sketch_values=True)
            max_candidate_values: Maximum number of candidate values to analyze (None = all, ignored if use_sketch_values=True)
            use_sketch_values: If True, use only values stored in sketches (recommended for evaluation)
            match_threshold: Override threshold for value-level matching only. If None, uses self.config.similarity_threshold.
            
        Returns:
            Tuple of (list of ValueMatch objects, ValueMatchStats)
        """
        # Initialize embedder if needed
        if self.embedder is None:
            self.embedder = create_mpnet_embedder(self.config.device)
        
        # Get query values from sketch if available, otherwise use query.values
        query_values_to_analyze = None
        if use_sketch_values:
            # Try to get values from query sketch (if it was built and stored)
            # For now, we'll rebuild the query sketch to get its representative values
            query_sketch = self._build_query_sketch(query)
            if query_sketch and query_sketch.representative_names:
                query_values_to_analyze = query_sketch.representative_names
                self.logger.info(f"Using {len(query_values_to_analyze)} values from query sketch")
        
        # Fallback to query.values if sketch values not available
        if query_values_to_analyze is None:
            query_values_to_analyze = query.values
            if max_query_values and len(query_values_to_analyze) > max_query_values:
                import random
                random.seed(42)
                query_values_to_analyze = random.sample(query_values_to_analyze, max_query_values)
                self.logger.info(f"Sampled {len(query_values_to_analyze)} query values from {len(query.values)}")
        
        # Get candidate values from sketch if available
        candidate_values = None
        if use_sketch_values:
            # Load candidate sketch to get its representative values
            candidate_sketch = None
            # Find the sketch file (try column_index 0 first, then iterate if needed)
            for column_index in range(10):  # Try up to 10 column indices
                candidate_sketch = load_offline_sketch(candidate_table, candidate_column, column_index, self.sketches_dir)
                if candidate_sketch is not None:
                    break
            
            if candidate_sketch and candidate_sketch.representative_names:
                candidate_values = candidate_sketch.representative_names
                self.logger.info(f"Using {len(candidate_values)} values from candidate sketch")
        
        # Fallback to loading from CSV if sketch values not available
        if candidate_values is None:
            if datalake_dir is None:
                raise ValueError("datalake_dir is required when sketch values are not available")
            
            candidate_values = self._load_candidate_values(datalake_dir, candidate_table, candidate_column)
            if not candidate_values:
                self.logger.warning(f"No candidate values found for {candidate_table}.{candidate_column} (file may not exist or column not found)")
                return [], ValueMatchStats(
                    total_query_values=len(query_values_to_analyze),
                    total_candidate_values=0,
                    query_values_with_matches=0,
                    candidate_values_with_matches=0,
                    total_matches=0,
                    avg_matches_per_query_value=0.0,
                    max_matches_per_query_value=0,
                    min_matches_per_query_value=0,
                    one_to_one_matches=0,
                    one_to_many_matches=0,
                    query_values_with_no_matches=len(query_values_to_analyze),
                    candidate_values_with_no_matches=0
                )
            
            # When use_sketch_values=True but sketch values aren't available, limit to sketch size
            # to approximate what the sketch would contain
            if use_sketch_values and len(candidate_values) > self.config.sketch_size:
                import random
                random.seed(42)
                candidate_values = random.sample(candidate_values, self.config.sketch_size)
                self.logger.info(f"Limited candidate values to sketch size ({self.config.sketch_size}) since sketch values not available")
            elif max_candidate_values and len(candidate_values) > max_candidate_values:
                import random
                random.seed(42)
                candidate_values = random.sample(candidate_values, max_candidate_values)
                self.logger.info(f"Sampled {len(candidate_values)} candidate values from original set")
        
        # Normalize values, but keep mapping back to raw strings
        # If values come from sketches, they're already normalized, but we still need to process them
        normalized_query_values = []
        raw_query_values = []
        for v in query_values_to_analyze:
            # If value is already normalized (from sketch), use it directly
            # Otherwise normalize it
            if use_sketch_values and isinstance(v, str):
                # Values from sketch are already normalized
                norm = v
                raw = v  # For sketch values, raw and normalized are the same
            else:
                norm = self._normalize_value(v)
                raw = str(v)
            
            if norm:
                normalized_query_values.append(norm)
                raw_query_values.append(raw)
        
        normalized_candidate_values = []
        raw_candidate_values = []
        for v in candidate_values:
            # If value is already normalized (from sketch), use it directly
            if use_sketch_values and isinstance(v, str):
                # Values from sketch are already normalized
                norm = v
                raw = v  # For sketch values, raw and normalized are the same
            else:
                norm = self._normalize_value(v)
                raw = str(v)
            
            if norm:
                normalized_candidate_values.append(norm)
                raw_candidate_values.append(raw)
        
        # Filtered/clean values (normalized) are used for embeddings,
        # but ValueMatch will carry the original raw strings (or normalized if from sketch).
        clean_query_values = normalized_query_values
        clean_candidate_values = normalized_candidate_values
        
        if not clean_query_values or not clean_candidate_values:
            return [], ValueMatchStats(
                total_query_values=len(clean_query_values),
                total_candidate_values=len(clean_candidate_values),
                query_values_with_matches=0,
                candidate_values_with_matches=0,
                total_matches=0,
                avg_matches_per_query_value=0.0,
                max_matches_per_query_value=0,
                min_matches_per_query_value=0,
                one_to_one_matches=0,
                one_to_many_matches=0,
                query_values_with_no_matches=len(clean_query_values),
                candidate_values_with_no_matches=len(clean_candidate_values)
            )
        
        # Generate embeddings with column context
        query_embeddings = self._embed_values_with_context(clean_query_values, query.column_name)
        candidate_embeddings = self._embed_values_with_context(clean_candidate_values, candidate_column)
        
        # Compute similarity matrix
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)
        candidate_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-12)
        similarity_matrix = np.dot(query_norm, candidate_norm.T)  # shape (num_query, num_candidate)
        
        # Find all matches above threshold
        threshold = float(match_threshold) if match_threshold is not None else float(self.config.similarity_threshold)
        matches = []
        query_value_match_counts = {}  # query_value -> count of matches
        candidate_value_match_counts = {}  # candidate_value -> count of matches
        
        # Track max similarity for debugging
        max_similarity = 0.0
        max_similarity_pair = None
        
        for i, q_val in enumerate(clean_query_values):
            query_value_match_counts[q_val] = 0
            for j, c_val in enumerate(clean_candidate_values):
                similarity = float(similarity_matrix[i, j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_similarity_pair = (q_val, c_val)
                if similarity >= threshold:
                    matches.append(ValueMatch(
                        query_value=raw_query_values[i],
                        candidate_value=raw_candidate_values[j],
                        similarity_score=similarity
                    ))
                    query_value_match_counts[q_val] += 1
                    if c_val not in candidate_value_match_counts:
                        candidate_value_match_counts[c_val] = 0
                    candidate_value_match_counts[c_val] += 1
        
        # Debug: Log if no matches found but we have values
        if len(matches) == 0 and len(clean_query_values) > 0 and len(clean_candidate_values) > 0:
            self.logger.debug(f"No matches above threshold {threshold:.3f} for {candidate_table}.{candidate_column}. Max similarity: {max_similarity:.3f} ({max_similarity_pair[0] if max_similarity_pair else 'N/A'} -> {max_similarity_pair[1] if max_similarity_pair else 'N/A'})")
        
        # Calculate statistics
        match_counts = list(query_value_match_counts.values())
        one_to_one = sum(1 for count in match_counts if count == 1)
        one_to_many = sum(1 for count in match_counts if count > 1)
        no_matches = sum(1 for count in match_counts if count == 0)
        
        avg_matches = np.mean(match_counts) if match_counts else 0.0
        max_matches = max(match_counts) if match_counts else 0
        min_matches = min(match_counts) if match_counts else 0
        
        stats = ValueMatchStats(
            total_query_values=len(clean_query_values),
            total_candidate_values=len(clean_candidate_values),
            query_values_with_matches=len(clean_query_values) - no_matches,
            candidate_values_with_matches=len(candidate_value_match_counts),
            total_matches=len(matches),
            avg_matches_per_query_value=float(avg_matches),
            max_matches_per_query_value=int(max_matches),
            min_matches_per_query_value=int(min_matches),
            one_to_one_matches=one_to_one,
            one_to_many_matches=one_to_many,
            query_values_with_no_matches=no_matches,
            candidate_values_with_no_matches=len(clean_candidate_values) - len(candidate_value_match_counts)
        )
        
        return matches, stats
    
    def _load_candidate_values(self, datalake_dir: Path, table_name: str, column_name: str) -> List[str]:
        """Load candidate values from datalake.
        
        Tries multiple filename variations to handle case sensitivity and naming differences.
        """
        # Try multiple filename variations
        # Handle AutoFJ naming: files are TitleCase like "FootballLeagueSeason_left.csv"
        # but table names might be lowercase like "footballleagueseason_left"
        def to_title_case_preserve_underscores(name: str) -> str:
            """Convert 'footballleagueseason_left' to 'FootballLeagueSeason_left' (preserve case after underscore)"""
            if '_' in name:
                parts = name.split('_')
                # Capitalize first part, keep rest as-is (for _left, _right suffixes)
                return f"{parts[0].capitalize()}_{'_'.join(parts[1:])}"
            return name.capitalize()
        
        def to_proper_title_case(name: str) -> str:
            """Convert compound words like 'televisionstation' to 'TelevisionStation'.
            Attempts to detect word boundaries by capitalizing after common patterns."""
            if '_' in name:
                parts = name.split('_')
                # Process first part for compound words
                first_part = parts[0]
                # Common word boundaries in compound words (heuristic approach)
                # Try to detect where words might start (after common prefixes/words)
                words = []
                remaining = first_part.lower()
                # Common word endings that suggest a new word starts
                word_endings = ['station', 'season', 'league', 'tournament', 'match', 'team', 
                              'building', 'facility', 'party', 'agency', 'event', 'name']
                for ending in word_endings:
                    if remaining.endswith(ending) and len(remaining) > len(ending):
                        prefix = remaining[:-len(ending)]
                        if prefix:
                            words.append(prefix.capitalize())
                        words.append(ending.capitalize())
                        remaining = ''
                        break
                if not words:
                    # Fallback: try to split on common patterns
                    # Look for patterns like "televisionstation" -> "television" + "station"
                    import re
                    # Try to split on common word boundaries
                    matches = re.findall(r'[a-z]+', first_part.lower())
                    if len(matches) > 1:
                        words = [w.capitalize() for w in matches]
                    else:
                        words = [first_part.capitalize()]
                result = ''.join(words) if words else first_part.capitalize()
                return f"{result}_{'_'.join(parts[1:])}"
            return name.capitalize()
        
        filename_variations = [
            f"{table_name}.csv",  # Try original case first (for TitleCase files)
            table_name,  # In case table_name already includes .csv
            f"{table_name.title()}.csv",  # Televisionstation_left -> Televisionstation_Left
            f"{to_proper_title_case(table_name)}.csv",  # televisionstation_left -> TelevisionStation_left
            f"{to_title_case_preserve_underscores(table_name)}.csv",  # FootballLeagueSeason_left.csv
            f"{table_name.lower()}.csv",
            f"{table_name.upper()}.csv",
        ]
        
        csv_file = None
        for filename_var in filename_variations:
            candidate_path = datalake_dir / filename_var
            if candidate_path.exists():
                csv_file = candidate_path
                break
        
        # If still not found, try case-insensitive search
        if csv_file is None and datalake_dir.exists():
            table_lower = table_name.lower().replace('.csv', '')
            for file_path in datalake_dir.glob("*.csv"):
                if file_path.stem.lower() == table_lower:
                    csv_file = file_path
                    break
        
        if csv_file is None:
            # File not found - return empty list (warning will be logged by caller)
            return []
        
        try:
            df = pd.read_csv(csv_file)
            
            # Try exact column name match first
            if column_name in df.columns:
                values = df[column_name].dropna().astype(str).tolist()
                return values
            
            # Try case-insensitive column matching
            column_lower = column_name.lower()
            for col in df.columns:
                if col.lower() == column_lower:
                    values = df[col].dropna().astype(str).tolist()
                    return values
            
            # Column not found
            return []
            
        except Exception as e:
            self.logger.warning(f"Error loading candidate values from {table_name}.{column_name} (file: {csv_file}): {e}")
            return []
    
    def print_stats(self) -> None:
        """Print query processing statistics."""
        stats = self.stats
        print(f"\n=== QUERY PROCESSING STATISTICS ===")
        print(f"Total queries: {stats.total_queries}")
        print(f"Successful queries: {stats.successful_queries}")
        print(f"Failed queries: {stats.failed_queries}")
        print(f"Total processing time: {stats.total_processing_time:.2f}s")
        print(f"Total candidates found: {stats.total_candidates_found}")
        print(f"Total high-quality candidates: {stats.total_high_quality_candidates}")
        
        if stats.successful_queries > 0:
            avg_time_per_query = stats.total_processing_time / stats.successful_queries
            avg_candidates_per_query = stats.total_candidates_found / stats.successful_queries
            avg_high_quality_per_query = stats.total_high_quality_candidates / stats.successful_queries
            print(f"Average time per query: {avg_time_per_query:.4f}s")
            print(f"Average candidates per query: {avg_candidates_per_query:.2f}")
            print(f"Average high-quality candidates per query: {avg_high_quality_per_query:.2f}")

# =============================================================================
# Utility Functions
# =============================================================================

def load_query_values_from_file(file_path: Path, column_name: str) -> List[str]:
    """Load values from a CSV file for a specific column."""
    try:
        df = pd.read_csv(file_path)
        if column_name in df.columns:
            return df[column_name].dropna().astype(str).tolist()
        else:
            print(f"Column '{column_name}' not found in {file_path}")
            return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def save_contributing_entities(contributing_entities: List[Tuple[str, str, float]],
                                output_file: Path, query_table: str, query_column: str,
                                candidate_table: str, candidate_column: str) -> None:
    """Save contributing entities (sketch values that led to column match) to CSV file.
    
    Args:
        contributing_entities: List of (query_value, candidate_value, similarity_score) tuples
        output_file: Path to output CSV file (without extension)
        query_table: Query table name
        query_column: Query column name
        candidate_table: Candidate table name
        candidate_column: Candidate column name
    """
    if not contributing_entities:
        return
    
    # Save contributing entities
    entities_data = []
    for query_val, candidate_val, similarity in contributing_entities:
        entities_data.append({
            'query_table': query_table,
            'query_column': query_column,
            'query_value': query_val,
            'candidate_table': candidate_table,
            'candidate_column': candidate_column,
            'candidate_value': candidate_val,
            'similarity_score': similarity
        })
    
    entities_df = pd.DataFrame(entities_data)
    csv_file = Path(str(output_file) + "_contributing_entities.csv")
    entities_df.to_csv(csv_file, index=False)
    print(f"  Saved {len(contributing_entities)} contributing entities to {csv_file}")


def save_value_matches(matches: List[ValueMatch], stats: ValueMatchStats, 
                       output_file: Path, query_table: str, query_column: str,
                       candidate_table: str, candidate_column: str,
                       format: str = "csv") -> None:
    """Save value-level match analysis to file.
    
    Args:
        matches: List of value matches
        stats: Value match statistics
        output_file: Path to output file (without extension)
        query_table: Query table name
        query_column: Query column name
        candidate_table: Candidate table name
        candidate_column: Candidate column name
        format: Output format - "csv", "parquet", or "both" (default: "csv")
    """
    # Save matches
    matches_data = []
    for match in matches:
        matches_data.append({
            'query_table': query_table,
            'query_column': query_column,
            'query_value': match.query_value,
            'candidate_table': candidate_table,
            'candidate_column': candidate_column,
            'candidate_value': match.candidate_value,
            'similarity_score': match.similarity_score
        })
    
    matches_df = pd.DataFrame(matches_data)
    
    # Save statistics
    stats_data = [{
        'query_table': query_table,
        'query_column': query_column,
        'candidate_table': candidate_table,
        'candidate_column': candidate_column,
        'total_query_values': stats.total_query_values,
        'total_candidate_values': stats.total_candidate_values,
        'query_values_with_matches': stats.query_values_with_matches,
        'candidate_values_with_matches': stats.candidate_values_with_matches,
        'total_matches': stats.total_matches,
        'avg_matches_per_query_value': stats.avg_matches_per_query_value,
        'max_matches_per_query_value': stats.max_matches_per_query_value,
        'min_matches_per_query_value': stats.min_matches_per_query_value,
        'one_to_one_matches': stats.one_to_one_matches,
        'one_to_many_matches': stats.one_to_many_matches,
        'query_values_with_no_matches': stats.query_values_with_no_matches,
        'candidate_values_with_no_matches': stats.candidate_values_with_no_matches
    }]
    stats_df = pd.DataFrame(stats_data)
    
    # Save to files based on format
    matches_file = output_file.parent / f"{output_file.stem}_matches"
    stats_file = output_file.parent / f"{output_file.stem}_stats"
    
    if format in ("csv", "both"):
        matches_df.to_csv(f"{matches_file}.csv", index=False)
        stats_df.to_csv(f"{stats_file}.csv", index=False)
        print(f"Saved {len(matches)} value matches to {matches_file}.csv")
        print(f"Saved statistics to {stats_file}.csv")
    
    if format in ("parquet", "both"):
        try:
            matches_df.to_parquet(f"{matches_file}.parquet", index=False, engine='pyarrow')
            stats_df.to_parquet(f"{stats_file}.parquet", index=False, engine='pyarrow')
            print(f"Saved {len(matches)} value matches to {matches_file}.parquet")
            print(f"Saved statistics to {stats_file}.parquet")
        except ImportError:
            print("Warning: pyarrow not available, skipping Parquet format. Install with: pip install pyarrow")
        except Exception as e:
            print(f"Warning: Could not save Parquet format: {e}")


def save_entity_linking_results(all_matches: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                                all_ground_truth: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                                pair_stats: Dict[Tuple[str, str, str, str], Dict],
                                output_dir: Path,
                                format: str = "csv") -> None:
    """Save entity linking evaluation results in a structured format.
    
    This function saves detailed entity linking results including:
    - All predicted value matches
    - All ground truth matches
    - Per-pair statistics
    - Evaluation metrics summary
    
    Args:
        all_matches: List of predicted matches as ((query_table, query_value), (candidate_table, candidate_value))
        all_ground_truth: List of ground truth matches in same format
        pair_stats: Dictionary mapping (query_table, query_column, candidate_table, candidate_column) -> stats dict
        output_dir: Directory to save results
        format: Output format - "csv", "parquet", or "both" (default: "csv" for universal compatibility)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert matches to DataFrames
    predicted_matches_data = []
    for (q_table, q_val), (c_table, c_val) in all_matches:
        predicted_matches_data.append({
            'query_table': q_table,
            'query_value': q_val,
            'candidate_table': c_table,
            'candidate_value': c_val
        })
    
    gt_matches_data = []
    for (q_table, q_val), (c_table, c_val) in all_ground_truth:
        gt_matches_data.append({
            'query_table': q_table,
            'query_value': q_val,
            'candidate_table': c_table,
            'candidate_value': c_val
        })
    
    predicted_df = pd.DataFrame(predicted_matches_data)
    gt_df = pd.DataFrame(gt_matches_data)
    
    # Convert pair stats to DataFrame
    pair_stats_data = []
    for (q_table, q_col, c_table, c_col), stats in pair_stats.items():
        row = {
            'query_table': q_table,
            'query_column': q_col,
            'candidate_table': c_table,
            'candidate_column': c_col
        }
        row.update(stats)
        pair_stats_data.append(row)
    
    pair_stats_df = pd.DataFrame(pair_stats_data) if pair_stats_data else pd.DataFrame()
    
    # Save files
    if format in ("csv", "both"):
        predicted_df.to_csv(output_dir / "predicted_matches.csv", index=False)
        gt_df.to_csv(output_dir / "ground_truth_matches.csv", index=False)
        if not pair_stats_df.empty:
            pair_stats_df.to_csv(output_dir / "pair_statistics.csv", index=False)
        print(f"Saved entity linking results to CSV in {output_dir}")
    
    if format in ("parquet", "both"):
        try:
            predicted_df.to_parquet(output_dir / "predicted_matches.parquet", index=False, engine='pyarrow')
            gt_df.to_parquet(output_dir / "ground_truth_matches.parquet", index=False, engine='pyarrow')
            if not pair_stats_df.empty:
                pair_stats_df.to_parquet(output_dir / "pair_statistics.parquet", index=False, engine='pyarrow')
            print(f"Saved entity linking results to Parquet in {output_dir}")
        except ImportError:
            print("Warning: pyarrow not available, skipping Parquet format. Install with: pip install pyarrow")
        except Exception as e:
            print(f"Warning: Could not save Parquet format: {e}")
    
    # Print summary
    print(f"\nEntity Linking Results Summary:")
    print(f"  Predicted matches: {len(predicted_df)}")
    print(f"  Ground truth matches: {len(gt_df)}")
    print(f"  Column pairs analyzed: {len(pair_stats_df)}")

def save_query_results(results: List[QueryResult], output_file: Path, 
                       query_sample_values: List[str] = None,
                       datalake_dir: Path = None) -> None:
    """Save query results to CSV file.
    
    Args:
        results: List of query results to save
        output_file: Path to output CSV file
        query_sample_values: First 5 values from query column (optional)
        datalake_dir: Path to datalake directory for loading candidate values (optional)
    """
    if not results:
        print("No results to save")
        return
    
    # Get first 5 query values
    query_sample = query_sample_values[:5] if query_sample_values else []
    query_sample_str = ", ".join(str(v) for v in query_sample) if query_sample else ""
    
    # Convert results to DataFrame
    data = []
    for result in results:
        # Get first 5 candidate values if datalake_dir is provided
        candidate_sample_str = ""
        if datalake_dir:
            try:
                # Try with .csv extension first
                csv_file = datalake_dir / f"{result.candidate_table}.csv"
                if not csv_file.exists() and result.candidate_table.endswith('.csv'):
                    # Try without .csv extension if table_name already includes it
                    csv_file = datalake_dir / result.candidate_table
                
                if csv_file.exists():
                    candidate_values = load_query_values_from_file(csv_file, result.candidate_column)
                    candidate_sample = candidate_values[:5] if candidate_values else []
                    candidate_sample_str = ", ".join(str(v) for v in candidate_sample) if candidate_sample else ""
            except Exception:
                # If loading fails, just leave candidate_sample_str empty
                pass
        
        # Format contributing entities if available
        contributing_entities_str = ""
        if result.contributing_entities:
            # Show top 10 contributing entity pairs
            top_contributors = result.contributing_entities[:10]
            pairs = [f"{q_val}↔{c_val}({sim:.2f})" for q_val, c_val, sim in top_contributors]
            contributing_entities_str = " | ".join(pairs)
        
        data.append({
            'candidate_table': result.candidate_table,
            'candidate_column': result.candidate_column,
            'similarity_score': result.similarity_score,
            'query_sample_values': query_sample_str,
            'candidate_sample_values': candidate_sample_str,
            'contributing_entities': contributing_entities_str,
            'num_contributing_entities': len(result.contributing_entities) if result.contributing_entities else 0
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(results)} results to {output_file}")
