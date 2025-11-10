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

# Import from other modules
from offline_embedding import MPNetEmbedder, create_mpnet_embedder, load_column_embedding
from offline_sketch import SemanticSketch, load_offline_sketch

# Import DeepJoin retriever
try:
    from deepjoin_retriever import DeepJoinIndexRetriever, create_deepjoin_retriever
except ImportError:
    DeepJoinIndexRetriever = None
    create_deepjoin_retriever = None

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
    
    # DeepJoin integration options
    use_deepjoin_index: bool = False  # Use DeepJoin for candidate filtering
    deepjoin_embeddings_path: Optional[str] = None  # Path to DeepJoin embeddings
    deepjoin_query_embeddings_path: Optional[str] = None  # Path to DeepJoin query embeddings
    deepjoin_index_path: Optional[str] = None  # Path to DeepJoin HNSW index
    deepjoin_scale: float = 1.0  # Scale factor for DeepJoin dataset
    deepjoin_encoder: str = "sherlock"  # DeepJoin encoder type
    deepjoin_candidate_limit: int = 5  # Number of candidates from DeepJoin index
    deepjoin_top_k: int = 200  # Number of top results from DeepJoin (for candidate filtering)
    deepjoin_threshold: float = 0.6  # DeepJoin similarity threshold
    
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
    semantic_matches: int
    semantic_density: float

@dataclass
class QueryStats:
    """Statistics for query processing."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    total_candidates_found: int = 0
    total_high_quality_candidates: int = 0
    total_deepjoin_candidates: int = 0
    deepjoin_queries_processed: int = 0

# =============================================================================
# Core Classes
# =============================================================================

class SemanticJoinQueryProcessor:
    """Processes semantic join queries using pre-built sketches."""
    
    def __init__(self, config: QueryConfig, sketches_dir: Path, embeddings_dir: Optional[Path] = None):
        self.config = config
        self.sketches_dir = sketches_dir
        self.embeddings_dir = embeddings_dir
        self.stats = QueryStats()
        self.embedder = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize DeepJoin retriever if enabled
        self.deepjoin_retriever = None
        self.deepjoin_query_embeddings = None
        if config.use_deepjoin_index:
            self._initialize_deepjoin_retriever()
            self._load_deepjoin_query_embeddings()
        
        # Load all available sketches for fast lookup
        self.available_sketches = self._discover_available_sketches()
        
        self.logger.info(f"Found {len(self.available_sketches)} available sketches")
    
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
    
    def _initialize_deepjoin_retriever(self):
        """Initialize the DeepJoin retriever if enabled."""
        if not self.config.use_deepjoin_index:
            return
        
        if DeepJoinIndexRetriever is None:
            self.logger.warning("DeepJoin integration requested but DeepJoin modules not available")
            return
        
        if not self.config.deepjoin_embeddings_path:
            self.logger.error("DeepJoin integration enabled but no embeddings path provided")
            return
        
        try:
            self.deepjoin_retriever = create_deepjoin_retriever(
                deepjoin_embeddings_path=self.config.deepjoin_embeddings_path,
                index_path=self.config.deepjoin_index_path,
                scale=self.config.deepjoin_scale,
                encoder=self.config.deepjoin_encoder
            )
            
            if self.deepjoin_retriever:
                self.logger.info("DeepJoin retriever initialized successfully")
            else:
                self.logger.error("Failed to initialize DeepJoin retriever")
                
        except Exception as e:
            self.logger.error(f"Error initializing DeepJoin retriever: {e}")
            self.deepjoin_retriever = None
    
    def _load_deepjoin_query_embeddings(self):
        """Load DeepJoin query embeddings if available."""
        if not self.config.deepjoin_query_embeddings_path:
            self.logger.warning("DeepJoin query embeddings path not provided")
            return
        
        try:
            import pickle
            with open(self.config.deepjoin_query_embeddings_path, 'rb') as f:
                self.deepjoin_query_embeddings = pickle.load(f)
            self.logger.info(f"Loaded DeepJoin query embeddings from {self.config.deepjoin_query_embeddings_path}")
        except Exception as e:
            self.logger.error(f"Error loading DeepJoin query embeddings: {e}")
            self.deepjoin_query_embeddings = None
    
    def _discover_available_sketches(self) -> Dict[Tuple[str, str, int], Path]:
        """Discover all available sketches."""
        sketches = {}
        
        if not self.sketches_dir.exists():
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
        
        return sketches
    
    def process_query(self, query: QueryColumn) -> List[QueryResult]:
        """Process a semantic join query."""
        start_time = time.time()
        self.stats.total_queries += 1
        
        try:
            # Use DeepJoin for candidate filtering if enabled
            candidate_tables = None
            if self.deepjoin_retriever and self.deepjoin_query_embeddings:
                candidate_tables = self._get_deepjoin_candidates(query)
            
            # Build query sketch
            query_sketch = self._build_query_sketch(query)
            if query_sketch is None:
                self.stats.failed_queries += 1
                return []
            
            # Find similar columns using sketch comparison
            results = self._find_similar_columns(query_sketch, query, candidate_tables)
            
            # Sort by similarity score and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            # Return up to top_k_return high-quality results, or fewer if not enough meet threshold
            final_results = results[:self.config.top_k_return]
            
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
    
    def _get_deepjoin_candidates(self, query: QueryColumn) -> Optional[Set[str]]:
        """Get candidate tables using DeepJoin query embeddings."""
        if not self.deepjoin_query_embeddings:
            return None
        
        try:
            # Find query embeddings for this specific query
            query_key = f"{query.table_name}.{query.column_name}"
            query_embeddings = None
            
            # Search for matching query embeddings
            # DeepJoin embeddings use "datalake-" prefix and no .csv extension
            # Query table names may include .csv extension
            query_table_base = query.table_name.replace('.csv', '')
            
            for table_name, embeddings in self.deepjoin_query_embeddings:
                # Try exact match first
                if table_name == query.table_name:
                    query_embeddings = embeddings
                    break
                # Try with datalake- prefix
                elif table_name == f"datalake-{query.table_name}":
                    query_embeddings = embeddings
                    break
                # Try removing datalake- prefix from embeddings
                elif table_name.startswith("datalake-") and table_name[9:] == query.table_name:
                    query_embeddings = embeddings
                    break
                # Try with datalake- prefix and without .csv extension
                elif table_name == f"datalake-{query_table_base}":
                    query_embeddings = embeddings
                    break
                # Try removing datalake- prefix and comparing without .csv extension
                elif table_name.startswith("datalake-") and table_name[9:] == query_table_base:
                    query_embeddings = embeddings
                    break
            
            if query_embeddings is None:
                self.logger.warning(f"No DeepJoin query embeddings found for {query_key}")
                # Debug: show available table names
                available_tables = [name for name, _ in self.deepjoin_query_embeddings[:5]]
                self.logger.debug(f"Available DeepJoin tables (first 5): {available_tables}")
                return None
            
            # Use DeepJoin to find similar tables
            similar_columns = self.deepjoin_retriever.find_similar_columns(
                query_embeddings=query_embeddings,
                top_k=self.config.deepjoin_top_k,  # Use separate DeepJoin top-k for more candidates
                candidate_limit=self.config.deepjoin_candidate_limit,
                threshold=self.config.deepjoin_threshold
            )
            
            # Extract unique table names
            candidate_tables = set(table_name for table_name, _ in similar_columns)
            num_candidates = len(candidate_tables)
            
            # Track DeepJoin statistics
            self.stats.total_deepjoin_candidates += num_candidates
            self.stats.deepjoin_queries_processed += 1
            
            self.logger.info(f"DeepJoin found {num_candidates} candidate tables for {query_key}")
            
            return candidate_tables
            
        except Exception as e:
            self.logger.error(f"Error getting DeepJoin candidates: {e}")
            return None
    
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
            if self.config.enable_large_table_sampling and len(clean_values) > self.config.large_table_sample_size:
                import random
                random.seed(42)
                clean_values = random.sample(clean_values, self.config.large_table_sample_size)
                self.logger.info(f"Sampled {len(clean_values)} values from {len(query.values)} for query sketch")
            
            # Generate embeddings with column context
            embeddings = self._embed_values_with_context(clean_values, query.column_name)
            
            if len(embeddings) == 0:
                return None
            
            # Build semantic sketch using k-closest to origin
            sketch = self._build_semantic_sketch_from_embeddings(embeddings)
            
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
        
        # Select representatives using k-closest to origin
        closest_indices, distances = self._k_closest_to_origin(embeddings, self.config.sketch_size)
        
        # Extract representative vectors and calculate distances to origin
        representative_vectors = embeddings[closest_indices]
        representative_ids = list(range(len(closest_indices)))  # Simple ID assignment
        
        # Calculate distances to origin for all representatives
        distances_to_origin = np.linalg.norm(representative_vectors, axis=1)
        
        return SemanticSketch(
            representative_vectors=representative_vectors,
            representative_ids=representative_ids,
            distances_to_origin=distances,
            embedding_dim=embeddings.shape[1],
            k=len(closest_indices),
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
    
    def _find_similar_columns(self, query_sketch: SemanticSketch, query: QueryColumn, candidate_tables: Optional[Set[str]] = None) -> List[QueryResult]:
        """Find similar columns using sketch comparison."""
        results = []
        
        # Filter sketches by candidate tables if provided
        sketches_to_process = self.available_sketches
        if candidate_tables:
            sketches_to_process = {
                (table_name, column_name, column_index): sketch_path
                for (table_name, column_name, column_index), sketch_path in self.available_sketches.items()
                if table_name in candidate_tables
            }
            self.logger.info(f"DeepJoin filtering: comparing against {len(sketches_to_process)} candidate sketches (filtered from {len(self.available_sketches)})")
        else:
            self.logger.info(f"Comparing query sketch against {len(self.available_sketches)} candidate sketches")
        
        for (table_name, column_name, column_index), sketch_path in sketches_to_process.items():
            try:
                # Skip the query table itself
                if table_name == query.table_name:
                    continue
                
                # Load candidate sketch
                candidate_sketch = load_offline_sketch(table_name, column_name, column_index, self.sketches_dir)
                if candidate_sketch is None:
                    continue
                
                # Compute semantic joinability
                semantic_matches, semantic_density = self._estimate_semantic_joinability(
                    query_sketch, candidate_sketch
                )
                
                # Only include results above threshold
                if semantic_density >= self.config.similarity_threshold:
                    result = QueryResult(
                        candidate_table=table_name,
                        candidate_column=column_name,
                        similarity_score=float(semantic_density),
                        semantic_matches=int(semantic_matches),
                        semantic_density=float(semantic_density)
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
        
        # Find semantic matches (similarity > threshold)
        semantic_matches = np.sum(similarity_matrix > self.config.similarity_threshold)
        
        # Calculate semantic density
        semantic_density = semantic_matches / min(a.k, b.k)
        
        return int(semantic_matches), float(semantic_density)
    
    def get_stats(self) -> QueryStats:
        """Get current query processing statistics."""
        return self.stats
    
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
            
        # DeepJoin statistics
        if stats.deepjoin_queries_processed > 0:
            avg_deepjoin_candidates = stats.total_deepjoin_candidates / stats.deepjoin_queries_processed
            print(f"DeepJoin queries processed: {stats.deepjoin_queries_processed}")
            print(f"Total DeepJoin candidates: {stats.total_deepjoin_candidates}")
            print(f"Average DeepJoin candidates per query: {avg_deepjoin_candidates:.2f}")
        else:
            print("DeepJoin integration: Not used")

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

def save_query_results(results: List[QueryResult], output_file: Path) -> None:
    """Save query results to CSV file."""
    if not results:
        print("No results to save")
        return
    
    # Convert results to DataFrame
    data = []
    for result in results:
        data.append({
            'candidate_table': result.candidate_table,
            'candidate_column': result.candidate_column,
            'similarity_score': result.similarity_score,
            'semantic_matches': result.semantic_matches,
            'semantic_density': result.semantic_density
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(results)} results to {output_file}")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Process semantic join queries")
    
    parser.add_argument("query_file", type=str, help="Path to query CSV file")
    parser.add_argument("query_column", type=str, help="Name of query column")
    parser.add_argument("sketches_dir", type=str, help="Path to sketches directory")
    parser.add_argument("--output-file", type=str, default="query_results.csv",
                       help="Output file for results")
    parser.add_argument("--top-k-return", type=int, default=10,
                       help="Number of final results to return")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                       help="Similarity threshold for semantic matching")
    parser.add_argument("--sketch-size", type=int, default=1024,
                       help="Number of representative vectors per sketch")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for MPNet model (auto, cpu, cuda, mps)")
    parser.add_argument("--embeddings-dir", type=str,
                       help="Path to embeddings directory (optional)")
    
    # DeepJoin integration arguments
    parser.add_argument("--use-deepjoin-index", action="store_true",
                       help="Use DeepJoin index for candidate filtering")
    parser.add_argument("--deepjoin-embeddings-path", type=str,
                       help="Path to DeepJoin embeddings pickle file")
    parser.add_argument("--deepjoin-query-embeddings-path", type=str,
                       help="Path to DeepJoin query embeddings pickle file")
    parser.add_argument("--deepjoin-index-path", type=str,
                       help="Path to DeepJoin HNSW index file (optional)")
    parser.add_argument("--deepjoin-scale", type=float, default=1.0,
                       help="Scale factor for DeepJoin dataset (0.0-1.0)")
    parser.add_argument("--deepjoin-encoder", type=str, default="sherlock",
                       choices=["sherlock", "sato"],
                       help="DeepJoin encoder type")
    parser.add_argument("--deepjoin-candidate-limit", type=int, default=5,
                       help="Number of candidates from DeepJoin index")
    parser.add_argument("--deepjoin-top-k", type=int, default=200,
                       help="Number of top results from DeepJoin (for candidate filtering)")
    parser.add_argument("--deepjoin-threshold", type=float, default=0.6,
                       help="DeepJoin similarity threshold")
    
    args = parser.parse_args()
    
    # Create config
    config = QueryConfig(
        top_k_return=args.top_k_return,
        similarity_threshold=args.similarity_threshold,
        sketch_size=args.sketch_size,
        device=args.device,
        use_deepjoin_index=args.use_deepjoin_index,
        deepjoin_embeddings_path=args.deepjoin_embeddings_path,
        deepjoin_query_embeddings_path=args.deepjoin_query_embeddings_path,
        deepjoin_index_path=args.deepjoin_index_path,
        deepjoin_scale=args.deepjoin_scale,
        deepjoin_encoder=args.deepjoin_encoder,
        deepjoin_candidate_limit=args.deepjoin_candidate_limit,
        deepjoin_top_k=args.deepjoin_top_k,
        deepjoin_threshold=args.deepjoin_threshold
    )
    
    # Load query values
    query_file = Path(args.query_file)
    if not query_file.exists():
        print(f"Error: Query file {query_file} does not exist")
        return 1
    
    query_values = load_query_values_from_file(query_file, args.query_column)
    if not query_values:
        print(f"Error: No values found in column '{args.query_column}'")
        return 1
    
    print(f"Loaded {len(query_values)} values from {args.query_column}")
    
    # Create query column
    query = QueryColumn(
        table_name=query_file.stem,
        column_name=args.query_column,
        values=query_values
    )
    
    # Initialize processor
    sketches_dir = Path(args.sketches_dir)
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else None
    
    processor = SemanticJoinQueryProcessor(config, sketches_dir, embeddings_dir)
    
    # Process query
    print(f"Processing query for {query.table_name}.{query.column_name}...")
    results = processor.process_query(query)
    
    # Save results
    output_file = Path(args.output_file)
    save_query_results(results, output_file)
    
    # Print statistics
    processor.print_stats()
    
    print(f"\nQuery processing completed!")
    print(f"Found {len(results)} semantically similar columns")
    print(f"Results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
