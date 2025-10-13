# Quick Usage Guide

## Basic Workflow

### 1. Build Embeddings
```bash
python run_offline_embeddings_parallel.py
```

### 2. Build Sketches  
```bash
python run_offline_sketch_parallel.py
```

### 3. Process Queries
```bash
python run_queries.py
```

### 4. Evaluate Results
```bash
python run_evaluation.py
```

## Alternative: Direct Commands

### Process Queries (Direct)
```bash
python run_query_processing.py datasets/freyja-semantic-join/datalake offline_data/sketches_k1024 datasets/freyja-semantic-join/freyja_query_columns.csv --output-dir query_results
```

### Evaluate Results (Direct)
```bash
python evaluate_semantic_join.py query_results/all_query_results.csv join/Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv datasets/freyja-semantic-join/freyja_ground_truth.csv --output-dir evaluation_results
```

## Key Hyperparameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `num_chunks` | `run_offline_embeddings_parallel.py` | 4 | Parallel chunks for embeddings |
| `num_chunks` | `run_offline_sketch_parallel.py` | 4 | Parallel chunks for sketches |
| `sketch_size` | `run_offline_sketch_parallel.py` | 1024 | K for k-closest-to-origin |
| `similarity_threshold` | `run_offline_sketch_parallel.py` | 0.7 | Similarity threshold |
| `device` | Both scripts | "auto" | Device (auto/cpu/cuda/mps) |

## Experiment Examples

### Different Sketch Sizes
```bash
# Edit run_offline_sketch_parallel.py: sketch_size = 512
python run_offline_sketch_parallel.py

# Edit run_offline_sketch_parallel.py: sketch_size = 2048  
python run_offline_sketch_parallel.py
```

### Different Similarity Thresholds
```bash
# Edit run_offline_sketch_parallel.py: similarity_threshold = 0.5
python run_offline_sketch_parallel.py

# Edit run_offline_sketch_parallel.py: similarity_threshold = 0.8
python run_offline_sketch_parallel.py
```

## File Structure
```
offline_data/
├── embeddings/           # MPNet embeddings
└── sketches_k1024/      # Semantic sketches

query_results/
└── all_query_results.csv # Query results

evaluation_results/
├── evaluation_metrics.json
└── summary_stats.json
```

## Log Files
- `embedding_chunk_*.log` - Embedding building logs
- `sketch_chunk_*.log` - Sketch building logs
