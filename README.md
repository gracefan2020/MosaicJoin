# Semantic Join Pipeline

A semantic join system that uses MPNet embeddings and k-closest-to-origin sketches to find semantically similar columns across datalakes.

## Overview

This system implements a two-stage semantic join pipeline:

1. **Offline Stage**: Build embeddings and create semantic sketches for all columns in a datalake
2. **Query Stage**: Given a query column, find the top-k most semantically similar columns

The system uses **k-closest-to-origin** sampling to create compact semantic sketches that capture the most representative vectors for each column.

## Architecture

### Core Components

- **`offline_embedding.py`**: Builds MPNet embeddings for datalake columns
- **`offline_sketch.py`**: Creates semantic sketches using k-closest-to-origin sampling
- **`query_time.py`**: Processes queries and finds similar columns
- **`run_offline_processing.py`**: Orchestrates offline embedding and sketch creation
- **`run_query_processing.py`**: Processes semantic join queries

### Parallel Processing Scripts

- **`run_offline_parallel.py`**: Builds embeddings in parallel chunks
- **`run_sketch_parallel.py`**: Builds sketches in parallel chunks

### Evaluation

- **`evaluate_semantic_join.py`**: Evaluates results against ground truth and DeepJoin baseline

## Quick Start

### 1. Build Embeddings (Parallel)

```bash
python run_offline_parallel.py
```

This will:
- Discover all CSV files in the datalake
- Split them into 4 parallel chunks
- Build MPNet embeddings for each chunk
- Save embeddings to `offline_data_chunk_*/embeddings/`

### 2. Build Sketches (Parallel)

```bash
python run_sketch_parallel.py
```

This will:
- Find all tables with embeddings
- Split them into 4 parallel chunks
- Create semantic sketches using k-closest-to-origin sampling
- Save sketches to `offline_data_chunk_*/sketches_k1024/`

### 3. Process Queries

```bash
python run_query_processing.py datasets/freyja-semantic-join/datalake offline_data/sketches_k1024 datasets/freyja-semantic-join/freyja_query_columns.csv --output-dir query_results
```

### 4. Evaluate Results

```bash
python evaluate_semantic_join.py query_results/all_query_results.csv Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv datasets/freyja-semantic-join/freyja_ground_truth.csv --output-dir evaluation_results
```

## Hyperparameters

### Embedding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"auto"` | Device for MPNet model (auto, cpu, cuda, mps) |
| `skip_empty_columns` | `True` | Skip columns with no data |
| `skip_single_value_columns` | `True` | Skip columns with only one unique value |
| `min_values_per_column` | `2` | Minimum values required per column |

### Sketch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sketch_size` | `1024` | Number of representative vectors per sketch (k) |
| `similarity_threshold` | `0.7` | Similarity threshold for validation |
| `use_centered_distance_for_sampling` | `False` | Use centroid distance instead of origin distance |

### Query Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k_return` | `50` | Number of final results to return |
| `similarity_threshold` | `0.7` | Similarity threshold for semantic matching |

### Parallel Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_chunks` | `4` | Number of parallel chunks to create |

## Design Decisions

### 1. K-Closest-to-Origin Sampling

**Decision**: Use k-closest-to-origin sampling for sketch creation.

**Rationale**:
- **Computational Efficiency**: Origin-based sampling is faster than centroid-based
- **Memory Efficiency**: Sketches are compact representations
- **Semantic Preservation**: Points closest to origin often represent common/typical values
- **Scalability**: Works well with large datalakes

**Alternative**: Centroid-based sampling (`use_centered_distance_for_sampling=True`) can be used for different semantic characteristics.

### 2. MPNet Embeddings

**Decision**: Use MPNet (SentenceTransformer) for text embeddings.

**Rationale**:
- **State-of-the-art**: MPNet provides high-quality semantic embeddings
- **Efficiency**: Good balance between quality and speed
- **Generalization**: Works well across different domains and data types

### 3. Sequential Processing (No Batching)

**Decision**: Process columns one-by-one with tqdm progress bars.

**Rationale**:
- **Simplicity**: Easier to debug and understand
- **Memory Efficiency**: No need to manage batch memory
- **Progress Tracking**: Clear visibility into processing progress
- **Resumability**: Can easily resume from failures

### 4. Parallel Chunk Processing

**Decision**: Split datalake into chunks and process in parallel.

**Rationale**:
- **Scalability**: Handle large datalakes efficiently
- **Resource Utilization**: Use multiple CPU cores/GPUs
- **Fault Tolerance**: Failure in one chunk doesn't affect others
- **Flexibility**: Easy to adjust number of chunks based on resources

### 5. Sketch Size in Directory Names

**Decision**: Include sketch size in directory names (`sketches_k1024/`).

**Rationale**:
- **Experiment Management**: Easy to compare different sketch sizes
- **No Conflicts**: Multiple experiments won't overwrite each other
- **Clarity**: Immediately know what parameters were used

## Configuration

### Modifying Hyperparameters

#### For Parallel Embedding Building

Edit `run_offline_parallel.py`:
```python
# Configuration
datalake_dir = "datasets/freyja-semantic-join/datalake"
output_dir = "offline_data"
num_chunks = 4  # Adjust based on your resources
device = "auto"  # or "cpu", "cuda", "mps"
```

#### For Parallel Sketch Building

Edit `run_sketch_parallel.py`:
```python
# Configuration
datalake_dir = "datasets/freyja-semantic-join/datalake"
embeddings_dir = "offline_data/embeddings"
output_dir = "offline_data"
num_chunks = 4
device = "auto"
sketch_size = 1024  # Try 512, 1024, 2048, 4096
similarity_threshold = 0.7  # Try 0.5, 0.6, 0.7, 0.8
```

## Workflow Examples

### Experiment 1: Different Sketch Sizes

```bash
# Build embeddings once
python run_offline_parallel.py

# Try different sketch sizes
# Edit run_sketch_parallel.py: sketch_size = 512
python run_sketch_parallel.py

# Edit run_sketch_parallel.py: sketch_size = 2048  
python run_sketch_parallel.py

# Compare results
python evaluate_semantic_join.py query_results/all_query_results.csv Deepjoin/output/deepjoin_results_K50_N20_T0.7.csv datasets/freyja-semantic-join/freyja_ground_truth.csv --output-dir evaluation_results_k512
```

### Experiment 2: Different Similarity Thresholds

```bash
# Build embeddings and sketches
python run_offline_parallel.py
python run_sketch_parallel.py

# Process queries with different thresholds
python run_query_processing.py datasets/freyja-semantic-join/datalake offline_data/sketches_k1024 datasets/freyja-semantic-join/freyja_query_columns.csv --output-dir query_results_thresh07 --similarity-threshold 0.7

python run_query_processing.py datasets/freyja-semantic-join/datalake offline_data/sketches_k1024 datasets/freyja-semantic-join/freyja_query_columns.csv --output-dir query_results_thresh05 --similarity-threshold 0.5
```

### Experiment 3: Centroid vs Origin Distance

```bash
# Build embeddings
python run_offline_parallel.py

# Build sketches with origin distance (default)
python run_sketch_parallel.py

# Build sketches with centroid distance
# Edit run_sketch_parallel.py: use_centered_distance_for_sampling = True
python run_sketch_parallel.py
```

## Performance Tips

### 1. Resource Management

- **CPU**: Use `num_chunks = CPU_CORES` for optimal parallelization
- **GPU**: Set `device = "cuda"` for faster embedding generation
- **Memory**: Monitor memory usage with large datalakes

### 2. Sketch Size Selection

- **Small sketches (512)**: Faster processing, lower memory, may miss some matches
- **Large sketches (2048+)**: Slower processing, higher memory, better recall
- **Sweet spot**: 1024 provides good balance for most use cases

### 3. Similarity Threshold Tuning

- **High threshold (0.8+)**: High precision, low recall
- **Low threshold (0.5-)**: High recall, low precision  
- **Balanced**: 0.7 works well for most scenarios

## File Structure

```
semantic-join/
├── offline_embedding.py          # Core embedding building
├── offline_sketch.py             # Core sketch creation
├── query_time.py                 # Core query processing
├── run_offline_processing.py     # Orchestrates offline processing
├── run_query_processing.py       # Orchestrates query processing
├── run_offline_parallel.py       # Parallel embedding building
├── run_sketch_parallel.py        # Parallel sketch building
├── evaluate_semantic_join.py     # Evaluation against baselines
├── datasets/                     # Data directory
│   └── freyja-semantic-join/
│       ├── datalake/             # CSV files
│       ├── freyja_query_columns.csv
│       └── freyja_ground_truth.csv
├── offline_data/                 # Generated embeddings and sketches
│   ├── embeddings/
│   └── sketches_k1024/
├── query_results/               # Query processing results
└── evaluation_results/          # Evaluation results
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_chunks` or use `device = "cpu"`
2. **Slow Processing**: Increase `num_chunks` or use `device = "cuda"`
3. **Low Recall**: Decrease `similarity_threshold` or increase `sketch_size`
4. **Low Precision**: Increase `similarity_threshold` or decrease `sketch_size`

### Log Files

- **Embedding logs**: `embedding_chunk_*.log`
- **Sketch logs**: `sketch_chunk_*.log`
- **Query logs**: Check console output
- **Evaluation logs**: Check console output

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Precision@k**: Fraction of top-k results that are correct
- **Recall@k**: Fraction of correct results found in top-k
- **F1@k**: Harmonic mean of precision and recall
- **Total Precision/Recall/F1**: Overall performance across all queries
- **Disagreement Analysis**: Comparison with DeepJoin baseline

## Citation

If you use this system in your research, please cite the relevant papers on semantic joins and MPNet embeddings.
