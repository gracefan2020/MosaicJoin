# Semantic Join Pipeline

A semantic join system that uses MPNet embeddings and k-closest-to-origin sketches to find semantically similar columns across datalakes.

## Overview

This system implements a three-stage semantic join pipeline:

1. **Offline Embedding Stage**: Build MPNet embeddings for all columns in a datalake
2. **Offline Sketch Stage**: Create semantic sketches using k-closest-to-origin sampling
3. **Query Stage**: Given a query column, find the top-k most semantically similar columns

The system uses **k-closest-to-origin** sampling to create compact semantic sketches that capture the most representative vectors for each column.

## Architecture

### Core Components

- **`offline_embedding.py`**: Builds MPNet embeddings for datalake columns (saves as pickle files)
- **`offline_sketch.py`**: Creates semantic sketches using k-closest-to-origin sampling (saves as pickle files)
- **`query_time.py`**: Processes individual queries and finds similar columns
- **`run_query_processing.py`**: Processes multiple queries from CSV file

### Scripts

- **`run_offline_embeddings_parallel.py`**: Builds embeddings in parallel chunks with automatic cleanup
- **`run_offline_sketch_parallel.py`**: Builds sketches in parallel chunks with automatic cleanup
- **`run_queries.py`**: Simple script to run query processing with common configurations

### Evaluation

- **`evaluate_semantic_join.py`**: Evaluates results against ground truth and DeepJoin baseline
- **`run_evaluation.py`**: Simple script to run evaluation with common configurations
- **`run_configuration_comparison.py`**: Runs multiple configurations and compares results

## Quick Start

### 1. Build Embeddings (Parallel)

```bash
python run_offline_embeddings_parallel.py
```

This will:
- **Automatically clean** all previous runs
- Discover all CSV files in the datalake
- Split them into parallel chunks (configurable)
- Build MPNet embeddings for each chunk
- Save embeddings as pickle files to `offline_data/embeddings/`

### 2. Build Sketches (Parallel)

```bash
python run_offline_sketch_parallel.py
```

This will:
- **Automatically clean** previous sketch data
- Find all tables with embeddings
- Split them into parallel chunks
- Create semantic sketches using k-closest-to-origin sampling
- Save sketches as pickle files to `offline_data/sketches_k1024/` (if sketch size is 1024)

### 3. Process Queries

```bash
python run_queries.py
```

This will:
- **Automatically clean** previous query results
- Process all queries from `datasets/freyja-semantic-join/freyja_query_columns.csv`
- Find semantically similar columns using pre-built sketches
- Save results to `query_results_k1024_t0.7_top50/` (configuration-based naming)

### 4. Evaluate Results

```bash
python run_evaluation.py
```

This will:
- **Automatically detect** the latest query results directory
- Compare semantic join results with DeepJoin baseline
- Evaluate against ground truth
- Generate evaluation metrics and plots
- Save results to `evaluation_results_k1024_t0.7_top50/` (matching configuration)

### 5. Compare Different Configurations (Optional)

```bash
python run_configuration_comparison.py
```

This will:
- Run multiple configurations automatically
- Test different sketch sizes, thresholds, and top-k values
- Save results in separate directories for easy comparison
- Generate evaluation summaries for each configuration

## Direct Module Usage

You can also run the core modules directly for more control:

### Individual Embedding Building

```bash
python offline_embedding.py "datasets/freyja-semantic-join/datalake" \
    --output-dir "offline_data/embeddings" \
    --device "auto" \
    --tables "table1.csv" "table2.csv"
```

### Individual Sketch Building

```bash
python offline_sketch.py "offline_data/embeddings" \
    --output-dir "offline_data/sketches_k1024" \
    --sketch-size 1024 \
    --similarity-threshold 0.7
```

### Individual Query Processing

```bash
python query_time.py "datasets/freyja-semantic-join/datalake/query_table.csv" \
    "column_name" \
    "offline_data/sketches_k1024" \
    --output-file "query_results.csv" \
    --top-k-return 50
```

### Multiple Query Processing

```bash
python run_query_processing.py "datasets/freyja-semantic-join/datalake" \
    "offline_data/sketches_k1024" \
    "datasets/freyja-semantic-join/freyja_query_columns.csv" \
    --output-dir "query_results" \
    --top-k-return 50 \
    --similarity-threshold 0.7
```

## Configuration

### Parallel Processing Parameters

#### Embedding Building (`run_offline_embeddings_parallel.py`)

```python
# Configuration
datalake_dir = "datasets/freyja-semantic-join/datalake"
output_dir = "offline_data"
num_chunks = 32  # Adjust based on your resources
device = "auto"  # or "cpu", "cuda", "mps"
```

#### Sketch Building (`run_offline_sketch_parallel.py`)

```python
# Configuration
embeddings_dir = "offline_data/embeddings"
output_dir = "offline_data"
num_chunks = 32  # Adjust based on your resources
sketch_size = 1024  # Try 512, 1024, 2048, 4096
similarity_threshold = 0.7  # Try 0.5, 0.6, 0.7, 0.8
```

### Core Module Parameters

#### Embedding Parameters (`offline_embedding.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `"auto"` | Device for MPNet model (auto, cpu, cuda, mps) |
| `skip_empty_columns` | `True` | Skip columns with no data |
| `save_metadata` | `True` | Save metadata for each embedding |

#### Sketch Parameters (`offline_sketch.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sketch_size` | `1024` | Number of representative vectors per sketch (k) |
| `similarity_threshold` | `0.7` | Similarity threshold for validation |
| `use_centered_distance_for_sampling` | `False` | Use centroid distance instead of origin distance |
| `skip_empty_columns` | `True` | Skip columns with no embeddings |

#### Query Parameters (`query_time.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k_return` | `50` | Number of final results to return |
| `similarity_threshold` | `0.7` | Similarity threshold for semantic matching |
| `sketch_size` | `1024` | Number of representative vectors per sketch |
| `device` | `"auto"` | Device for MPNet model |

## File Formats

### Data Storage

- **Embeddings**: Saved as pickle files (`.pkl`) containing NumPy arrays
- **Sketches**: Saved as pickle files (`.pkl`) containing semantic sketch objects
- **Metadata**: Saved as JSON files for each embedding/sketch
- **Logs**: Text files with processing logs and statistics

### Directory Structure

```
offline_data/
├── embeddings/
│   ├── table1/
│   │   ├── column1_0.pkl
│   │   ├── column1_0_metadata.json
│   │   ├── column2_1.pkl
│   │   └── column2_1_metadata.json
│   └── table2/
│       └── ...
└── sketches_k1024/
    ├── table1/
    │   ├── column1_0.pkl
    │   ├── column1_0_metadata.json
    │   └── ...
    └── table2/
        └── ...

query_results_k1024_t0.7_top50/
├── all_query_results.csv
├── query_summary.json
└── query_001_table1_column1.csv
    └── ...

evaluation_results_k1024_t0.7_top50/
├── evaluation_summary.json
├── precision_recall_plots.png
└── disagreement_analysis.csv
```

## Design Decisions

### K-Closest-to-Origin Sampling

**Decision**: Use k-closest-to-origin sampling for sketch creation.

**Rationale**:
- **Computational Efficiency**: Origin-based sampling is faster than centroid-based
- **Memory Efficiency**: Sketches are compact representations
- **Semantic Preservation**: Points closest to origin often represent common/typical values
- **Scalability**: Works well with large datalakes

### MPNet Embeddings

**Decision**: Use MPNet (SentenceTransformer) for text embeddings.

**Rationale**:
- **State-of-the-art**: MPNet provides high-quality semantic embeddings
- **Efficiency**: Good balance between quality and speed
- **Generalization**: Works well across different domains and data types
- **Contextual**: Uses column names for better semantic understanding