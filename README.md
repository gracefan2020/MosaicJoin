# MosaicJoin: Compact Semantic Sketches for Value-Level Join Discovery

MosaicJoin embeds datalake columns, builds k-representative sketches (e.g., farthest-point sampling), and retrieves top-k joinable columns for query columns using chamfer-based similarity.

## Environment Setup

### Clone the repository

```bash
git clone <repository-url>
cd MosaicJoin
```

### Create conda environment and install dependencies

```bash
conda create -n MosaicJoin python=3.10 -y
conda activate MosaicJoin
pip install -r requirements.txt
```

### Data layout

Place your datalake and query specifications as follows:

```
datasets/
  {experiment}/
    datalake/           # CSV tables (*.csv)
    query_columns.csv  # columns: target_ds, target_attr
    groundtruth-joinable.csv  # groundtruth
```

AutoFJ tables can be found at: https://github.com/cucumberpeel/thematchmakers/tree/main/data/autofj

WT tables can be found at: https://github.com/cucumberpeel/thematchmakers/tree/main/data/wt 

Freyja tables can be found at: https://freyja-data-discovery.github.io/#resources 

WDC tables can be found at: https://drive.google.com/drive/folders/19vwb45WCayF2j8oPOFf2QVHVopIrgFva  

---

## Pipeline (Offline → Query → Combine → Evaluate)

### 1. Build embeddings

Submit SLURM jobs to embed all columns in the datalake. Output: `{experiment}-experiments/{experiment}_offline_data_{model}/embeddings_{model}_{dim}/`

```bash
python run_offline_embeddings_parallel.py --experiment EXPERIMENT [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment` | Experiment name | — |
| `--embedding_model` | Embedding model | `embeddinggemma` |
| `--embedding_dim` | Output dimension | `128` |

**Experiment choices:** `autofj`, `wt`, `freyja`, `autofj-wdc`, `wt-wdc`, `freyja-wdc`

**Embedding model choices:** `embeddinggemma`, `mpnet`, `bge`

Note: for `embeddinggemma`, you can choose output dimension of 128, 256, 512, 768

---

### 2. Build sketches

Build k-representative sketches from embeddings, then consolidate. Output: `{experiment}-experiments/.../sketches_{model}_{dim}_k{size}_farthest_point/`

```bash
python run_offline_sketch_parallel.py --experiment EXPERIMENT [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment` | Experiment name (required) | — |
| `--embedding_model` | Model (must match embeddings) | `embeddinggemma` |
| `--embedding_dim` | Dimension | `128` |
| `--sketch_size` | k representatives per column | `64` |
| `--selection-method` | Sketch sampling method | `farthest_point` |

**Selection method choices:** `farthest_point` (also called `k-center` in the paper), `random`, `first_k`, `kmeans`, `k_closest`

---

### 3. Run query processing (SLURM)

Submit SLURM array jobs to process queries. Output: `{experiment}-experiments/.../query_results_..._slurm/job_*/`

```bash
python run_queries_slurm.py --experiment EXPERIMENT [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment` | Experiment name (required) | — |
| `--embedding_model` | Model | `embeddinggemma` |
| `--embedding_dim` | Dimension | `128` |
| `--d-sketch-size` | Datalake sketch size (0 = full embeddings) | `64` |
| `--query-sketch-size` | Query sketch size (0 = no sketching) | `0` |
| `--query-sample-size` | Max values per query column | `1000` |
| `--similarity-method` | Chamfer variant | `symmetric_chamfer` |
| `--top-k-return` | Results per query | `50` |

**Similarity method choices:** `chamfer`, `inverse_chamfer`, `symmetric_chamfer`, `harmonic_chamfer`

---

### 4. Combine results and report timing

Merge `job_*/` outputs into one CSV, extract timing from SLURM logs, and validate.

```bash
python combine_slurm_results.py --experiment EXPERIMENT [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--experiment` | Experiment name (required) | — |
| `--embedding_model` | Model | `embeddinggemma` |
| `--embedding_dim` | Dimension | `128` |
| `--d-sketch-size` | Datalake sketch size | `64` |
| `--query-sample-size` | Query sample size | `1000` |
| `--similarity-method` | Chamfer method | `symmetric_chamfer` |
| `--top-k-return` | Top-k used | `50` |
| `--output-dir` | Override inferred output path | — |

---

### 5. Evaluate retrieval

Evaluate combined results against ground truth.

```bash
# Combined experiments (recommended)
python evaluate_retrieval.py --combined --experiments autofj freyja wt [OPTIONS]

# Single method vs ground truth
python evaluate_retrieval.py --results results.csv --ground-truth gt.csv [OPTIONS]

# LLM annotation (WDC benchmarks)
python evaluate_retrieval.py --llm-annotation --benchmarks autofj-wdc wt-wdc [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--combined` | Run combined experiments | `False` |
| `--experiments` | Experiments to evaluate | — |
| `--metrics` | Metrics to compute | — |
| `--k-values` | Top-k values for metrics | `1 3 5 10 20 30 40 50` |
| `--results` | Results CSV (single mode) | — |
| `--ground-truth` | Ground truth CSV (single mode) | — |
| `--llm-annotation` | WDC LLM-verified evaluation | `False` |
| `--benchmarks` | WDC benchmarks | — |
| `--ablation` | Run ablation studies | `False` |

**Experiment choices:** `autofj`, `freyja`, `wt`, `gdc`  
**Metric choices:** `HITS`, `Precision`, `Recall`, `NDCG`, `MRR`

---

## Quick start example

```bash
conda activate MosaicJoin

# 1. Build embeddings (requires datasets/wt/datalake/ and query_columns.csv)
python run_offline_embeddings_parallel.py --experiment wt --embedding_model embeddinggemma --embedding_dim 128

# 2. Build sketches (after embeddings complete)
python run_offline_sketch_parallel.py --experiment wt --embedding_model embeddinggemma --embedding_dim 128 --sketch_size 64

# 3. Run queries (submits SLURM jobs)
python run_queries_slurm.py --experiment wt --d-sketch-size 64 --query-sample-size 1000

# 4. Combine (after SLURM jobs complete)
python combine_slurm_results.py --experiment wt --d-sketch-size 64 --query-sample-size 1000

# 5. Evaluate
python evaluate_retrieval.py --combined --experiments wt --metrics HITS NDCG MRR --k-values 1 5 10 50
```

---

## Evaluation script

```bash
./run_evaluation.sh all      # autofj + freyja + wt
./run_evaluation.sh autofj  # AutoFJ only
./run_evaluation.sh freyja # Freyja only
./run_evaluation.sh wt     # WT only
./run_evaluation.sh wdc     # LLM annotation (WDC)
```

---