#!/usr/bin/env python3
"""
Query processing: submit SLURM jobs for semantic join queries.

Usage: python run_queries_slurm.py --experiment wt [options]
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from query_time import (
    SemanticJoinQueryProcessor,
    QueryConfig,
    QueryColumn,
    save_query_results,
)

QUERIES_PER_JOB = 10


def paths_for_experiment(exp: str, embedding_model: str, embedding_dim: int,
                         d_sketch_size: int, query_sample_size: int,
                         similarity_method: str, top_k_return: int) -> dict:
    """Infer datalake, sketches, query file, output from experiment (same pattern as run_offline_sketch)."""
    exp_dir = f"{exp}-experiments"
    return {
        "datalake_dir": Path(f"datasets/{exp}/datalake"),
        "query_file": Path(f"datasets/{exp}/query_columns.csv"),
        "sketches_dir": Path(f"{exp_dir}/{exp}_offline_data_{embedding_model}/sketches_{embedding_model}_{embedding_dim}_k{d_sketch_size}_farthest_point"),
        "embeddings_dir": Path(f"{exp_dir}/{exp}_offline_data_{embedding_model}/embeddings_{embedding_model}_{embedding_dim}"),
        "output_dir": Path(f"{exp_dir}/{exp}_query_results_{embedding_model}{embedding_dim}_D{d_sketch_size}_Q{query_sample_size}_{similarity_method}_top{top_k_return}_slurm"),
    }


def load_queries(path: Path) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    return [{"target_ds": row["target_ds"], "target_attr": row["target_attr"]} for _, row in df.iterrows()]


def load_column_values(datalake_dir: Path, table_name: str, column_name: str) -> List[str]:
    path = datalake_dir / f"{table_name}.csv"
    if not path.exists() and table_name.endswith(".csv"):
        path = datalake_dir / table_name
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if column_name not in df.columns:
        return []
    return df[column_name].dropna().astype(str).tolist()


def run_queries(datalake_dir: Path, sketches_dir: Path, query_file: Path,
                output_dir: Path, embeddings_dir: Optional[Path],
                d_sketch_size: int, query_indices: Optional[List[int]] = None,
                **cfg) -> int:
    """Process queries and save results."""
    queries = load_queries(query_file)
    if query_indices is not None:
        queries = [queries[i] for i in query_indices if i < len(queries)]

    print(f"Loading query specifications from {query_file}")
    print(f"Found {len(queries)} query specifications")
    print(f"\nProcessing {len(queries)} queries with similarity_method={cfg.get('similarity_method', 'symmetric_chamfer')}...")

    config = QueryConfig(
        top_k_return=cfg.get("top_k_return", 50),
        similarity_threshold=cfg.get("similarity_threshold", 0.1),
        query_sketch_size=cfg.get("query_sketch_size", 0),
        d_sketch_size=d_sketch_size,
        device=cfg.get("device", "auto"),
        similarity_method=cfg.get("similarity_method", "symmetric_chamfer"),
        embedding_model=cfg.get("embedding_model", "embeddinggemma"),
        embedding_dim=cfg.get("embedding_dim", 128),
        large_table_sample_size=cfg.get("query_sample_size") or 0,
    )
    processor = SemanticJoinQueryProcessor(
        config, sketches_dir, embeddings_dir, datalake_dir=datalake_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
    base_idx = query_indices[0] if query_indices else 0
    successful = 0

    for i, q in enumerate(queries):
        tn, cn = q["target_ds"], q["target_attr"]
        values = load_column_values(datalake_dir, tn, cn)
        if not values:
            print(f"Processing query {i+1}/{len(queries)}: {tn}.{cn}")
            print(f"  Skipping: No values found")
            continue
        print(f"Processing query {i+1}/{len(queries)}: {tn}.{cn}")
        print(f"  Loaded {len(values)} values")
        results = processor.process_query(QueryColumn(table_name=tn, column_name=cn, values=values))
        if not results:
            print(f"  No similar columns found")
            continue
        successful += 1
        print(f"  Found {len(results)} similar columns")
        save_query_results(results, output_dir / f"query_{base_idx + i + 1:03d}_{tn}_{cn}.csv")
        q_sample = ", ".join(str(v) for v in values[:5])
        for r in results:
            if r.candidate_table == tn and r.candidate_column == cn:
                continue
            c_vals = load_column_values(datalake_dir, r.candidate_table, r.candidate_column)
            c_sample = ", ".join(str(v) for v in c_vals[:5]) if c_vals else ""
            all_rows.append({
                "query_table": tn, "query_column": cn, "query_index": base_idx + i + 1,
                "candidate_table": r.candidate_table, "candidate_column": r.candidate_column,
                "similarity_score": r.similarity_score,
                "query_sample_values": q_sample,
                "candidate_sample_values": c_sample,
            })

    if all_rows:
        combined_path = output_dir / "all_query_results.csv"
        pd.DataFrame(all_rows).to_csv(combined_path, index=False)
        print(f"\nCombined results saved to: {combined_path}")

    processor.print_stats()
    print(f"\nQuery processing completed!")
    print(f"Successful queries: {successful}/{len(queries)}")
    print(f"Total results: {len(all_rows)}")
    print(f"Results saved to: {output_dir}")
    return successful


def main() -> int:
    p = argparse.ArgumentParser(description="Query processing (submits SLURM)")
    p.add_argument("--experiment", required=True,
                   choices=["autofj", "wt", "freyja", "autofj-wdc", "wt-wdc", "freyja-wdc"])
    p.add_argument("--top-k-return", type=int, default=50)
    p.add_argument("--similarity-method", default="symmetric_chamfer",
                   choices=["chamfer", "inverse_chamfer", "symmetric_chamfer", "harmonic_chamfer"])
    p.add_argument("--d-sketch-size", type=int, default=64)
    p.add_argument("--query-sketch-size", type=int, default=0)
    p.add_argument("--embedding-model", default="embeddinggemma", choices=["embeddinggemma", "mpnet", "bge"])
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--query-sample-size", type=int, default=1000)
    args = p.parse_args()

    paths = paths_for_experiment(
        args.experiment, args.embedding_model, args.embedding_dim,
        args.d_sketch_size, args.query_sample_size,
        args.similarity_method, args.top_k_return
    )
    datalake_dir = paths["datalake_dir"]
    query_file = paths["query_file"]
    sketches_dir = paths["sketches_dir"]
    embeddings_dir = paths["embeddings_dir"]
    output_dir = paths["output_dir"]

    # Worker mode: SLURM array task (SLURM_ARRAY_TASK_ID set) — compute chunk from env
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None:
        task_id = int(task_id)
        num_queries = len(load_queries(query_file))
        start = task_id * QUERIES_PER_JOB
        if start >= num_queries:
            return 0
        end = min(start + QUERIES_PER_JOB - 1, num_queries - 1)
        query_indices = list(range(start, end + 1))
        worker_out = output_dir / f"job_{task_id}"
        emb_dir = embeddings_dir if args.d_sketch_size == 0 else None
        run_queries(datalake_dir, sketches_dir, query_file, worker_out, emb_dir, args.d_sketch_size,
                    query_indices=query_indices,
                    top_k_return=args.top_k_return,
                    similarity_threshold=0.1,
                    similarity_method=args.similarity_method,
                    query_sketch_size=args.query_sketch_size,
                    device="cuda" if __import__("torch").cuda.is_available() else "auto",
                    embedding_model=args.embedding_model,
                    embedding_dim=args.embedding_dim,
                    query_sample_size=args.query_sample_size)
        return 0

    # Driver mode: submit SLURM jobs
    if not query_file.exists():
        print(f"Error: {query_file} not found")
        return 1
    num_queries = len(pd.read_csv(query_file))
    num_jobs = math.ceil(num_queries / QUERIES_PER_JOB)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = output_dir / "run_slurm_jobs.sh"

    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=query
#SBATCH --account=torch_pr_66_general
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output={output_dir}/slurm_%a.out
#SBATCH --error={output_dir}/slurm_%a.err
#SBATCH --array=0-{num_jobs-1}

python run_queries_slurm.py --experiment {args.experiment} \\
  --embedding-model {args.embedding_model} --embedding-dim {args.embedding_dim} \\
  --d-sketch-size {args.d_sketch_size} --query-sketch-size {args.query_sketch_size} \\
  --query-sample-size {args.query_sample_size} --similarity-method {args.similarity_method} \\
  --top-k-return {args.top_k_return}
''')
    script.chmod(0o755)
    subprocess.run(f"sbatch {script}", shell=True)
    print(f"Submitted {num_jobs} jobs. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
