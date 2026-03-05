#!/usr/bin/env python3
"""
Combine SLURM query job outputs: merge job_* results, print timing, validate.

Usage: python combine_slurm_results.py --experiment wt [options]
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

QUERIES_PER_JOB = 10
EXP_DIR_SUFFIX = "experiments"


def output_dir_for(exp: str, embedding_model: str, embedding_dim: int,
                   d_sketch_size: int, query_sample_size: int,
                   similarity_method: str, top_k_return: int) -> Path:
    """Infer output dir from experiment and config (matches run_queries_slurm)."""
    exp_dir = f"{exp}-{EXP_DIR_SUFFIX}"
    return Path(f"{exp_dir}/{exp}_query_results_{embedding_model}{embedding_dim}_D{d_sketch_size}_Q{query_sample_size}_{similarity_method}_top{top_k_return}_slurm")


def extract_timing(output_dir: Path) -> None:
    """Parse slurm_*.out for timing stats; print averages across jobs."""
    total_vals, embed_vals, search_vals = [], [], []
    for f in sorted(output_dir.glob("slurm_*.out")):
        if not f.is_file():
            continue
        text = f.read_text(errors="ignore")
        if m := re.search(r"Average time per query \(embedding \+ search\): ([0-9.]+)", text):
            total_vals.append(float(m.group(1)))
        if m := re.search(r"Avg embedding time per query: ([0-9.]+)", text):
            embed_vals.append(float(m.group(1)))
        if m := re.search(r"Avg search time per query: ([0-9.]+)", text):
            search_vals.append(float(m.group(1)))
    if total_vals:
        print(f"\nRuntime (avg across {len(total_vals)} jobs):")
        print(f"  Avg time per query (embedding + search): {sum(total_vals)/len(total_vals):.4f}s")
        if embed_vals:
            print(f"  Avg embedding time per query: {sum(embed_vals)/len(embed_vals):.4f}s")
        if search_vals:
            print(f"  Avg search time per query: {sum(search_vals)/len(search_vals):.4f}s")


def load_results(output_dir: Path, num_jobs: int) -> pd.DataFrame:
    """Load from job_* subdirs: all_query_results.csv or individual query_*.csv."""
    all_dfs = []
    for job_id in range(num_jobs):
        job_dir = output_dir / f"job_{job_id}"
        if not job_dir.exists():
            continue
        combined = job_dir / "all_query_results.csv"
        if combined.exists():
            all_dfs.append(pd.read_csv(combined))
        else:
            for f in sorted(job_dir.glob("query_*.csv")):
                all_dfs.append(pd.read_csv(f))
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def main() -> int:
    p = argparse.ArgumentParser(description="Combine SLURM query results and report timing")
    p.add_argument("--experiment", required=True,
                   choices=["autofj", "wt", "freyja", "gdc", "autofj-wdc", "wt-wdc", "freyja-wdc"])
    p.add_argument("--embedding-model", default="embeddinggemma", choices=["embeddinggemma", "mpnet", "bge"])
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--d-sketch-size", type=int, default=64)
    p.add_argument("--query-sample-size", type=int, default=1000)
    p.add_argument("--similarity-method", default="symmetric_chamfer")
    p.add_argument("--top-k-return", type=int, default=50)
    p.add_argument("--output-dir", help="Override inferred output dir")
    args = p.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else output_dir_for(
        args.experiment, args.embedding_model, args.embedding_dim,
        args.d_sketch_size, args.query_sample_size,
        args.similarity_method, args.top_k_return
    )
    print(f"Output: {output_dir}")

    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist")
        return 1

    # Infer num_jobs from query file or job_* dirs
    query_file = Path(f"datasets/{args.experiment}/query_columns.csv")
    if query_file.exists():
        num_queries = len(pd.read_csv(query_file))
        num_jobs = math.ceil(num_queries / QUERIES_PER_JOB)
    else:
        job_ids = [int(d.name.split("_")[1]) for d in output_dir.glob("job_*")
                   if d.is_dir() and d.name.split("_")[-1].isdigit()]
        num_jobs = max(job_ids) + 1 if job_ids else 1

    combined = load_results(output_dir, num_jobs)
    if combined.empty:
        job_dirs = list(output_dir.glob("job_*"))
        print("No results to combine." + (" job_* dirs exist but are empty." if job_dirs else " No job_* dirs."))
        if job_dirs:
            print("  Check slurm_*.err for failures.")
        return 1

    out_file = output_dir / "all_query_results.csv"
    combined.to_csv(out_file, index=False)
    print(f"Combined {len(combined)} results → {out_file}")

    extract_timing(output_dir)

    # Validate: queries with fewer than top_k results
    if "query_table" in combined.columns and "query_column" in combined.columns:
        counts = combined.groupby(["query_table", "query_column"]).size()
        incomplete = (counts < args.top_k_return).sum()
        if incomplete > 0:
            print(f"\nWarning: {incomplete} queries have fewer than {args.top_k_return} results")
        else:
            print(f"\nAll {len(counts)} queries have {args.top_k_return} results")

    print(f"Done: {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
