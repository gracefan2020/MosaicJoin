#!/usr/bin/env python3
"""
Run offline sketch building in parallel via SLURM, then consolidate.

Submits N build jobs (one per chunk of tables), then one consolidate job that
depends on all builds and removes original table subdirs to save space.
"""

import os
import re
import subprocess
from pathlib import Path
import argparse


def discover_tables(embeddings_dir: Path) -> list[str]:
    """Table names = subdir names under embeddings dir."""
    return sorted(d.name for d in embeddings_dir.iterdir() if d.is_dir())


def split_chunks(items: list, n: int) -> list[list]:
    """Split into n roughly equal chunks."""
    size, rem = len(items) // n, len(items) % n
    return [items[i * size + min(i, rem):(i + 1) * size + min(i + 1, rem)] for i in range(n)]


def submit_slurm(cmd: str) -> str | None:
    """Submit sbatch, return job ID or None."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    m = re.search(r"Submitted batch job (\d+)", r.stdout)
    return m.group(1) if m else None


def main(experiment: str, embedding_model: str = "embeddinggemma", embedding_dim: int = 128,
        sketch_size: int = 64, selection_method: str = "farthest_point") -> None:
    """Submit 10 sketch build jobs + 1 consolidate job (afterok dependency)."""
    exp_dir = f"{experiment}-experiments"
    embeddings_dir = Path(f"{exp_dir}/{experiment}_offline_data_{embedding_model}/embeddings_{embedding_model}_{embedding_dim}")
    output_dir = f"{exp_dir}/{experiment}_offline_data_{embedding_model}"
    sketches_dir = f"{output_dir}/sketches_{embedding_model}_{embedding_dim}_k{sketch_size}_{selection_method}"

    if not embeddings_dir.exists():
        print(f"Error: {embeddings_dir} does not exist")
        return

    tables = discover_tables(embeddings_dir)
    chunks = split_chunks(tables, 10)
    job_ids = []

    for i, chunk in enumerate(chunks, 1):
        tables_arg = " ".join(f'"{t}"' for t in chunk)
        cmd = (
            f'python offline_sketch.py build "{embeddings_dir}" '
            f'--output-dir "{sketches_dir}" --sketch-size {sketch_size} '
            f'--selection-method {selection_method} --no-consolidate --tables {tables_arg}'
        )
        script_path = f"{exp_dir}/sketch_chunk_{embedding_model}_{embedding_dim}_k{sketch_size}_{i}.sh"
        Path(script_path).write_text(f"#!/bin/bash\n{cmd}\n")
        os.chmod(script_path, 0o755)
        jid = submit_slurm(
            f"sbatch --account torch_pr_66_general --gres=gpu:1 --nodes=1 --tasks-per-node=1 "
            f"--cpus-per-task=4 --mem=32GB --time=24:00:00 "
            f"--output={exp_dir}/sketch_chunk_{embedding_model}_{embedding_dim}_k{sketch_size}_{i}.log {script_path}"
        )
        if jid:
            job_ids.append(jid)

    # Consolidate runs after all build jobs succeed
    if job_ids:
        dep = ":".join(job_ids)
        cons_script = f"{exp_dir}/consolidate_sketches_{embedding_model}_{embedding_dim}_k{sketch_size}.sh"
        Path(cons_script).write_text(
            f"#!/bin/bash\npython offline_sketch.py consolidate \"{sketches_dir}\" --remove-originals\n"
        )
        os.chmod(cons_script, 0o755)
        submit_slurm(
            f"sbatch --account torch_pr_66_general --dependency=afterok:{dep} "
            f"--nodes=1 --tasks-per-node=1 --cpus-per-task=4 --mem=64GB --time=2:00:00 "
            f"--output={exp_dir}/consolidate_sketches_{embedding_model}_{embedding_dim}_k{sketch_size}.log {cons_script}"
        )
        print(f"Submitted {len(job_ids)} build jobs + 1 consolidate job (after build)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   choices=["autofj", "wt", "freyja", "autofj-wdc", "wt-wdc", "freyja-wdc"])
    p.add_argument("--embedding_model", default="embeddinggemma", choices=["embeddinggemma", "mpnet", "bge"])
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--sketch_size", type=int, default=64)
    p.add_argument("--selection-method", default="farthest_point",
                   choices=["farthest_point", "random", "first_k", "kmeans", "k_closest"])
    args = p.parse_args()
    main(args.experiment, args.embedding_model, args.embedding_dim, args.sketch_size, args.selection_method)
