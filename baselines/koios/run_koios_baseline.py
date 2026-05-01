#!/usr/bin/env python3
"""Minimal KOIOS semantic-overlap baseline for MosaicJoin datasets.

The runner consumes the set files and ``ft.sqlite3`` produced by
``build_koios_embedding_db.py``.  It scores a query set against candidate sets
with KOIOS's semantic overlap: maximum bipartite matching over value embeddings,
divided by the query cardinality.  Edges below ``alpha`` contribute zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing
import os
import sqlite3
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from build_koios_embedding_db import normalize_text, resolve_datalake_dir

LOGGER = logging.getLogger("koios-baseline")
EPS = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_SETS_DIR = "work/autofj/sets"
DEFAULT_DB = "work/autofj/ft.sqlite3"
SUPPORTED_DATASETS = ("autofj", "autofj-wdc", "freyja", "freyja-wdc", "wt", "wt-wdc")
EdgeScores = Dict[str, Dict[str, float]]
_SCORE_COLUMNS: Sequence["ColumnSet"] = ()


@dataclass
class ColumnSet:
    table_name: str
    column_name: str
    values: List[str]


@dataclass
class QuerySpec:
    query_table: str
    query_column: str


@dataclass
class CandidateSearchResult:
    ids: List[int]
    edges: EdgeScores


class EmbeddingStore:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.cache: Dict[str, Optional[Tuple[float, ...]]] = {}

    def close(self) -> None:
        self.conn.close()

    def vector(self, value: str) -> Optional[Tuple[float, ...]]:
        if value in self.cache:
            return self.cache[value]

        row = self.conn.execute("SELECT vec FROM wv WHERE word = ?", (value,)).fetchone()
        if row is None:
            self.cache[value] = None
            return None

        vec = array("f")
        vec.frombytes(row[0])
        norm = math.sqrt(sum(x * x for x in vec))
        out = tuple(x / norm for x in vec) if norm > 0.0 else None
        self.cache[value] = out
        return out


def build_value_postings(columns: Sequence[ColumnSet]) -> Dict[str, List[int]]:
    postings: Dict[str, List[int]] = {}
    for idx, column in enumerate(columns):
        for value in column.values:
            postings.setdefault(value, []).append(idx)
    return postings


class FaissCandidateIndex:
    def __init__(
        self,
        db_path: Path,
        postings: Dict[str, List[int]],
        embeddings: EmbeddingStore,
        alpha: float,
        device: str,
        gpu_id: int,
        k: int,
    ):
        import faiss
        import numpy as np

        if k <= 0:
            raise ValueError("--faiss_k must be positive")
        self.faiss = faiss
        self.np = np
        self.postings = postings
        self.embeddings = embeddings
        self.alpha = alpha
        self.k = k
        self.words, vectors = self._load_vectors(db_path)
        self.index = self._build_index(vectors, device, gpu_id)

    def _load_vectors(self, db_path: Path):
        row_count = self.embeddings.conn.execute("SELECT COUNT(*) FROM wv").fetchone()[0]
        vec_len = self.embeddings.conn.execute("SELECT length(vec) FROM wv LIMIT 1").fetchone()[0]
        dim = vec_len // 4

        words: List[str] = []
        vectors = self.np.empty((row_count, dim), dtype=self.np.float32)
        for idx, (word, blob) in enumerate(self.embeddings.conn.execute("SELECT word, vec FROM wv")):
            words.append(word)
            vectors[idx] = self.np.frombuffer(blob, dtype=self.np.float32)

        self.faiss.normalize_L2(vectors)
        return words, vectors

    def _build_index(self, vectors, device: str, gpu_id: int):
        index = self.faiss.IndexFlatIP(vectors.shape[1])
        if device == "gpu":
            if not hasattr(self.faiss, "StandardGpuResources"):
                raise RuntimeError("Installed FAISS package does not include GPU support.")
            if self.faiss.get_num_gpus() <= gpu_id:
                raise RuntimeError(f"FAISS sees fewer than {gpu_id + 1} GPU(s).")
            resources = self.faiss.StandardGpuResources()
            index = self.faiss.index_cpu_to_gpu(resources, gpu_id, index)

        index.add(vectors)
        LOGGER.info("Loaded %s embedding vectors into FAISS %s index", len(self.words), device)
        return index

    def candidates(self, query: ColumnSet) -> CandidateSearchResult:
        candidate_ids: Set[int] = set()
        for value in query.values:
            candidate_ids.update(self.postings.get(value, ()))

        query_items = []
        for value in query.values:
            if is_number(value):
                continue
            vector = self.embeddings.vector(value)
            if vector is not None:
                query_items.append((value, vector))
        if not query_items:
            return CandidateSearchResult(sorted(candidate_ids), {})

        query_values = [value for value, _ in query_items]
        query_vectors = [vector for _, vector in query_items]
        q = self.np.ascontiguousarray(query_vectors, dtype=self.np.float32)
        self.faiss.normalize_L2(q)
        threshold = self.alpha - EPS
        scores, ids = self.index.search(q, min(self.k, len(self.words)))

        valid_edges: EdgeScores = {}
        for value, row_scores, row_ids in zip(query_values, scores, ids):
            for score, idx in zip(row_scores, row_ids):
                if idx < 0 or score < threshold:
                    break
                word = self.words[int(idx)]
                valid_edges.setdefault(value, {})[word] = float(score)
                candidate_ids.update(self.postings.get(word, ()))

        return CandidateSearchResult(sorted(candidate_ids), valid_edges)

    def candidate_ids(self, query: ColumnSet) -> List[int]:
        return self.candidates(query).ids


def setup_logging(level_name: str) -> None:
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def read_values(path: Path, max_values: int) -> List[str]:
    values: List[str] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            value = normalize_text(line)
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
            if max_values > 0 and len(values) >= max_values:
                break
    return values


def infer_column(set_file: str) -> Tuple[str, str]:
    if "__" not in set_file:
        return set_file, "value"
    table_name, column_name = set_file.rsplit("__", 1)
    return table_name, column_name


def load_manifest(manifest_path: Optional[Path]) -> Dict[str, Tuple[str, str]]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {
        row["set_file"]: (row["table_name"], row["column_name"])
        for row in rows
        if "set_file" in row and "table_name" in row and "column_name" in row
    }


def load_sets(
    sets_dir: Path,
    manifest_path: Optional[Path],
    max_sets: int,
    max_values_per_set: int,
) -> List[ColumnSet]:
    manifest = load_manifest(manifest_path)
    files = sorted(p for p in sets_dir.iterdir() if p.is_file())
    if max_sets > 0:
        files = files[:max_sets]

    columns: List[ColumnSet] = []
    for path in files:
        table_name, column_name = manifest.get(path.name, infer_column(path.name))
        values = read_values(path, max_values_per_set)
        if values:
            columns.append(ColumnSet(table_name, column_name, values))

    if not columns:
        raise ValueError(f"No KOIOS set files found in {sets_dir}")
    return columns


def read_csv_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f), []) or []


def canonical_column_name(header: Sequence[str], desired: str) -> Optional[str]:
    desired_norm = (desired or "").strip().lower()
    for name in header:
        if name.strip().lower() == desired_norm:
            return name
    return None


def _resolve_table_path(table_ref: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    ref = Path((table_ref or "").strip())
    candidates = [ref] if ref.suffix.lower() == ".csv" else [ref, ref.with_suffix(".csv")]
    for base in search_dirs:
        for candidate in candidates:
            path = base / candidate
            if path.is_file():
                return path
    return ref if ref.is_file() else None


def _parse_query_pairs(query_file: Path, default_query_column: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with query_file.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return pairs

        header_lower = [h.strip().lower() for h in header]
        key_pairs = [
            ("target_ds", "target_attr"),
            ("query_table", "query_column"),
            ("source_table", "source_column"),
            ("table_name", "column_name"),
            ("table", "column"),
        ]
        for table_key, col_key in key_pairs:
            if table_key in header_lower and col_key in header_lower:
                table_idx = header_lower.index(table_key)
                col_idx = header_lower.index(col_key)
                for row in reader:
                    if len(row) <= max(table_idx, col_idx):
                        continue
                    table_name = row[table_idx].strip()
                    column_name = row[col_idx].strip() or default_query_column
                    if table_name:
                        pairs.append((table_name, column_name))
                return pairs

        if "left_table" in header_lower:
            table_idx = header_lower.index("right_table" if "right_table" in header_lower else "left_table")
            seen: Set[str] = set()
            for row in reader:
                if len(row) <= table_idx:
                    continue
                table_name = row[table_idx].strip()
                if table_name and table_name not in seen:
                    pairs.append((table_name, default_query_column))
                    seen.add(table_name)
            return pairs

        for row in reader:
            if not row:
                continue
            table_name = row[0].strip()
            column_name = row[1].strip() if len(row) > 1 else default_query_column
            if table_name:
                pairs.append((table_name, column_name or default_query_column))
    return pairs


def discover_query_specs(
    dataset_dir: Optional[Path],
    datalake_dir: Optional[Path],
    query_source: Optional[Path],
    default_query_column: str,
    available: Dict[Tuple[str, str], ColumnSet],
) -> List[QuerySpec]:
    if dataset_dir is None and query_source is None:
        return [QuerySpec(col.table_name, col.column_name) for col in available.values()]

    search_dirs: List[Path] = []
    if dataset_dir is not None:
        queries_dir = dataset_dir / "queries"
        if queries_dir.is_dir():
            search_dirs.append(queries_dir)
        if datalake_dir is not None:
            search_dirs.append(datalake_dir)

    source = query_source
    if source is None and dataset_dir is not None:
        for name in ("query_columns.csv", "autofj_query_columns.csv", "gdc_breakdown_query_columns.csv"):
            candidate = dataset_dir / name
            if candidate.is_file():
                source = candidate
                break

    if source is None:
        return [QuerySpec(col.table_name, col.column_name) for col in available.values()]

    if source.is_file():
        specs: List[QuerySpec] = []
        seen: Set[Tuple[str, str]] = set()
        for table_ref, column_name in _parse_query_pairs(source, default_query_column):
            table_path = _resolve_table_path(table_ref, search_dirs) if search_dirs else None
            table_name = table_path.name if table_path is not None else table_ref
            key = (table_name, column_name)
            if key not in seen:
                specs.append(QuerySpec(table_name, column_name))
                seen.add(key)
        return specs

    if source.is_dir():
        specs = []
        for csv_path in sorted(source.glob("*.csv")):
            column_name = canonical_column_name(read_csv_header(csv_path), default_query_column)
            if column_name:
                specs.append(QuerySpec(csv_path.name, column_name))
        return specs

    raise ValueError(f"Query source does not exist: {source}")


def is_number(value: str) -> bool:
    compact = value.replace(".", "")
    return bool(compact) and compact.isdigit()


def cosine(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def value_similarity(left: str, right: str, embeddings: EmbeddingStore, alpha: float) -> float:
    if left == right:
        return 1.0
    if is_number(left) or is_number(right):
        return 0.0
    left_vec = embeddings.vector(left)
    right_vec = embeddings.vector(right)
    if left_vec is None or right_vec is None:
        return 0.0
    score = cosine(left_vec, right_vec)
    return score if score + EPS >= alpha else 0.0


def hungarian_max_weight(weights: Sequence[Sequence[float]]) -> float:
    n_rows = len(weights)
    n_cols = max((len(row) for row in weights), default=0)
    size = max(n_rows, n_cols)
    if size == 0:
        return 0.0

    cost = [[0.0] * (size + 1) for _ in range(size + 1)]
    for i, row in enumerate(weights, start=1):
        for j in range(1, n_cols + 1):
            cost[i][j] = 1.0 - (row[j - 1] if j <= len(row) else 0.0)

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = cost[i0][j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * (size + 1)
    for j in range(1, size + 1):
        assignment[p[j]] = j

    total = 0.0
    for i in range(1, n_rows + 1):
        j = assignment[i]
        if 1 <= j <= n_cols:
            total += weights[i - 1][j - 1]
    return total


def semantic_overlap(
    query: ColumnSet,
    candidate: ColumnSet,
    embeddings: Optional[EmbeddingStore],
    alpha: float,
    valid_edges: Optional[EdgeScores] = None,
) -> float:
    if not query.values:
        return 0.0

    common = set(query.values).intersection(candidate.values)
    base_score = float(len(common))
    query_values = [value for value in query.values if value not in common]
    candidate_values = [value for value in candidate.values if value not in common]
    if not query_values or not candidate_values:
        return base_score / len(query.values)

    if valid_edges is None:
        if embeddings is None:
            raise ValueError("embeddings are required without FAISS valid edges")
        weights = [
            [value_similarity(qv, cv, embeddings, alpha) for cv in candidate_values]
            for qv in query_values
        ]
    else:
        candidate_value_set = set(candidate_values)
        query_values = [
            qv
            for qv in query_values
            if candidate_value_set.intersection(valid_edges.get(qv, ()))
        ]
        candidate_values = [
            cv
            for cv in candidate_values
            if any(cv in valid_edges.get(qv, ()) for qv in query_values)
        ]
        if not query_values or not candidate_values:
            return base_score / len(query.values)
        weights = [
            [valid_edges.get(qv, {}).get(cv, 0.0) for cv in candidate_values]
            for qv in query_values
        ]
    return (base_score + hungarian_max_weight(weights)) / len(query.values)


def score_query(
    query: ColumnSet,
    candidates: Sequence[ColumnSet],
    embeddings: Optional[EmbeddingStore],
    alpha: float,
    top_k: int,
    include_self_matches: bool,
    valid_edges: Optional[EdgeScores] = None,
) -> List[Tuple[ColumnSet, float]]:
    scored: List[Tuple[ColumnSet, float]] = []
    cutoff = 0.0

    for candidate in candidates:
        if not include_self_matches and candidate.table_name == query.table_name:
            continue
        if top_k > 0 and len(scored) >= top_k:
            upper = min(len(query.values), len(candidate.values)) / len(query.values)
            if upper + EPS < cutoff:
                continue

        score = semantic_overlap(query, candidate, embeddings, alpha, valid_edges)
        scored.append((candidate, score))
        scored.sort(key=lambda item: (-item[1], item[0].table_name, item[0].column_name))
        if top_k > 0 and len(scored) > top_k:
            scored.pop()
        if top_k > 0 and len(scored) == top_k:
            cutoff = scored[-1][1]

    return scored


def _score_candidate_chunk(args) -> List[Tuple[int, float]]:
    query, candidate_ids, alpha, include_self_matches, valid_edges = args
    scored: List[Tuple[int, float]] = []
    for candidate_id in candidate_ids:
        candidate = _SCORE_COLUMNS[candidate_id]
        if not include_self_matches and candidate.table_name == query.table_name:
            continue
        score = semantic_overlap(query, candidate, None, alpha, valid_edges)
        scored.append((candidate_id, score))
    return scored


def score_query_parallel(
    query: ColumnSet,
    candidate_ids: Sequence[int],
    columns: Sequence[ColumnSet],
    pool,
    workers: int,
    alpha: float,
    top_k: int,
    include_self_matches: bool,
    valid_edges: EdgeScores,
) -> List[Tuple[ColumnSet, float]]:
    if pool is None or workers <= 1 or len(candidate_ids) <= 1:
        candidates = [columns[idx] for idx in candidate_ids]
        return score_query(query, candidates, None, alpha, top_k, include_self_matches, valid_edges)

    chunk_count = min(workers, len(candidate_ids))
    chunk_size = math.ceil(len(candidate_ids) / chunk_count)
    tasks = [
        (query, list(candidate_ids[start : start + chunk_size]), alpha, include_self_matches, valid_edges)
        for start in range(0, len(candidate_ids), chunk_size)
    ]

    scored: List[Tuple[ColumnSet, float]] = []
    for chunk in pool.map(_score_candidate_chunk, tasks):
        scored.extend((columns[candidate_id], score) for candidate_id, score in chunk)

    scored.sort(key=lambda item: (-item[1], item[0].table_name, item[0].column_name))
    return scored[:top_k] if top_k > 0 else scored


def fill_zero_results(
    results: List[Tuple[ColumnSet, float]],
    query: ColumnSet,
    columns: Sequence[ColumnSet],
    top_k: int,
    include_self_matches: bool,
) -> List[Tuple[ColumnSet, float]]:
    if top_k <= 0 or len(results) >= top_k:
        return results

    seen = {(candidate.table_name, candidate.column_name) for candidate, _ in results}
    for candidate in columns:
        key = (candidate.table_name, candidate.column_name)
        if key in seen or (not include_self_matches and candidate.table_name == query.table_name):
            continue
        results.append((candidate, 0.0))
        seen.add(key)
        if len(results) >= top_k:
            break

    results.sort(key=lambda item: (-item[1], item[0].table_name, item[0].column_name))
    return results[:top_k]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal KOIOS semantic-overlap baseline.")
    parser.add_argument(
        "--dataset_name",
        choices=SUPPORTED_DATASETS,
        help="Shortcut for datasets/<name> and baselines/koios/work/<name> inputs.",
    )
    parser.add_argument("--sets_dir", default=DEFAULT_SETS_DIR, help="KOIOS set directory.")
    parser.add_argument("--db", default=DEFAULT_DB, help="KOIOS SQLite embedding DB.")
    parser.add_argument("--manifest_json", default="", help="Set manifest from build_koios_embedding_db.py.")
    parser.add_argument("--dataset_dir", default="", help="Dataset root containing query metadata.")
    parser.add_argument("--datalake_dir", default="", help="Optional datalake directory.")
    parser.add_argument("--query_source", default="", help="Optional query CSV file or query directory.")
    parser.add_argument("--column_name", default="title", help="Default query column.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Minimum semantic edge similarity.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k candidates per query. Use 0 for all.")
    parser.add_argument("--include_self_matches", action="store_true", help="Include same-table candidates.")
    parser.add_argument("--max_datalake_sets", type=int, default=0, help="Optional cap on set files.")
    parser.add_argument("--max_values_per_set", type=int, default=0, help="Optional cap on values per set.")
    parser.add_argument("--max_queries", type=int, default=0, help="Optional cap on query columns.")
    parser.add_argument("--candidate_mode", default="all", choices=["all", "faiss"], help="Candidate source.")
    parser.add_argument("--faiss_device", default="gpu", choices=["cpu", "gpu"], help="FAISS index device.")
    parser.add_argument("--faiss_gpu_id", type=int, default=0, help="GPU id for FAISS GPU indexes.")
    parser.add_argument("--faiss_k", type=int, default=2048, help="Nearest tokens per query value in FAISS search.")
    parser.add_argument(
        "--score_workers",
        type=int,
        default=int(os.environ.get("KOIOS_SCORE_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="CPU workers for FAISS candidate verification.",
    )
    parser.add_argument("--out_csv", default="", help="Output CSV path.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run(args: argparse.Namespace) -> int:
    if args.dataset_name:
        if not args.dataset_dir:
            args.dataset_dir = str(REPO_ROOT / "datasets" / args.dataset_name)
        if args.sets_dir == DEFAULT_SETS_DIR:
            args.sets_dir = str(BASELINE_ROOT / "work" / args.dataset_name / "sets")
        if args.db == DEFAULT_DB:
            args.db = str(BASELINE_ROOT / "work" / args.dataset_name / "ft.sqlite3")
        if not args.out_csv:
            args.out_csv = str(BASELINE_ROOT / f"koios_ft_{args.dataset_name}.csv")

    if not args.out_csv:
        raise ValueError("Provide --out_csv, or use --dataset_name for the default output name.")

    sets_dir = Path(args.sets_dir).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"KOIOS embedding DB not found: {db_path}")
    manifest_path = Path(args.manifest_json).expanduser().resolve() if args.manifest_json else None
    if manifest_path is None:
        candidate = sets_dir.parent / "sets_manifest.json"
        manifest_path = candidate if candidate.is_file() else None

    columns = load_sets(
        sets_dir=sets_dir,
        manifest_path=manifest_path,
        max_sets=args.max_datalake_sets,
        max_values_per_set=args.max_values_per_set,
    )
    available = {(col.table_name, col.column_name): col for col in columns}
    LOGGER.info("Loaded %s KOIOS sets from %s", len(columns), sets_dir)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    datalake_dir = None
    if args.datalake_dir:
        datalake_dir = resolve_datalake_dir(Path(args.datalake_dir).expanduser().resolve())
    elif dataset_dir is not None:
        datalake_dir = resolve_datalake_dir(dataset_dir)

    query_source = Path(args.query_source).expanduser().resolve() if args.query_source else None
    query_specs = discover_query_specs(
        dataset_dir=dataset_dir,
        datalake_dir=datalake_dir,
        query_source=query_source,
        default_query_column=args.column_name,
        available=available,
    )
    if args.max_queries > 0:
        query_specs = query_specs[: args.max_queries]
    if not query_specs:
        raise ValueError("No query columns discovered.")

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score_workers = max(1, args.score_workers)
    score_pool = None
    if args.candidate_mode == "faiss" and score_workers > 1:
        global _SCORE_COLUMNS
        _SCORE_COLUMNS = columns
        score_pool = multiprocessing.get_context("fork").Pool(processes=score_workers)
        LOGGER.info("Using %s CPU workers for FAISS candidate verification", score_workers)

    embeddings = EmbeddingStore(db_path)
    candidate_index = None
    if args.candidate_mode == "faiss":
        candidate_index = FaissCandidateIndex(
            db_path=db_path,
            postings=build_value_postings(columns),
            embeddings=embeddings,
            alpha=args.alpha,
            device=args.faiss_device,
            gpu_id=args.faiss_gpu_id,
            k=args.faiss_k,
        )

    total_written = 0
    online_start = time.perf_counter()
    try:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_table", "query_column", "candidate_table", "candidate_column", "similarity_score"])

            for query_idx, spec in enumerate(query_specs, start=1):
                query = available.get((spec.query_table, spec.query_column))
                if query is None:
                    LOGGER.warning("Skipping query not found in KOIOS sets: %s.%s", spec.query_table, spec.query_column)
                    continue

                start = time.perf_counter()
                candidates = columns
                valid_edges = None
                if candidate_index is not None:
                    candidate_search = candidate_index.candidates(query)
                    valid_edges = candidate_search.edges
                    results = score_query_parallel(
                        query=query,
                        candidate_ids=candidate_search.ids,
                        columns=columns,
                        pool=score_pool,
                        workers=score_workers,
                        alpha=args.alpha,
                        top_k=args.top_k,
                        include_self_matches=args.include_self_matches,
                        valid_edges=valid_edges,
                    )
                    candidates = [columns[idx] for idx in candidate_search.ids]
                else:
                    results = score_query(
                        query=query,
                        candidates=candidates,
                        embeddings=embeddings,
                        alpha=args.alpha,
                        top_k=args.top_k,
                        include_self_matches=args.include_self_matches,
                        valid_edges=valid_edges,
                    )
                if candidate_index is not None:
                    results = fill_zero_results(
                        results=results,
                        query=query,
                        columns=columns,
                        top_k=args.top_k,
                        include_self_matches=args.include_self_matches,
                    )
                for candidate, score in results:
                    writer.writerow(
                        [
                            query.table_name,
                            query.column_name,
                            candidate.table_name,
                            candidate.column_name,
                            f"{score:.6f}",
                        ]
                    )
                    total_written += 1

                LOGGER.info(
                    "[%s/%s] %s.%s candidates=%s written=%s t=%.3fs",
                    query_idx,
                    len(query_specs),
                    query.table_name,
                    query.column_name,
                    len(candidates),
                    len(results),
                    time.perf_counter() - start,
                )
    finally:
        if score_pool is not None:
            score_pool.close()
            score_pool.join()
        embeddings.close()

    online_seconds = time.perf_counter() - online_start
    LOGGER.info("Wrote KOIOS baseline results: %s (%s rows)", out_path, total_written)
    LOGGER.info("[TIMING] offline_datalake_embedding_seconds=0.000 (loaded cached index)")
    LOGGER.info("[TIMING] online_query_seconds=%.3f", online_seconds)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    try:
        return run(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
