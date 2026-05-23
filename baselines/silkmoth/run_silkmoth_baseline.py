#!/usr/bin/env python3
"""Minimal SILKMOTH baseline for SemSketch-style benchmarks.

This implementation focuses on the paper's Jaccard search setting with
alpha = 0: token-based element similarity, weighted signature generation,
check filtering, nearest-neighbor refinement, and exact maximum-weight
bipartite matching verification.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import multiprocessing
import os
import re
import resource
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from resource_monitor import (
    ResourceMonitor,
    add_resource_monitor_args,
    default_resource_log_path,
    log_resource_summary,
)

LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
EPS = 1e-12
_SCORE_DATALAKE_COLUMNS: Sequence["ColumnSet"] = ()


@dataclass
class QuerySpec:
    query_table: str
    query_column: str
    csv_path: Path


@dataclass(frozen=True)
class Element:
    text: str
    tokens: frozenset[str]

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass
class ColumnSet:
    table_name: str
    column_name: str
    csv_path: Path
    elements: List[Element]
    token_to_element_ids: Dict[str, List[int]]

    @property
    def size(self) -> int:
        return len(self.elements)


@dataclass
class QuerySignature:
    per_element_tokens: List[Set[str]]
    element_bounds: List[float]
    theta: float


@dataclass
class CandidateState:
    best_sig_sims: Dict[int, float] = field(default_factory=dict)


def setup_logging(level_name: str) -> None:
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def log_mem(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    LOGGER.info("[MEM] %s rss=%.1f MB", msg, rss_mb)


def _has_csv_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(p.is_file() and p.suffix.lower() == ".csv" for p in path.iterdir())
    except OSError:
        return False


def resolve_datalake_dir(path: Path) -> Path:
    nested = path / "datalake"
    if _has_csv_files(nested):
        return nested
    if _has_csv_files(path):
        return path
    return nested if nested.exists() else path


def read_csv_header(csv_path: Path) -> List[str]:
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            return next(reader, []) or []
    except OSError:
        return []


def canonical_column_name(header: Sequence[str], desired: str) -> Optional[str]:
    desired_norm = (desired or "").strip().lower()
    if not desired_norm:
        return None
    for name in header:
        if name.strip().lower() == desired_norm:
            return name
    return None


def discover_selected_columns(csv_path: Path, column_name: str) -> List[str]:
    header = read_csv_header(csv_path)
    if not header:
        return []

    column_norm = (column_name or "").strip().lower()
    if column_norm in {"*", "all"}:
        out: List[str] = []
        for name in header:
            if name.strip().lower() in {"id", "index"}:
                continue
            out.append(name)
        return out

    matched = canonical_column_name(header, column_name)
    return [matched] if matched else []


def read_selected_column_values(csv_path: Path, selected_columns: Sequence[str]) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {col: [] for col in selected_columns}
    if not selected_columns:
        return values

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return values

        index_by_column: Dict[str, int] = {}
        for col in selected_columns:
            for idx, name in enumerate(header):
                if name == col:
                    index_by_column[col] = idx
                    break

        for row in reader:
            for col, idx in index_by_column.items():
                values[col].append(row[idx] if idx < len(row) else "")

    return values


def _resolve_table_path(table_ref: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    ref = (table_ref or "").strip()
    if not ref:
        return None

    ref_path = Path(ref)
    if ref_path.is_file():
        return ref_path

    candidates = [ref_path]
    if ref_path.suffix.lower() != ".csv":
        candidates.append(ref_path.with_suffix(".csv"))

    for base in search_dirs:
        for candidate in candidates:
            joined = base / candidate
            if joined.is_file():
                return joined
    return None


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
            ("table_name", "col_name"),
            ("table", "column"),
        ]

        indices: Optional[Tuple[int, int]] = None
        for table_key, col_key in key_pairs:
            if table_key in header_lower and col_key in header_lower:
                indices = (header_lower.index(table_key), header_lower.index(col_key))
                break

        if indices is None and "left_table" in header_lower:
            table_key = "right_table" if "right_table" in header_lower else "left_table"
            table_idx = header_lower.index(table_key)
            seen: Set[str] = set()
            for row in reader:
                if len(row) <= table_idx:
                    continue
                table_name = row[table_idx].strip()
                if table_name and table_name not in seen:
                    pairs.append((table_name, default_query_column))
                    seen.add(table_name)
            return pairs

        if indices is None:
            for row in reader:
                if len(row) < 1:
                    continue
                table_name = row[0].strip()
                col_name = row[1].strip() if len(row) > 1 else default_query_column
                if table_name:
                    pairs.append((table_name, col_name or default_query_column))
            return pairs

        table_idx, col_idx = indices
        for row in reader:
            if len(row) <= max(table_idx, col_idx):
                continue
            table_name = row[table_idx].strip()
            col_name = row[col_idx].strip() or default_query_column
            if table_name:
                pairs.append((table_name, col_name))

    return pairs


def discover_query_specs(
    dataset_dir: Path,
    datalake_dir: Path,
    query_source: Optional[Path],
    default_query_column: str,
) -> Tuple[List[QuerySpec], str]:
    source = query_source
    search_dirs: List[Path] = []
    queries_dir = dataset_dir / "queries"
    if queries_dir.is_dir():
        search_dirs.append(queries_dir)
    search_dirs.append(datalake_dir)

    if source is None:
        candidate_files = [
            dataset_dir / "query_columns.csv",
            dataset_dir / "autofj_query_columns.csv",
            dataset_dir / "gdc_breakdown_query_columns.csv",
            dataset_dir / "groundtruth-joinable.csv",
        ]
        for candidate in candidate_files:
            if candidate.is_file():
                source = candidate
                break
        if source is None:
            if queries_dir.is_dir() and _has_csv_files(queries_dir):
                source = queries_dir
            else:
                source = datalake_dir

    specs: List[QuerySpec] = []
    seen_keys: Set[Tuple[str, str]] = set()
    if source.is_file():
        pairs = _parse_query_pairs(source, default_query_column=default_query_column)
        for table_ref, query_col in pairs:
            table_path = _resolve_table_path(table_ref, search_dirs=search_dirs)
            if table_path is None:
                LOGGER.warning("Query table not found: %s", table_ref)
                continue
            key = (table_path.name, query_col)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            specs.append(QuerySpec(query_table=table_path.name, query_column=query_col, csv_path=table_path))
        return specs, f"file:{source}"

    if source.is_dir():
        for csv_path in sorted(source.glob("*.csv")):
            for col_name in discover_selected_columns(csv_path, default_query_column):
                key = (csv_path.name, col_name)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                specs.append(QuerySpec(query_table=csv_path.name, query_column=col_name, csv_path=csv_path))
        return specs, f"dir:{source}"

    raise ValueError(f"Query source does not exist: {source}")


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def tokenize_text(text: str) -> frozenset[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if tokens:
        return frozenset(tokens)
    if text:
        return frozenset({text})
    return frozenset()


def unique_normalized_values(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        norm = normalize_text(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def build_column_set(table_name: str, column_name: str, csv_path: Path, raw_values: Sequence[str]) -> Optional[ColumnSet]:
    elements: List[Element] = []
    token_to_element_ids: Dict[str, List[int]] = defaultdict(list)

    for value in unique_normalized_values(raw_values):
        tokens = tokenize_text(value)
        if not tokens:
            continue
        elem_idx = len(elements)
        elements.append(Element(text=value, tokens=tokens))
        for token in tokens:
            token_to_element_ids[token].append(elem_idx)

    if not elements:
        return None

    return ColumnSet(
        table_name=table_name,
        column_name=column_name,
        csv_path=csv_path,
        elements=elements,
        token_to_element_ids=dict(token_to_element_ids),
    )


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left.intersection(right))
    if inter == 0:
        return 0.0
    union = len(left) + len(right) - inter
    return inter / union if union else 0.0


def size_filter_allows(query_size: int, candidate_size: int, metric: str, threshold: float) -> bool:
    if query_size <= 0 or candidate_size <= 0:
        return False

    if metric == "containment":
        return True

    # For SET-SIMILARITY, very different sizes cannot reach δ.
    lower = threshold * query_size
    upper = query_size / threshold if threshold > 0 else float("inf")
    return candidate_size + EPS >= lower and candidate_size <= upper + EPS


def build_datalake_index(
    datalake_dir: Path,
    column_name: str,
    max_datalake_tables: int,
) -> Tuple[List[ColumnSet], Dict[str, array], Dict[str, int]]:
    csv_files = sorted(datalake_dir.glob("*.csv"))
    if max_datalake_tables > 0:
        csv_files = csv_files[:max_datalake_tables]
    if not csv_files:
        raise ValueError(f"No CSV files found in datalake: {datalake_dir}")

    columns: List[ColumnSet] = []
    inverted_index: Dict[str, array] = defaultdict(lambda: array("I"))

    for csv_path in csv_files:
        selected_columns = discover_selected_columns(csv_path, column_name)
        if not selected_columns:
            continue
        raw_values_by_column = read_selected_column_values(csv_path, selected_columns)
        for selected in selected_columns:
            column = build_column_set(csv_path.name, selected, csv_path, raw_values_by_column.get(selected, []))
            if column is None:
                continue
            col_idx = len(columns)
            columns.append(column)
            for token, elem_ids in column.token_to_element_ids.items():
                postings = inverted_index[token]
                for elem_id in elem_ids:
                    postings.append(col_idx)
                    postings.append(elem_id)
            if len(columns) % 1000 == 0:
                log_mem(f"indexed_columns={len(columns)} tokens={len(inverted_index)}")

    if not columns:
        raise ValueError(f"No datalake columns were indexed from {datalake_dir}")

    posting_counts = {token: len(postings) // 2 for token, postings in inverted_index.items()}
    return columns, dict(inverted_index), posting_counts


def build_query_signature(query: ColumnSet, posting_counts: Dict[str, int], threshold: float) -> QuerySignature:
    token_to_value: Dict[str, float] = defaultdict(float)
    token_to_elements: Dict[str, Set[int]] = defaultdict(set)
    per_element_tokens: List[Set[str]] = [set() for _ in query.elements]
    element_bounds: List[float] = [1.0 for _ in query.elements]

    for elem_idx, element in enumerate(query.elements):
        if element.token_count <= 0:
            continue
        weight = 1.0 / element.token_count
        for token in element.tokens:
            token_to_value[token] += weight
            token_to_elements[token].add(elem_idx)

    ranked_tokens: List[Tuple[float, int, str]] = []
    for token, value in token_to_value.items():
        cost = posting_counts.get(token, 0)
        ratio = (cost / value) if value > 0 else float("inf")
        ranked_tokens.append((ratio, cost, token))
    ranked_tokens.sort(key=lambda item: (item[0], item[1], item[2]))

    total_upper = float(len(query.elements))
    for _ratio, _cost, token in ranked_tokens:
        changed = False
        for elem_idx in token_to_elements[token]:
            if token in per_element_tokens[elem_idx]:
                continue
            per_element_tokens[elem_idx].add(token)
            changed = True
            total_upper -= 1.0 / query.elements[elem_idx].token_count
        if changed and total_upper + EPS < threshold:
            break

    for elem_idx, element in enumerate(query.elements):
        if element.token_count <= 0:
            element_bounds[elem_idx] = 0.0
            continue
        sig_size = len(per_element_tokens[elem_idx])
        element_bounds[elem_idx] = (element.token_count - sig_size) / element.token_count

    return QuerySignature(
        per_element_tokens=per_element_tokens,
        element_bounds=element_bounds,
        theta=threshold,
    )


def select_candidates(
    query: ColumnSet,
    signature: QuerySignature,
    datalake_columns: Sequence[ColumnSet],
    inverted_index: Dict[str, array],
    include_self_matches: bool,
    metric: str,
    relatedness_threshold: float,
) -> Dict[int, CandidateState]:
    candidate_states: Dict[int, CandidateState] = {}

    for query_elem_idx, sig_tokens in enumerate(signature.per_element_tokens):
        if not sig_tokens:
            continue

        query_element = query.elements[query_elem_idx]
        seen_by_candidate: Dict[int, Set[int]] = {}
        for token in sig_tokens:
            postings = inverted_index.get(token)
            if not postings:
                continue
            for p in range(0, len(postings), 2):
                candidate_idx = postings[p]
                candidate_elem_idx = postings[p + 1]
                candidate = datalake_columns[candidate_idx]
                if not include_self_matches and candidate.table_name == query.table_name:
                    continue
                if not size_filter_allows(query.size, candidate.size, metric, relatedness_threshold):
                    continue

                candidate_seen = seen_by_candidate.setdefault(candidate_idx, set())
                if candidate_elem_idx in candidate_seen:
                    continue
                candidate_seen.add(candidate_elem_idx)

                sim = jaccard_similarity(query_element.tokens, candidate.elements[candidate_elem_idx].tokens)
                if sim <= 0.0:
                    continue

                state = candidate_states.get(candidate_idx)
                if state is None:
                    state = CandidateState()
                    candidate_states[candidate_idx] = state
                prev = state.best_sig_sims.get(query_elem_idx)
                if prev is None or sim > prev:
                    state.best_sig_sims[query_elem_idx] = sim

    filtered: Dict[int, CandidateState] = {}
    for candidate_idx, state in candidate_states.items():
        for query_elem_idx, sim in state.best_sig_sims.items():
            if sim + EPS >= signature.element_bounds[query_elem_idx]:
                filtered[candidate_idx] = state
                break
    return filtered


def nearest_neighbor_similarity(
    query_element: Element,
    candidate: ColumnSet,
    initial_best: float = 0.0,
) -> float:
    best = initial_best
    seen: Set[int] = set()
    for token in query_element.tokens:
        for candidate_elem_idx in candidate.token_to_element_ids.get(token, []):
            if candidate_elem_idx in seen:
                continue
            seen.add(candidate_elem_idx)
            sim = jaccard_similarity(query_element.tokens, candidate.elements[candidate_elem_idx].tokens)
            if sim > best:
                best = sim
    return best


def nearest_neighbor_filter(
    query: ColumnSet,
    signature: QuerySignature,
    candidates: Dict[int, CandidateState],
    datalake_columns: Sequence[ColumnSet],
) -> Dict[int, CandidateState]:
    refined: Dict[int, CandidateState] = {}
    base_total = sum(signature.element_bounds)

    for candidate_idx, state in candidates.items():
        total = base_total
        exact_query_elems: Set[int] = set()

        for query_elem_idx, best_sig_sim in state.best_sig_sims.items():
            bound = signature.element_bounds[query_elem_idx]
            if best_sig_sim + EPS >= bound:
                total += best_sig_sim - bound
                exact_query_elems.add(query_elem_idx)

        if total + EPS < signature.theta:
            continue

        candidate = datalake_columns[candidate_idx]
        pending = [idx for idx in range(len(query.elements)) if idx not in exact_query_elems]
        pending.sort(key=lambda idx: state.best_sig_sims.get(idx, 0.0) - signature.element_bounds[idx])

        pruned = False
        for query_elem_idx in pending:
            bound = signature.element_bounds[query_elem_idx]
            best_known = state.best_sig_sims.get(query_elem_idx, 0.0)
            nn_sim = nearest_neighbor_similarity(
                query.elements[query_elem_idx],
                candidate,
                initial_best=best_known,
            )
            if nn_sim + EPS < bound:
                total += nn_sim - bound
                if total + EPS < signature.theta:
                    pruned = True
                    break

        if not pruned:
            refined[candidate_idx] = state

    return refined


def hungarian_max_weight(weights: Sequence[Sequence[float]]) -> float:
    n_rows = len(weights)
    n_cols = max((len(row) for row in weights), default=0)
    size = max(n_rows, n_cols)
    if size == 0:
        return 0.0

    cost = [[0.0] * (size + 1) for _ in range(size + 1)]
    for i in range(n_rows):
        row = weights[i]
        for j in range(n_cols):
            value = row[j] if j < len(row) else 0.0
            cost[i + 1][j + 1] = 1.0 - value

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
    for i in range(1, size + 1):
        j = assignment[i]
        if i <= n_rows and j <= n_cols:
            total += weights[i - 1][j - 1]
    return total


def verify_candidate(
    query: ColumnSet,
    candidate: ColumnSet,
    metric: str,
    relatedness_threshold: float,
) -> float:
    query_values = {e.text for e in query.elements}
    candidate_values = {e.text for e in candidate.elements}
    common_values = query_values.intersection(candidate_values)
    identical_score = float(len(common_values))

    remaining_query = [elem for elem in query.elements if elem.text not in common_values]
    remaining_candidate = [elem for elem in candidate.elements if elem.text not in common_values]

    max_possible = identical_score + min(len(remaining_query), len(remaining_candidate))
    threshold_score = relatedness_threshold * query.size
    if max_possible + EPS < threshold_score:
        return 0.0

    weights: List[List[float]] = []
    for query_elem in remaining_query:
        row: List[float] = []
        for candidate_elem in remaining_candidate:
            if query_elem.tokens.isdisjoint(candidate_elem.tokens):
                row.append(0.0)
            else:
                row.append(jaccard_similarity(query_elem.tokens, candidate_elem.tokens))
        weights.append(row)

    match_score = identical_score + hungarian_max_weight(weights)
    if metric == "containment":
        return match_score / query.size if query.size else 0.0

    denom = query.size + candidate.size - match_score
    return match_score / denom if denom > 0 else 0.0


def _verify_candidate_chunk(args) -> List[Tuple[int, float]]:
    query, candidate_ids, metric, relatedness_threshold = args
    scored: List[Tuple[int, float]] = []
    for candidate_idx in candidate_ids:
        score = verify_candidate(
            query=query,
            candidate=_SCORE_DATALAKE_COLUMNS[candidate_idx],
            metric=metric,
            relatedness_threshold=relatedness_threshold,
        )
        if score + EPS >= relatedness_threshold:
            scored.append((candidate_idx, score))
    return scored


def verify_candidates(
    query: ColumnSet,
    candidate_ids: Sequence[int],
    datalake_columns: Sequence[ColumnSet],
    pool,
    workers: int,
    metric: str,
    relatedness_threshold: float,
) -> List[Tuple[ColumnSet, float]]:
    if pool is None or workers <= 1 or len(candidate_ids) <= 1:
        scored: List[Tuple[ColumnSet, float]] = []
        for candidate_idx in candidate_ids:
            candidate = datalake_columns[candidate_idx]
            score = verify_candidate(
                query=query,
                candidate=candidate,
                metric=metric,
                relatedness_threshold=relatedness_threshold,
            )
            if score + EPS >= relatedness_threshold:
                scored.append((candidate, score))
    else:
        chunk_count = min(workers, len(candidate_ids))
        chunk_size = math.ceil(len(candidate_ids) / chunk_count)
        tasks = [
            (query, list(candidate_ids[start : start + chunk_size]), metric, relatedness_threshold)
            for start in range(0, len(candidate_ids), chunk_size)
        ]

        scored = []
        for chunk in pool.map(_verify_candidate_chunk, tasks):
            scored.extend((datalake_columns[candidate_idx], score) for candidate_idx, score in chunk)

    scored.sort(key=lambda item: (-item[1], item[0].table_name, item[0].column_name))
    return scored


def score_query_against_datalake(
    query: ColumnSet,
    datalake_columns: Sequence[ColumnSet],
    inverted_index: Dict[str, array],
    posting_counts: Dict[str, int],
    metric: str,
    relatedness_threshold: float,
    include_self_matches: bool,
    score_pool,
    score_workers: int,
) -> Tuple[List[Tuple[ColumnSet, float]], Dict[str, int]]:
    # SILKMOTH paper: θ = δ |R|
    theta = relatedness_threshold * query.size

    # Section 4: weighted signature generation
    signature = build_query_signature(
        query,
        posting_counts,
        threshold=theta,
    )

    # Algorithm 1: candidate selection + check filter
    initial_candidates = select_candidates(
        query=query,
        signature=signature,
        datalake_columns=datalake_columns,
        inverted_index=inverted_index,
        include_self_matches=include_self_matches,
        metric=metric,
        relatedness_threshold=relatedness_threshold,
    )

    # Algorithm 2: nearest-neighbor refinement
    refined_candidates = nearest_neighbor_filter(
        query=query,
        signature=signature,
        candidates=initial_candidates,
        datalake_columns=datalake_columns,
    )

    # Final verification: exact maximum bipartite matching
    scored = verify_candidates(
        query=query,
        candidate_ids=list(refined_candidates),
        datalake_columns=datalake_columns,
        pool=score_pool,
        workers=score_workers,
        metric=metric,
        relatedness_threshold=relatedness_threshold,
    )

    stats = {
        "theta": theta,
        "signature_tokens": sum(len(tokens) for tokens in signature.per_element_tokens),
        "signature_candidates": len(initial_candidates),
        "nn_candidates": len(refined_candidates),
        "verified_matches": len(scored),
    }
    return scored, stats


def build_query_column(spec: QuerySpec) -> Optional[ColumnSet]:
    selected = discover_selected_columns(spec.csv_path, spec.query_column)
    if not selected:
        LOGGER.warning("Query column not found: %s.%s", spec.query_table, spec.query_column)
        return None

    values_by_column = read_selected_column_values(spec.csv_path, selected)
    values = values_by_column.get(selected[0], [])
    return build_column_set(spec.query_table, selected[0], spec.csv_path, values)


def run(args: argparse.Namespace) -> int:
    if args.datalake_dir:
        datalake_dir = resolve_datalake_dir(Path(args.datalake_dir).expanduser().resolve())
        dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else datalake_dir.parent
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        datalake_dir = resolve_datalake_dir(dataset_dir)
    else:
        raise ValueError("Provide either --dataset_dir or --datalake_dir")

    if not datalake_dir.is_dir():
        raise FileNotFoundError(f"Datalake directory not found: {datalake_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    query_source = Path(args.query_source).expanduser().resolve() if args.query_source else None
    default_query_column = args.default_query_column or args.column_name

    LOGGER.info("dataset_dir=%s", dataset_dir)
    LOGGER.info("datalake_dir=%s", datalake_dir)
    LOGGER.info(
        "metric=%s, threshold=%.3f, column_name=%s",
        args.metric,
        args.relatedness_threshold,
        args.column_name,
    )

    build_start = time.perf_counter()
    datalake_columns, inverted_index, posting_counts = build_datalake_index(
        datalake_dir=datalake_dir,
        column_name=args.column_name,
        max_datalake_tables=args.max_datalake_tables,
    )
    build_seconds = time.perf_counter() - build_start
    LOGGER.info("Indexed datalake columns: %s", len(datalake_columns))
    LOGGER.info("Indexed tokens: %s", len(inverted_index))
    LOGGER.info("[TIMING] index_build_seconds=%.3f", build_seconds)

    query_specs, query_source_desc = discover_query_specs(
        dataset_dir=dataset_dir,
        datalake_dir=datalake_dir,
        query_source=query_source,
        default_query_column=default_query_column,
    )
    if args.max_queries > 0:
        query_specs = query_specs[: args.max_queries]
    if not query_specs:
        raise ValueError("No query columns discovered.")
    LOGGER.info("Query source: %s (%s query columns)", query_source_desc, len(query_specs))

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    score_workers = max(1, args.score_workers)
    score_pool = None
    if score_workers > 1:
        global _SCORE_DATALAKE_COLUMNS
        _SCORE_DATALAKE_COLUMNS = datalake_columns
        score_pool = multiprocessing.get_context("fork").Pool(processes=score_workers)
        LOGGER.info("Using %s CPU workers for candidate verification", score_workers)

    total_written = 0
    resource_log_csv = default_resource_log_path(str(out_path), args.resource_log_csv)
    resources = ResourceMonitor(resource_log_csv, args.resource_sample_interval).start()
    online_start = time.perf_counter()
    try:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query_table",
                    "query_column",
                    "candidate_table",
                    "candidate_column",
                    "similarity_score",
                ]
            )

            for query_idx, spec in enumerate(query_specs, start=1):
                query = build_query_column(spec)
                if query is None:
                    continue

                start = time.perf_counter()
                scored, stats = score_query_against_datalake(
                    query=query,
                    datalake_columns=datalake_columns,
                    inverted_index=inverted_index,
                    posting_counts=posting_counts,
                    metric=args.metric,
                    relatedness_threshold=args.relatedness_threshold,
                    include_self_matches=args.include_self_matches,
                    score_pool=score_pool,
                    score_workers=score_workers,
                )
                query_seconds = time.perf_counter() - start

                for candidate, score in scored:
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
                    "[%s/%s] %s.%s size=%s signature_tokens=%s candidates=%s nn=%s verified=%s kept=%s t=%.3fs",
                    query_idx,
                    len(query_specs),
                    query.table_name,
                    query.column_name,
                    query.size,
                    stats["signature_tokens"],
                    stats["signature_candidates"],
                    stats["nn_candidates"],
                    stats["verified_matches"],
                    len(scored),
                    query_seconds,
                )
    finally:
        if score_pool is not None:
            score_pool.close()
            score_pool.join()

    online_seconds = time.perf_counter() - online_start
    resources.stop()
    LOGGER.info("Wrote ranked results: %s (%s rows)", out_path, total_written)
    LOGGER.info("[TIMING] online_query_seconds=%.3f", online_seconds)
    log_resource_summary(resources.summary(), LOGGER.info)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal SILKMOTH baseline for SemSketch datasets.")
    parser.add_argument("--dataset_dir", help="Dataset root (contains datalake/ and query metadata).")
    parser.add_argument(
        "--datalake_dir",
        help="Path to datalake directory (or parent directory that contains datalake/).",
    )
    parser.add_argument(
        "--query_source",
        help=(
            "Optional query source: CSV file (e.g. autofj_query_columns.csv) or "
            "directory of query CSVs. If omitted, query files are auto-discovered."
        ),
    )
    parser.add_argument(
        "--column_name",
        default="title",
        help="Indexed datalake column name (default: title). Use '*' or 'all' for all columns.",
    )
    parser.add_argument(
        "--default_query_column",
        default="",
        help="Fallback query column name when query metadata does not specify one.",
    )
    parser.add_argument(
        "--metric",
        choices=["containment", "similarity"],
        default="containment",
        help="Set-relatedness metric used in final verification.",
    )
    parser.add_argument(
        "--relatedness_threshold",
        type=float,
        default=0.7,
        help="Relatedness threshold delta from the paper (default: 0.7).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help=(
            "Optional output truncation only. SILKMOTH itself is threshold-based "
            "and verifies all refined candidates. Use 0 to keep all."
        ),
    )
    parser.add_argument(
        "--include_self_matches",
        action="store_true",
        help="Include candidates from the same table as the query.",
    )
    parser.add_argument(
        "--max_datalake_tables",
        type=int,
        default=0,
        help="Optional cap on datalake CSV files to index (0 = all).",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="Optional cap on query columns to process (0 = all).",
    )
    parser.add_argument(
        "--score_workers",
        type=int,
        default=int(os.environ.get("SILKMOTH_SCORE_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="CPU workers for candidate verification.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path with query/candidate/similarity rows.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    add_resource_monitor_args(parser)
    return parser


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
    raise SystemExit(main())
