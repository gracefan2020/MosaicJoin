#!/usr/bin/env python3
"""
Simplified PEXESO baseline runner for SemSketch benchmarks.

This script adapts the core PEXESO search flow (pivot mapping + hierarchical grid
block-and-verify) to this repository's dataset layouts.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "model"

FASTTEXT_DEFAULT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
)
FASTTEXT_DEFAULT_ZIP = "crawl-300d-2M-subword.zip"


# ---------------------------------------------------------------------------
# PEXESO core structures
# ---------------------------------------------------------------------------


class Grid:
    def __init__(self, grid_id: int, level: int, max_level: int, center: np.ndarray, length: float):
        self.id = grid_id
        self.level = level
        self.max_level = max_level
        self.center = center
        self.length = length
        self.child: Dict[int, Grid] = {}
        self.vector: List[np.ndarray] = []
        self.vec_ids: List[int] = []
        self.embeddings: List[np.ndarray] = []

    def is_leaf(self) -> bool:
        return self.level == self.max_level

    def leaves(self) -> List["Grid"]:
        if self.is_leaf():
            return [self]
        out: List[Grid] = []
        for ch in self.child.values():
            out.extend(ch.leaves())
        return out


class HierarchicalGrid:
    def __init__(self, n_dims: int, n_layers: int, origin: float, total_length: float):
        center = np.full((n_dims,), origin + total_length / 2.0, dtype=np.float64)
        self.root = Grid(-1, 0, n_layers, center, float(total_length))
        self.base = 2
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.total_length = float(total_length)

    def _grid_id(self, bins: Sequence[int], previous: int, parts: int) -> int:
        decimal = 0
        power = 0
        for b in bins:
            decimal += int(b) * (self.base ** power)
            power += 1
        return decimal + previous * parts

    def add_vector(self, vector: np.ndarray, vec_id: int, emb: np.ndarray) -> Grid:
        node = self.root
        previous_id = 0
        parts = self.base ** self.n_dims
        current_len = self.total_length
        remainder = vector.copy()
        center = node.center.copy()

        for level in range(self.n_layers):
            bins: List[int] = []
            current_len /= self.base
            for dim in range(self.n_dims):
                if current_len <= 0:
                    b = 0
                else:
                    b = int(remainder[dim] / current_len)
                bins.append(b)
                remainder[dim] = remainder[dim] % current_len if current_len > 0 else 0.0
                center[dim] = vector[dim] - remainder[dim] + current_len / 2.0

            grid_id = self._grid_id(bins, previous_id, parts)
            previous_id = grid_id
            nxt = node.child.get(grid_id)
            if nxt is None:
                nxt = Grid(grid_id, level + 1, self.n_layers, center.copy(), current_len)
                node.child[grid_id] = nxt
            if nxt.is_leaf():
                nxt.vector.append(vector)
                nxt.vec_ids.append(vec_id)
                nxt.embeddings.append(emb)
            node = nxt
        return node


class InvertedIndex:
    def __init__(self):
        self.index: Dict[Grid, Set[int]] = {}

    def add(self, grid: Grid, col_id: int) -> None:
        self.index.setdefault(grid, set()).add(col_id)

    def search(self, grid: Grid) -> Set[int]:
        return self.index.get(grid, set())


@dataclass
class Pair:
    q_point: np.ndarray
    q_embedding: np.ndarray
    candidate_grids: List[Grid]
    matched_grids: List[Grid]


def _add_pair(
    pairs: Dict[int, Pair],
    q_id: int,
    q_point: np.ndarray,
    q_embedding: np.ndarray,
    grid: Grid,
    matched: bool,
) -> None:
    pair = pairs.get(q_id)
    if pair is None:
        pair = Pair(
            q_point=q_point,
            q_embedding=q_embedding,
            candidate_grids=[],
            matched_grids=[],
        )
        pairs[q_id] = pair
    if matched:
        pair.matched_grids.append(grid)
    else:
        pair.candidate_grids.append(grid)


def _add_pairs(
    pairs: Dict[int, Pair],
    q_id: int,
    q_point: np.ndarray,
    q_embedding: np.ndarray,
    grids: Iterable[Grid],
    matched: bool,
) -> None:
    pair = pairs.get(q_id)
    if pair is None:
        pair = Pair(
            q_point=q_point,
            q_embedding=q_embedding,
            candidate_grids=[],
            matched_grids=[],
        )
        pairs[q_id] = pair
    if matched:
        pair.matched_grids.extend(grids)
    else:
        pair.candidate_grids.extend(grids)


def _filter_axis(query: np.ndarray, point: np.ndarray, tau: float) -> bool:
    for dim in range(len(query)):
        if query[dim] - tau > point[dim] or query[dim] + tau < point[dim]:
            return True
    return False


def _match_axis(query: np.ndarray, point: np.ndarray, tau: float) -> bool:
    for dim in range(len(query)):
        if query[dim] + point[dim] <= tau:
            return True
    return False


def _filter_point_grid(query: np.ndarray, grid: Grid, tau: float) -> bool:
    for dim in range(len(query)):
        if abs(grid.center[dim] - query[dim]) > grid.length / 2.0 + tau:
            return True
    return False


def _filter_grid_grid(query_grid: Grid, grid: Grid, tau: float) -> bool:
    for dim in range(len(query_grid.center)):
        if abs(grid.center[dim] - query_grid.center[dim]) > grid.length / 2.0 + query_grid.length / 2.0 + tau:
            return True
    return False


def _match_point_grid(query: np.ndarray, grid: Grid, tau: float) -> bool:
    for dim in range(len(query)):
        if tau - query[dim] >= grid.center[dim] + grid.length / 2.0:
            return True
    return False


def _match_grid_grid(query_grid: Grid, grid: Grid, tau: float) -> bool:
    for dim in range(len(query_grid.center)):
        if tau - (query_grid.center[dim] + query_grid.length / 2.0) >= grid.center[dim] + grid.length / 2.0:
            return True
    return False


def block(query_grid: Grid, index_grid: Grid, pairs: Dict[int, Pair], tau: float) -> None:
    for query_child in query_grid.child.values():
        for index_child in index_grid.child.values():
            if query_child.is_leaf() and index_child.is_leaf():
                for i, q_point in enumerate(query_child.vector):
                    q_id = query_child.vec_ids[i]
                    q_emb = query_child.embeddings[i]
                    if _match_point_grid(q_point, index_child, tau):
                        _add_pair(pairs, q_id, q_point, q_emb, index_child, matched=True)
                    elif not _filter_point_grid(q_point, index_child, tau):
                        _add_pair(pairs, q_id, q_point, q_emb, index_child, matched=False)
            elif _match_grid_grid(query_child, index_child, tau):
                leaf_grids = index_child.leaves()
                for q_leaf in query_child.leaves():
                    for i, q_point in enumerate(q_leaf.vector):
                        q_id = q_leaf.vec_ids[i]
                        q_emb = q_leaf.embeddings[i]
                        _add_pairs(pairs, q_id, q_point, q_emb, leaf_grids, matched=True)
            elif not _filter_grid_grid(query_child, index_child, tau):
                block(query_child, index_child, pairs, tau)


def verify(
    pairs: Dict[int, Pair],
    inverted_index: InvertedIndex,
    tau: float,
    threshold_count: float,
    index_sets: Sequence[Set[int]],
    query_size: int,
    time_threshold: float,
) -> Tuple[List[int], List[int], bool]:
    start = time.perf_counter()
    over_time = False

    match_count = [0] * len(index_sets)
    mismatch_count = [0] * len(index_sets)

    for pair in pairs.values():
        for grid in pair.matched_grids:
            for col in inverted_index.search(grid):
                match_count[col] += 1

    for pair in pairs.values():
        if over_time:
            break
        for grid in pair.candidate_grids:
            if time.perf_counter() - start > time_threshold:
                over_time = True
                break
            for col in inverted_index.search(grid):
                if time.perf_counter() - start > time_threshold:
                    over_time = True
                    break
                # Same pruning as the original code path.
                if query_size - mismatch_count[col] < threshold_count * 2:
                    continue
                for i, grid_point in enumerate(grid.vector):
                    if time.perf_counter() - start > time_threshold:
                        over_time = True
                        break
                    if grid.vec_ids[i] not in index_sets[col]:
                        continue
                    if match_count[col] >= threshold_count:
                        break
                    if _filter_axis(grid_point, pair.q_point, tau):
                        mismatch_count[col] += 1
                    elif _match_axis(grid_point, pair.q_point, tau):
                        match_count[col] += 1
                    else:
                        distance = float(np.linalg.norm(grid.embeddings[i] - pair.q_embedding))
                        if distance <= tau:
                            match_count[col] += 1
                        else:
                            mismatch_count[col] += 1

    result = [idx for idx, cnt in enumerate(match_count) if cnt >= threshold_count]
    return result, match_count, over_time


def build_hierarchical_grid(
    points: np.ndarray,
    embeddings: np.ndarray,
    n_layers: int,
    x_min: float,
    x_max: float,
) -> Tuple[HierarchicalGrid, Dict[int, Grid], float]:
    if x_max <= x_min:
        x_max = x_min + 1.0
    n_dims = int(points.shape[1])
    total_length = float(x_max - x_min)
    grid = HierarchicalGrid(n_dims=n_dims, n_layers=n_layers, origin=x_min, total_length=total_length)
    id_to_grid: Dict[int, Grid] = {}
    for idx in range(points.shape[0]):
        id_to_grid[idx] = grid.add_vector(points[idx], idx, embeddings[idx])
    return grid, id_to_grid, total_length / (2 ** n_layers)


# ---------------------------------------------------------------------------
# Embeddings + data handling
# ---------------------------------------------------------------------------


@dataclass
class QuerySpec:
    query_table: str
    query_column: str
    csv_path: Path


class ValueEmbedder:
    def __init__(
        self,
        mode: str,
        dim: int,
        seed: int,
        embedding_pickle: Optional[Path],
        mpnet_model: Optional[str],
        batch_size: int,
    ):
        self.mode = mode
        self.dim = dim
        self.seed = seed
        self.embedding_pickle = embedding_pickle
        self.mpnet_model = mpnet_model
        self.batch_size = batch_size
        self._value_cache: Dict[str, Optional[np.ndarray]] = {}
        self.external_vectors: Optional[Dict[str, np.ndarray]] = None
        self._mpnet = None
        self._fasttext = None
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if self.mode == "glove":
            if embedding_pickle is None:
                raise ValueError("--embedding_mode glove requires --embedding_pickle")
            with embedding_pickle.open("rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                raise ValueError(f"Embedding pickle must be a dict, got {type(payload)}")
            vectors: Dict[str, np.ndarray] = {}
            inferred_dim = None
            for token, vec in payload.items():
                arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim != 1:
                    continue
                if inferred_dim is None:
                    inferred_dim = int(arr.shape[0])
                if arr.shape[0] == inferred_dim:
                    vectors[str(token)] = arr
            if not vectors:
                raise ValueError(f"No valid token vectors found in {embedding_pickle}")
            self.external_vectors = vectors
            self.dim = inferred_dim if inferred_dim is not None else self.dim

        if self.mode == "fasttext":
            kind, path = self._ensure_fasttext_artifact()
            if kind == "bin":
                self._fasttext = self._load_fasttext_bin(path)
                self.dim = int(self._fasttext.get_dimension())
            else:
                self.external_vectors, self.dim = self._load_fasttext_vectors(path)

        if self.mode == "mpnet":
            if mpnet_model is None:
                raise ValueError("--embedding_mode mpnet requires --mpnet_model")
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers is required for --embedding_mode mpnet"
                ) from exc
            self._mpnet = SentenceTransformer(mpnet_model, cache_folder=str(MODEL_DIR))

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _unit_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    def _token_vector(self, token: str) -> Optional[np.ndarray]:
        if self.mode in {"glove", "fasttext"}:
            if self.mode == "glove":
                assert self.external_vectors is not None
                vec = self.external_vectors.get(token)
                if vec is None:
                    return None
            else:
                if self._fasttext is not None:
                    vec = self._fasttext.get_word_vector(token)
                else:
                    assert self.external_vectors is not None
                    vec = self.external_vectors.get(token)
                    if vec is None:
                        return None
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                return None
            return (vec / norm).astype(np.float32)
        return None

    def embed_value(self, value: str) -> Optional[np.ndarray]:
        if self.mode == "mpnet":
            raise RuntimeError("embed_value is not supported in mpnet mode; use embed_values")
        cached = self._value_cache.get(value)
        if cached is not None or value in self._value_cache:
            return cached

        tokens = self._tokenize(value)
        if not tokens:
            self._value_cache[value] = None
            return None
        vectors: List[np.ndarray] = []
        for token in tokens:
            vec = self._token_vector(token)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            self._value_cache[value] = None
            return None

        arr = np.stack(vectors, axis=0)
        avg = arr.mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg))
        if norm <= 1e-12:
            self._value_cache[value] = None
            return None
        avg = avg / norm
        self._value_cache[value] = avg
        return avg

    def embed_values(self, values: List[str]) -> List[Optional[np.ndarray]]:
        results: List[Optional[np.ndarray]] = [None] * len(values)
        pending: List[str] = []
        pending_idx: List[int] = []

        for idx, value in enumerate(values):
            cached = self._value_cache.get(value)
            if cached is not None or value in self._value_cache:
                results[idx] = cached
            else:
                pending.append(value)
                pending_idx.append(idx)

        if not pending:
            return results

        if self.mode == "mpnet":
            assert self._mpnet is not None
            embeddings = self._mpnet.encode(
                pending,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            for value, emb, idx in zip(pending, embeddings, pending_idx):
                norm = float(np.linalg.norm(emb))
                if norm <= 1e-12:
                    self._value_cache[value] = None
                    results[idx] = None
                else:
                    normalized = emb / norm
                    self._value_cache[value] = normalized
                    results[idx] = normalized
            return results

        for value, idx in zip(pending, pending_idx):
            results[idx] = self.embed_value(value)
        return results

    def _load_fasttext_bin(self, model_path: Path):
        try:
            import fasttext
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "fasttext import failed. On macOS, this often means the wheel was built "
                "for a newer OS. Try: `pip uninstall -y fasttext` then "
                "`pip install --no-binary :all: fasttext` (requires Xcode CLT), "
                "or install via conda-forge."
            ) from exc

        return fasttext.load_model(str(model_path))

    def _load_fasttext_vectors(self, path: Path) -> Tuple[Dict[str, np.ndarray], int]:
        vectors: Dict[str, np.ndarray] = {}
        inferred_dim = None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            parts = first.strip().split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                inferred_dim = int(parts[1])
            else:
                token = parts[0]
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                inferred_dim = vec.shape[0]
                vectors[token] = vec

            for line_idx, line in enumerate(f, start=1):
                parts = line.rstrip().split()
                if len(parts) <= 2:
                    continue
                token = parts[0]
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
                if inferred_dim is None:
                    inferred_dim = vec.shape[0]
                if vec.shape[0] != inferred_dim:
                    continue
                vectors[token] = vec
                if line_idx % 200000 == 0:
                    print(f"[INFO] loaded {len(vectors)} fasttext vectors")

        if not vectors or inferred_dim is None:
            raise ValueError(f"No vectors loaded from {path}")
        return vectors, inferred_dim

    def _ensure_fasttext_artifact(self) -> Tuple[str, Path]:
        cache_dir = MODEL_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / FASTTEXT_DEFAULT_ZIP

        if not zip_path.is_file():
            self._download_file(FASTTEXT_DEFAULT_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            bin_files = [n for n in names if n.endswith(".bin")]
            vec_files = [n for n in names if n.endswith(".vec")]

            if bin_files:
                target = bin_files[0]
                kind = "bin"
            elif vec_files:
                target = vec_files[0]
                kind = "vec"
            else:
                raise ValueError(f"No .bin or .vec found in {zip_path}")

            out_path = cache_dir / Path(target).name
            if not out_path.is_file():
                zf.extract(target, path=cache_dir)
                extracted = cache_dir / target
                if extracted != out_path:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    extracted.replace(out_path)

        if kind == "vec":
            print(f"[WARN] fastText archive has no .bin; using .vec vectors from {out_path}")
        return kind, out_path

    def _download_file(self, url: str, dest: Path) -> None:
        print(f"[INFO] downloading fastText vectors to {MODEL_DIR}: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, dest.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_size = int(total) if total and total.isdigit() else None
            downloaded = 0
            chunk_size = 1024 * 1024 * 8
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded / total_size * 100
                    if int(pct) % 10 == 0:
                        print(f"[INFO] download {pct:.0f}%")


def _normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def _has_csv_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() == ".csv":
            return True
    return False


def resolve_datalake_dir(path: Path) -> Path:
    nested = path / "datalake"
    if _has_csv_files(nested):
        return nested
    if _has_csv_files(path):
        return path
    return path


def _canonical_column_name(fieldnames: Sequence[str], target: str) -> Optional[str]:
    target_lower = target.strip().lower()
    for col in fieldnames:
        if col.strip().lower() == target_lower:
            return col
    return None


def _read_columns(
    csv_path: Path,
    selected_columns: Sequence[str],
) -> Dict[str, Set[str]]:
    values: Dict[str, Set[str]] = {col: set() for col in selected_columns}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return values
        for row in reader:
            for col in selected_columns:
                raw = row.get(col, "")
                norm = _normalize_text(raw)
                if norm:
                    values[col].add(norm)
    return values


def _discover_selected_columns(csv_path: Path, column_name: str) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
    if not header:
        return []

    if column_name.strip().lower() in {"*", "all"}:
        cols = []
        for col in header:
            if col.strip().lower() in {"id", "index"}:
                continue
            cols.append(col)
        return cols

    matched = _canonical_column_name(header, column_name)
    return [matched] if matched else []


def _resolve_table_path(table_ref: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    ref = (table_ref or "").strip()
    if not ref:
        return None
    ref_path = Path(ref)
    if ref_path.is_file():
        return ref_path
    if ref_path.suffix.lower() != ".csv":
        ref_path_csv = Path(f"{ref}.csv")
        if ref_path_csv.is_file():
            return ref_path_csv
    for base in search_dirs:
        if not base:
            continue
        candidate = base / ref
        if candidate.is_file():
            return candidate
        if candidate.suffix.lower() != ".csv":
            candidate_csv = candidate.with_suffix(".csv")
            if candidate_csv.is_file():
                return candidate_csv
    return None


def _parse_query_pairs(query_file: Path, default_query_column: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with query_file.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pairs

        header_lower = [h.strip().lower() for h in header]
        candidates = [
            ("target_ds", "target_attr"),
            ("query_table", "query_column"),
            ("source_table", "source_column"),
            ("table_name", "col_name"),
            ("table_name", "column_name"),
            ("table", "column"),
        ]
        idx_pair: Optional[Tuple[int, int]] = None
        for table_key, col_key in candidates:
            if table_key in header_lower and col_key in header_lower:
                idx_pair = (header_lower.index(table_key), header_lower.index(col_key))
                break

        if idx_pair is None and "left_table" in header_lower:
            table_idx = header_lower.index("left_table")
            seen = set()
            for row in reader:
                if len(row) <= table_idx:
                    continue
                table_name = row[table_idx].strip()
                if table_name and table_name not in seen:
                    pairs.append((table_name, default_query_column))
                    seen.add(table_name)
            return pairs

        if idx_pair is None:
            for row in reader:
                if len(row) >= 2:
                    table_name = row[0].strip()
                    col_name = row[1].strip() or default_query_column
                    if table_name:
                        pairs.append((table_name, col_name))
            return pairs

        table_idx, col_idx = idx_pair
        for row in reader:
            if len(row) <= max(table_idx, col_idx):
                continue
            table_name = row[table_idx].strip()
            col_name = row[col_idx].strip() or default_query_column
            if table_name:
                pairs.append((table_name, col_name))
    return pairs


def _discover_query_specs(
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
        default_file = dataset_dir / "autofj_query_columns.csv"
        ground_truth = dataset_dir / "groundtruth-joinable.csv"
        if default_file.is_file():
            source = default_file
        elif queries_dir.is_dir() and _has_csv_files(queries_dir):
            source = queries_dir
        elif ground_truth.is_file():
            source = ground_truth
        else:
            source = datalake_dir

    specs: List[QuerySpec] = []
    if source.is_file():
        pairs = _parse_query_pairs(source, default_query_column=default_query_column)
        seen = set()
        for table_ref, query_col in pairs:
            table_path = _resolve_table_path(table_ref, search_dirs=search_dirs)
            if table_path is None:
                print(f"[WARN] query table not found: {table_ref}")
                continue
            query_table = table_path.name
            key = (query_table, query_col)
            if key in seen:
                continue
            seen.add(key)
            specs.append(QuerySpec(query_table=query_table, query_column=query_col, csv_path=table_path))
        return specs, f"file:{source}"

    if source.is_dir():
        for csv_path in sorted(source.glob("*.csv")):
            selected = _discover_selected_columns(csv_path, default_query_column)
            for col_name in selected:
                specs.append(
                    QuerySpec(query_table=csv_path.name, query_column=col_name, csv_path=csv_path)
                )
        return specs, f"dir:{source}"

    raise ValueError(f"Query source does not exist: {source}")


def _select_farthest_pivots(points: np.ndarray, num_pivots: int, seed: int) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        raise ValueError("Cannot select pivots from empty point set")
    num_pivots = max(1, min(num_pivots, n))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = [first]
    min_dist = np.linalg.norm(points - points[first], axis=1)

    for _ in range(1, num_pivots):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist = np.linalg.norm(points - points[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return points[np.asarray(selected, dtype=np.int32)]


def _distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64))


def _select_lof_pivots(points: np.ndarray, num_pivots: int, seed: int) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        from sklearn.neighbors import LocalOutlierFactor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "scikit-learn is required for pivot_method=lof; install it or use --pivot_method farthest"
        ) from exc

    data = np.asarray(points, dtype=np.float64)
    n = data.shape[0]
    if n == 0:
        raise ValueError("Cannot select pivots from empty point set")
    num_pivots = max(1, min(num_pivots, n))
    c = 5 if n > 5 * num_pivots else max(1, int(n / num_pivots))

    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    y_pred = lof.fit_predict(data)
    outliers = data[y_pred == -1]
    top_n_outliers = outliers[: c * num_pivots]
    if len(top_n_outliers) < num_pivots:
        top_n_outliers = data[:num_pivots]

    distances = _distance_matrix(data, top_n_outliers)
    k1 = num_pivots if num_pivots < len(top_n_outliers) else len(top_n_outliers)
    pca = PCA(n_components=k1)
    pca.fit(distances)
    components = pca.components_

    selected: List[np.ndarray] = []
    used: Set[int] = set()
    for i in range(k1):
        scores: List[Tuple[float, int]] = []
        for j in range(len(top_n_outliers)):
            idxs = np.where(np.all(data == top_n_outliers[j], axis=1))[0]
            if len(idxs) == 0:
                continue
            index = int(idxs[0])
            proj = float(np.dot(components[i], distances[index]))
            scores.append((abs(proj), index))
        scores.sort()
        for _, index in scores:
            if index not in used:
                used.add(index)
                selected.append(data[index])
                break

    if len(selected) < num_pivots:
        # Fill with farthest-first to reach requested count.
        remaining = num_pivots - len(selected)
        extras = _select_farthest_pivots(data, remaining, seed)
        selected.extend(list(extras))
    return np.asarray(selected[:num_pivots], dtype=np.float64)


def _map_points_to_pivots(points: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    # shape: (num_points, num_pivots)
    # Broadcasting distance computation without external dependencies.
    diff = points[:, np.newaxis, :] - pivots[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64))
    return distances.astype(np.float64, copy=False)


def _build_index_payload(
    datalake_dir: Path,
    column_name: str,
    embedder: ValueEmbedder,
    num_pivots: int,
    num_layers: int,
    seed: int,
    pivot_method: str,
) -> Dict:
    print(f"[INFO] Building PEXESO index from {datalake_dir}")
    column_items: List[Tuple[str, str]] = []
    column_sets: List[Set[int]] = []
    value_to_id: Dict[str, int] = {}
    value_embeddings: List[np.ndarray] = []

    csv_files = sorted(datalake_dir.glob("*.csv"))
    for idx, csv_path in enumerate(csv_files):
        selected_cols = _discover_selected_columns(csv_path, column_name)
        if not selected_cols:
            continue
        values_by_col = _read_columns(csv_path, selected_cols)
        for col in selected_cols:
            ids: Set[int] = set()
            values = list(values_by_col.get(col, set()))
            if not values:
                continue
            embeddings = embedder.embed_values(values)
            for value, vec in zip(values, embeddings):
                if vec is None:
                    continue
                value_id = value_to_id.get(value)
                if value_id is None:
                    value_id = len(value_embeddings)
                    value_to_id[value] = value_id
                    value_embeddings.append(vec)
                ids.add(value_id)
            if ids:
                column_items.append((csv_path.name, col))
                column_sets.append(ids)
        if (idx + 1) % 500 == 0 or idx + 1 == len(csv_files):
            print(f"[INFO] indexed {idx + 1}/{len(csv_files)} tables")

    if not value_embeddings:
        raise ValueError("No indexable values found in datalake")

    embedding_matrix = np.asarray(value_embeddings, dtype=np.float32)
    if pivot_method == "lof":
        pivots = _select_lof_pivots(embedding_matrix, num_pivots=num_pivots, seed=seed)
    else:
        pivots = _select_farthest_pivots(embedding_matrix, num_pivots=num_pivots, seed=seed)
    mapped_points = _map_points_to_pivots(embedding_matrix, pivots)

    x_min = float(math.floor(float(np.min(mapped_points))))
    x_max = float(math.ceil(float(np.max(mapped_points))))
    if x_max <= x_min:
        x_max = x_min + 1.0

    index_grid, id_to_grid, cell_size = build_hierarchical_grid(
        mapped_points,
        embedding_matrix,
        n_layers=num_layers,
        x_min=x_min,
        x_max=x_max,
    )
    inverted_index = InvertedIndex()
    for col_id, ids in enumerate(column_sets):
        for value_id in ids:
            inverted_index.add(id_to_grid[value_id], col_id)

    print(
        "[INFO] index ready: "
        f"{len(column_items)} columns, {embedding_matrix.shape[0]} unique values, "
        f"{pivots.shape[0]} pivots"
    )

    return {
        "meta": {
            "embedding_mode": embedder.mode,
            "embedding_dim": embedder.dim,
            "embedding_pickle": str(embedder.embedding_pickle) if embedder.embedding_pickle else None,
            "fasttext_url": FASTTEXT_DEFAULT_URL if embedder.mode == "fasttext" else None,
            "fasttext_cache_dir": str(MODEL_DIR),
            "mpnet_model": str(embedder.mpnet_model) if embedder.mpnet_model else None,
            "num_pivots": num_pivots,
            "num_layers": num_layers,
            "pivot_method": pivot_method,
            "column_name": column_name,
            "seed": seed,
        },
        "column_items": column_items,
        "column_sets": column_sets,
        "value_to_id": value_to_id,
        "embedding_matrix": embedding_matrix,
        "pivots": pivots,
        "x_min": x_min,
        "x_max": x_max,
        "num_layers": num_layers,
        "cell_size": cell_size,
        "index_grid": index_grid,
        "inverted_index": inverted_index,
    }


def _load_query_values(csv_path: Path, query_column: str) -> Tuple[Optional[str], List[str]]:
    selected_cols = _discover_selected_columns(csv_path, query_column)
    if not selected_cols:
        return None, []
    canonical = selected_cols[0]
    values_by_col = _read_columns(csv_path, [canonical])
    values = sorted(values_by_col.get(canonical, set()))
    return canonical, values


def _containment_score(query_ids: Set[int], candidate_ids: Set[int]) -> float:
    if not query_ids:
        return 0.0
    inter = len(query_ids.intersection(candidate_ids))
    return float(inter) / float(len(query_ids))


def run_baseline(args: argparse.Namespace) -> int:
    if args.datalake_dir:
        datalake_dir = resolve_datalake_dir(Path(args.datalake_dir))
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else datalake_dir.parent
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        datalake_dir = resolve_datalake_dir(dataset_dir)
    else:
        raise ValueError("Provide either --dataset_dir or --datalake_dir")

    if not datalake_dir.is_dir():
        raise ValueError(f"Datalake directory not found: {datalake_dir}")

    query_source = Path(args.query_source) if args.query_source else None
    query_specs, query_source_desc = _discover_query_specs(
        dataset_dir=dataset_dir,
        datalake_dir=datalake_dir,
        query_source=query_source,
        default_query_column=args.default_query_column,
    )
    if not query_specs:
        raise ValueError("No query columns discovered")
    print(f"[INFO] Query source: {query_source_desc} ({len(query_specs)} query columns)")

    embedding_pickle = Path(args.embedding_pickle) if args.embedding_pickle else None
    if args.embedding_mode == "glove" and embedding_pickle is None:
        candidates: List[Path] = []
        if args.pexeso_repo:
            candidates.append(Path(args.pexeso_repo) / "model" / "glove.pikle")
        for candidate in candidates:
            if candidate.is_file():
                embedding_pickle = candidate
                break
    embedding_mode = args.embedding_mode
    if embedding_mode == "glove" and embedding_pickle is None:
        hint = None
        if args.pexeso_repo:
            hint = str(Path(args.pexeso_repo) / "model" / "glove.pikle")
        msg = "embedding_mode glove requires --embedding_pickle"
        if hint:
            msg += f" (expected at {hint})"
        raise ValueError(msg)
    embedder = ValueEmbedder(
        mode=embedding_mode,
        dim=0,
        seed=args.seed,
        embedding_pickle=embedding_pickle,
        mpnet_model=args.mpnet_model,
        batch_size=args.embedding_batch_size,
    )

    payload: Optional[Dict] = None
    if args.index_cache and Path(args.index_cache).is_file() and not args.rebuild_index:
        cache_path = Path(args.index_cache)
        print(f"[INFO] Loading index cache: {cache_path}")
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        meta = payload.get("meta") if isinstance(payload, dict) else None
        expected_meta = {
            "embedding_mode": embedder.mode,
            "embedding_dim": embedder.dim,
            "embedding_pickle": str(embedder.embedding_pickle) if embedder.embedding_pickle else None,
            "fasttext_url": FASTTEXT_DEFAULT_URL if embedder.mode == "fasttext" else None,
            "fasttext_cache_dir": str(MODEL_DIR),
            "mpnet_model": str(embedder.mpnet_model) if embedder.mpnet_model else None,
            "num_pivots": args.num_pivots,
            "num_layers": args.num_layers,
            "pivot_method": args.pivot_method,
            "column_name": args.column_name,
            "seed": args.seed,
        }
        if not isinstance(meta, dict):
            print("[WARN] index cache missing metadata; rebuilding to avoid mismatched embeddings")
            payload = None
        else:
            for key, value in expected_meta.items():
                if meta.get(key) != value:
                    print(f"[WARN] index cache mismatch for {key}; rebuilding")
                    payload = None
                    break
    if payload is None:
        payload = _build_index_payload(
            datalake_dir=datalake_dir,
            column_name=args.column_name,
            embedder=embedder,
            num_pivots=args.num_pivots,
            num_layers=args.num_layers,
            seed=args.seed,
            pivot_method=args.pivot_method,
        )
        if args.index_cache:
            cache_path = Path(args.index_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(payload, f)
            print(f"[INFO] Saved index cache: {cache_path}")

    column_items: List[Tuple[str, str]] = payload["column_items"]
    column_sets: List[Set[int]] = payload["column_sets"]
    value_to_id: Dict[str, int] = payload["value_to_id"]
    pivots: np.ndarray = payload["pivots"]
    x_min: float = payload["x_min"]
    x_max: float = payload["x_max"]
    num_layers: int = payload["num_layers"]
    index_grid: HierarchicalGrid = payload["index_grid"]
    inverted_index: InvertedIndex = payload["inverted_index"]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped = 0
    start = time.perf_counter()
    with out_csv.open("w", encoding="utf-8", newline="") as f:
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

        for idx, spec in enumerate(query_specs):
            canonical_col, query_values = _load_query_values(spec.csv_path, spec.query_column)
            if not query_values or canonical_col is None:
                skipped += 1
                continue

            query_embeddings: List[np.ndarray] = []
            query_known_ids: Set[int] = set()
            query_vecs = embedder.embed_values(query_values)
            for value, vec in zip(query_values, query_vecs):
                if vec is None:
                    continue
                query_embeddings.append(vec)
                value_id = value_to_id.get(value)
                if value_id is not None:
                    query_known_ids.add(value_id)

            if not query_embeddings:
                skipped += 1
                continue

            query_matrix = np.asarray(query_embeddings, dtype=np.float32)
            query_points = _map_points_to_pivots(query_matrix, pivots)
            query_grid, _, _ = build_hierarchical_grid(
                query_points,
                query_matrix,
                n_layers=num_layers,
                x_min=x_min,
                x_max=x_max,
            )

            pairs: Dict[int, Pair] = {}
            block(query_grid.root, index_grid.root, pairs, args.tau)

            threshold_count = args.containment_threshold * len(query_values)
            candidate_indices, match_count, over_time = verify(
                pairs=pairs,
                inverted_index=inverted_index,
                tau=args.tau,
                threshold_count=threshold_count,
                index_sets=column_sets,
                query_size=len(query_embeddings),
                time_threshold=args.time_threshold,
            )
            if over_time:
                print(
                    f"[WARN] verify timed out for {spec.query_table}.{canonical_col} "
                    f"(>{args.time_threshold}s); partial candidates kept"
                )

            scored: List[Tuple[str, str, float]] = []
            for col_idx in candidate_indices:
                cand_table, cand_col = column_items[col_idx]
                if not args.include_same_table and cand_table == spec.query_table:
                    continue
                overlap_score = _containment_score(query_known_ids, column_sets[col_idx])
                pexeso_score = float(match_count[col_idx]) / float(max(1, len(query_embeddings)))
                score = max(overlap_score, pexeso_score)
                scored.append((cand_table, cand_col, score))

            scored.sort(key=lambda x: x[2], reverse=True)
            if args.top_k > 0:
                scored = scored[: args.top_k]
            for cand_table, cand_col, score in scored:
                writer.writerow(
                    [spec.query_table, canonical_col, cand_table, cand_col, f"{score:.6f}"]
                )
                rows_written += 1

            if (idx + 1) % 20 == 0 or idx + 1 == len(query_specs):
                print(f"[INFO] processed {idx + 1}/{len(query_specs)} queries")

    elapsed = time.perf_counter() - start
    print(
        f"[INFO] Done. Wrote {rows_written} rows to {out_csv} "
        f"(skipped {skipped} queries, elapsed {elapsed:.1f}s)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simplified PEXESO baseline and emit evaluator-compatible CSV."
    )
    parser.add_argument(
        "--dataset_dir",
        help="Dataset root (contains datalake/, optional queries/, autofj_query_columns.csv).",
    )
    parser.add_argument(
        "--datalake_dir",
        help="Path to datalake directory (or parent directory that contains datalake/).",
    )
    parser.add_argument(
        "--query_source",
        help=(
            "Optional query source: CSV file (e.g., autofj_query_columns.csv) or "
            "directory of query tables. If omitted, auto-detects from dataset_dir."
        ),
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path with query/candidate/similarity rows.",
    )
    parser.add_argument(
        "--column_name",
        default="title",
        help="Indexed datalake column name (default: title). Use '*' or 'all' for all columns.",
    )
    parser.add_argument(
        "--default_query_column",
        default="title",
        help="Fallback query column when query source only provides table names.",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k candidates per query.")
    parser.add_argument(
        "--include_same_table",
        action="store_true",
        help="Include candidates from the same table as query.",
    )
    parser.add_argument("--num_pivots", type=int, default=3, help="Number of PEXESO pivots.")
    parser.add_argument("--num_layers", type=int, default=4, help="Hierarchical grid depth.")
    parser.add_argument(
        "--pivot_method",
        choices=["lof", "farthest"],
        default="lof",
        help="Pivot selection method (paper default: lof).",
    )
    parser.add_argument("--tau", type=float, default=0.6, help="Distance threshold (tau).")
    parser.add_argument(
        "--containment_threshold",
        type=float,
        default=0.4,
        help="Minimum matched-ratio threshold used in verification.",
    )
    parser.add_argument(
        "--time_threshold",
        type=float,
        default=30.0,
        help="Per-query verification timeout (seconds).",
    )
    parser.add_argument(
        "--embedding_mode",
        choices=["glove", "fasttext", "mpnet"],
        default="mpnet",
        help="Value embedding mode: glove (paper default), fasttext, or mpnet (default).",
    )
    parser.add_argument(
        "--embedding_pickle",
        help="Token->vector pickle path (required for glove mode).",
    )
    parser.add_argument(
        "--pexeso_repo",
        default="/Users/yifanwu/Desktop/VIDA/tmp/LakeBench/join/Pexeso",
        help="Optional PEXESO repo path to locate model/glove.pikle.",
    )
    parser.add_argument(
        "--mpnet_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="MPNet v2 base model name or local path (default: HF).",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=256,
        help="Batch size for mpnet embeddings.",
    )
    parser.add_argument("--seed", type=int, default=128, help="Random seed.")
    parser.add_argument("--index_cache", help="Optional pickle path to cache the built index.")
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Force rebuilding index even when --index_cache exists.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_baseline(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
