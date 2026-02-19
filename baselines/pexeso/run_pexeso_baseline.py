#!/usr/bin/env python3
"""
Simplified PEXESO baseline runner for SemSketch benchmarks.

This script adapts the core PEXESO search flow (pivot mapping + hierarchical grid
block-and-verify) to this repository's dataset layouts.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import pickle
import time
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent / "model"

FASTTEXT_DEFAULT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
)
FASTTEXT_DEFAULT_ZIP = "crawl-300d-2M-subword.zip"
LOGGER = logging.getLogger(__name__)


def setup_logging(level_name: str = "INFO") -> None:
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested


def log_cuda_runtime() -> None:
    LOGGER.info("torch=%s", torch.__version__)
    LOGGER.info("cuda_available=%s", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return
    try:
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    LOGGER.info("cuda_device_count=%s", count)
    for idx in range(count):
        try:
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_mem_gb = props.total_memory / float(1024 ** 3)
            LOGGER.info(
                "cuda_device[%s] name=%s, capability=%s.%s, mem_gb=%.1f",
                idx,
                name,
                props.major,
                props.minor,
                total_mem_gb,
            )
        except Exception as exc:
            LOGGER.warning("failed to query cuda device %s: %s", idx, exc)


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
        # Finalized leaf arrays used by fast vectorized verification.
        self.vector_arr: Optional[np.ndarray] = None
        self.vec_ids_arr: Optional[np.ndarray] = None
        self.embeddings_arr: Optional[np.ndarray] = None
        self.vec_id_to_local: Optional[Dict[int, int]] = None

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
        self.local_index: Dict[Grid, Dict[int, List[int]]] = {}
        self.local_index_arr: Dict[Grid, Dict[int, np.ndarray]] = {}

    def add(self, grid: Grid, col_id: int, local_idx: Optional[int] = None) -> None:
        self.index.setdefault(grid, set()).add(col_id)
        if local_idx is not None:
            self.local_index.setdefault(grid, {}).setdefault(col_id, []).append(local_idx)

    def search(self, grid: Grid) -> Set[int]:
        return self.index.get(grid, set())

    def finalize(self) -> None:
        arr_map: Dict[Grid, Dict[int, np.ndarray]] = {}
        for grid, col_map in self.local_index.items():
            out: Dict[int, np.ndarray] = {}
            for col, idx_list in col_map.items():
                if not idx_list:
                    continue
                out[col] = np.asarray(sorted(set(idx_list)), dtype=np.int32)
            arr_map[grid] = out
        self.local_index_arr = arr_map
        self.local_index = {}

    def local_indices(self, grid: Grid, col_id: int) -> np.ndarray:
        return self.local_index_arr.get(grid, {}).get(col_id, np.empty((0,), dtype=np.int32))


@dataclass
class Pair:
    q_point: np.ndarray
    q_embedding: np.ndarray
    candidate_grids: List[Grid]
    matched_grids: List[Grid]


@dataclass
class QueryGpuRuntime:
    device: torch.device
    # Cache per-grid tensors to avoid repeated host->device copies in verify().
    grid_tensor_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]


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
    gpu_runtime: Optional[QueryGpuRuntime] = None,
    prune_multiplier: float = 2.0,
) -> Tuple[List[int], List[int], bool]:
    start = time.perf_counter()
    over_time = False

    match_count = [0] * len(index_sets)
    mismatch_count = [0] * len(index_sets)

    # Original PEXESO behavior: every M-grid hit contributes directly.
    for pair in pairs.values():
        for grid in pair.matched_grids:
            for col in inverted_index.search(grid):
                match_count[col] += 1

    for pair in pairs.values():
        if over_time:
            break
        q_point_t = None
        q_embedding_t = None
        if gpu_runtime is not None:
            q_point_t = torch.as_tensor(pair.q_point, device=gpu_runtime.device, dtype=torch.float64)
            q_embedding_t = torch.as_tensor(
                pair.q_embedding,
                device=gpu_runtime.device,
                dtype=torch.float32,
            )
        for grid in pair.candidate_grids:
            if time.perf_counter() - start > time_threshold:
                over_time = True
                break
            cols = inverted_index.search(grid)
            if not cols:
                continue
            for col in cols:
                if time.perf_counter() - start > time_threshold:
                    over_time = True
                    break

                # Lemma 7 pruning (paper code uses T*2 in filter7).
                if query_size - mismatch_count[col] < (prune_multiplier * threshold_count):
                    continue

                idx = inverted_index.local_indices(grid, col)
                if idx.size == 0:
                    continue
                if match_count[col] >= threshold_count:
                    continue

                point_match = _grid_col_match_mask(
                    grid=grid,
                    query_point=pair.q_point,
                    query_embedding=pair.q_embedding,
                    tau=tau,
                    local_indices=idx,
                    query_point_t=q_point_t,
                    query_embedding_t=q_embedding_t,
                    gpu_runtime=gpu_runtime,
                )

                if point_match.size == 0:
                    continue
                needed = float(threshold_count - match_count[col])
                if needed <= 0:
                    continue

                point_match_i = point_match.astype(np.int32, copy=False)
                cum_match = np.cumsum(point_match_i, dtype=np.int64)
                if cum_match[-1] >= needed:
                    prefix_len = int(np.argmax(cum_match >= needed)) + 1
                else:
                    prefix_len = int(point_match.size)

                if prefix_len <= 0:
                    continue
                added_match = int(cum_match[prefix_len - 1])
                added_total = int(prefix_len)
                added_mismatch = added_total - added_match
                match_count[col] += added_match
                mismatch_count[col] += added_mismatch

            if over_time:
                break

    result = [idx for idx, cnt in enumerate(match_count) if cnt >= threshold_count]
    return result, match_count, over_time


def _grid_col_match_mask(
    grid: Grid,
    query_point: np.ndarray,
    query_embedding: np.ndarray,
    tau: float,
    local_indices: np.ndarray,
    query_point_t: Optional[torch.Tensor] = None,
    query_embedding_t: Optional[torch.Tensor] = None,
    gpu_runtime: Optional[QueryGpuRuntime] = None,
) -> np.ndarray:
    if local_indices.size == 0:
        return np.empty((0,), dtype=np.bool_)

    assert grid.vector_arr is not None
    assert grid.embeddings_arr is not None

    if gpu_runtime is not None and query_point_t is not None and query_embedding_t is not None:
        points_t, embs_t = _grid_cuda_tensors(grid, gpu_runtime)
        idx_t = torch.as_tensor(local_indices, device=gpu_runtime.device, dtype=torch.int64)
        points = torch.index_select(points_t, 0, idx_t)

        axis_ok = torch.all(torch.abs(points - query_point_t.unsqueeze(0)) <= tau, dim=1)
        out = torch.zeros((int(local_indices.size),), device=gpu_runtime.device, dtype=torch.bool)
        if bool(torch.any(axis_ok).item()):
            axis_idx = torch.nonzero(axis_ok, as_tuple=False).squeeze(1)
            axis_points = points[axis_idx]
            guaranteed = torch.any(axis_points + query_point_t.unsqueeze(0) <= tau, dim=1)
            if bool(torch.any(guaranteed).item()):
                out[axis_idx[guaranteed]] = True

            remaining = axis_idx[~guaranteed]
            if int(remaining.numel()) > 0:
                rem_idx = idx_t[remaining]
                embs = torch.index_select(embs_t, 0, rem_idx)
                diff = embs - query_embedding_t.unsqueeze(0)
                dist2 = torch.sum(diff * diff, dim=1)
                out[remaining[dist2 <= float(tau * tau)]] = True
        return out.detach().cpu().numpy()

    points = grid.vector_arr[local_indices]
    axis_ok = np.all(np.abs(points - query_point[None, :]) <= tau, axis=1)
    out = np.zeros((int(local_indices.size),), dtype=np.bool_)
    if not np.any(axis_ok):
        return out

    axis_idx = np.where(axis_ok)[0]
    axis_points = points[axis_ok]
    guaranteed = np.any(axis_points + query_point[None, :] <= tau, axis=1)
    if np.any(guaranteed):
        out[axis_idx[guaranteed]] = True

    remaining = axis_idx[~guaranteed]
    if remaining.size == 0:
        return out

    rem_local = local_indices[remaining]
    embs = grid.embeddings_arr[rem_local]
    diff = embs - query_embedding[None, :]
    dist2 = np.sum(diff * diff, axis=1, dtype=np.float32)
    out[remaining[dist2 <= float(tau * tau)]] = True
    return out


def _grid_cuda_tensors(
    grid: Grid,
    gpu_runtime: QueryGpuRuntime,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = id(grid)
    cached = gpu_runtime.grid_tensor_cache.get(cache_key)
    if cached is not None:
        return cached

    assert grid.vector_arr is not None
    assert grid.embeddings_arr is not None
    points_t = torch.as_tensor(grid.vector_arr, device=gpu_runtime.device, dtype=torch.float64)
    embs_t = torch.as_tensor(grid.embeddings_arr, device=gpu_runtime.device, dtype=torch.float32)
    gpu_runtime.grid_tensor_cache[cache_key] = (points_t, embs_t)
    return points_t, embs_t


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


def finalize_leaf_arrays(root: Grid) -> None:
    for leaf in root.leaves():
        if leaf.vector_arr is None:
            if leaf.vector:
                leaf.vector_arr = np.asarray(leaf.vector, dtype=np.float64)
            else:
                leaf.vector_arr = np.empty((0, len(root.center)), dtype=np.float64)
        if leaf.vec_ids_arr is None:
            leaf.vec_ids_arr = np.asarray(leaf.vec_ids, dtype=np.int64)
        if leaf.embeddings_arr is None:
            if leaf.embeddings:
                leaf.embeddings_arr = np.asarray(leaf.embeddings, dtype=np.float32)
            else:
                leaf.embeddings_arr = np.empty((0, 0), dtype=np.float32)
        if leaf.vec_id_to_local is None:
            leaf.vec_id_to_local = {int(v): i for i, v in enumerate(leaf.vec_ids_arr)}


def ensure_index_runtime_structures(
    index_grid: HierarchicalGrid,
    inverted_index: InvertedIndex,
    column_sets: Sequence[Set[int]],
) -> None:
    finalize_leaf_arrays(index_grid.root)

    if not hasattr(inverted_index, "index"):
        inverted_index.index = {}
    if not hasattr(inverted_index, "local_index"):
        inverted_index.local_index = {}
    if not hasattr(inverted_index, "local_index_arr"):
        inverted_index.local_index_arr = {}

    if inverted_index.local_index:
        inverted_index.finalize()
    if inverted_index.local_index_arr:
        return

    # Backward-compatibility for old index cache files that predate local postings.
    arr_map: Dict[Grid, Dict[int, np.ndarray]] = {}
    for grid, cols in inverted_index.index.items():
        assert grid.vec_ids_arr is not None
        vec_ids_list = [int(v) for v in grid.vec_ids_arr.tolist()]
        out: Dict[int, np.ndarray] = {}
        for col in cols:
            ids = column_sets[col]
            local_idx = [i for i, vid in enumerate(vec_ids_list) if vid in ids]
            if local_idx:
                out[col] = np.asarray(local_idx, dtype=np.int32)
        if out:
            arr_map[grid] = out
    inverted_index.local_index_arr = arr_map


def _grid_col_match_fast(
    grid: Grid,
    col: int,
    query_point: np.ndarray,
    query_embedding: np.ndarray,
    tau: float,
    inverted_index: InvertedIndex,
    query_point_t: Optional[torch.Tensor] = None,
    query_embedding_t: Optional[torch.Tensor] = None,
    gpu_runtime: Optional[QueryGpuRuntime] = None,
) -> bool:
    idx = inverted_index.local_indices(grid, col)
    if idx.size == 0:
        return False

    assert grid.vector_arr is not None
    assert grid.embeddings_arr is not None

    if gpu_runtime is not None and query_point_t is not None and query_embedding_t is not None:
        points_t, embs_t = _grid_cuda_tensors(grid, gpu_runtime)
        idx_t = torch.as_tensor(idx, device=gpu_runtime.device, dtype=torch.int64)
        points = torch.index_select(points_t, 0, idx_t)

        axis_ok = torch.all(torch.abs(points - query_point_t.unsqueeze(0)) <= tau, dim=1)
        if not bool(torch.any(axis_ok).item()):
            return False

        candidates = points[axis_ok]
        if bool(torch.any(torch.any(candidates + query_point_t.unsqueeze(0) <= tau, dim=1)).item()):
            return True

        rem_idx = idx_t[axis_ok]
        embs = torch.index_select(embs_t, 0, rem_idx)
        diff = embs - query_embedding_t.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=1)
        return bool(torch.any(dist2 <= float(tau * tau)).item())

    points = grid.vector_arr[idx]
    # Filter1: if any axis is outside tau window, that point cannot match.
    axis_ok = np.all(np.abs(points - query_point[None, :]) <= tau, axis=1)
    if not np.any(axis_ok):
        return False

    candidates = points[axis_ok]
    # Match1: sufficient condition for guaranteed match in pivot space.
    if np.any(np.any(candidates + query_point[None, :] <= tau, axis=1)):
        return True

    rem_idx = idx[axis_ok]
    embs = grid.embeddings_arr[rem_idx]
    diff = embs - query_embedding[None, :]
    dist2 = np.sum(diff * diff, axis=1, dtype=np.float32)
    return bool(np.any(dist2 <= float(tau * tau)))


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
        device: str,
    ):
        self.mode = mode
        self.dim = dim
        self.seed = seed
        self.embedding_pickle = embedding_pickle
        self.mpnet_model = mpnet_model
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._value_cache: Dict[str, Optional[np.ndarray]] = {}
        self._token_cache: Dict[str, Optional[np.ndarray]] = {}
        self.external_vectors: Optional[Dict[str, np.ndarray]] = None
        self._mpnet = None
        self._fasttext = None
        self._fasttext_torch_device: Optional[torch.device] = None
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
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self._fasttext_torch_device = torch.device(self.device)
                LOGGER.info(
                    "fasttext GPU path enabled on %s; token aggregation runs on CUDA",
                    self._fasttext_torch_device,
                )
            else:
                LOGGER.info("fasttext GPU path disabled; using CPU")

        if self.mode == "mpnet":
            if mpnet_model is None:
                raise ValueError("--embedding_mode mpnet requires --mpnet_model")
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "sentence-transformers is required for --embedding_mode mpnet"
                ) from exc
            self._mpnet = SentenceTransformer(
                mpnet_model,
                cache_folder=str(MODEL_DIR),
                device=self.device,
            )
            model_device = None
            if hasattr(self._mpnet, "device"):
                model_device = str(self._mpnet.device)
            elif hasattr(self._mpnet, "_target_device"):
                model_device = str(self._mpnet._target_device)
            LOGGER.info(
                "MPNet model loaded: model=%s, requested_device=%s, actual_device=%s",
                mpnet_model,
                self.device,
                model_device,
            )

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _unit_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    def _token_vector(self, token: str) -> Optional[np.ndarray]:
        cached = self._token_cache.get(token)
        if cached is not None or token in self._token_cache:
            return cached
        if self.mode in {"glove", "fasttext"}:
            if self.mode == "glove":
                assert self.external_vectors is not None
                vec = self.external_vectors.get(token)
                if vec is None:
                    self._token_cache[token] = None
                    return None
            else:
                if self._fasttext is not None:
                    vec = self._fasttext.get_word_vector(token)
                else:
                    assert self.external_vectors is not None
                    vec = self.external_vectors.get(token)
                    if vec is None:
                        self._token_cache[token] = None
                        return None
            norm = float(np.linalg.norm(vec))
            if norm <= 1e-12:
                self._token_cache[token] = None
                return None
            out = (vec / norm).astype(np.float32)
            self._token_cache[token] = out
            return out
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
        pending_map: Dict[str, List[int]] = {}

        for idx, value in enumerate(values):
            cached = self._value_cache.get(value)
            if cached is not None or value in self._value_cache:
                results[idx] = cached
            else:
                pending_map.setdefault(value, []).append(idx)

        pending = list(pending_map.keys())

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
            for value, emb in zip(pending, embeddings):
                norm = float(np.linalg.norm(emb))
                if norm <= 1e-12:
                    self._value_cache[value] = None
                    for idx in pending_map[value]:
                        results[idx] = None
                else:
                    normalized = emb / norm
                    self._value_cache[value] = normalized
                    for idx in pending_map[value]:
                        results[idx] = normalized
            return results

        if self.mode == "fasttext" and self._fasttext_torch_device is not None:
            self._embed_values_fasttext_torch(pending, pending_map, results)
            return results

        for value in pending:
            vec = self.embed_value(value)
            for idx in pending_map[value]:
                results[idx] = vec
        return results

    def _embed_values_fasttext_torch(
        self,
        pending: List[str],
        pending_map: Dict[str, List[int]],
        results: List[Optional[np.ndarray]],
    ) -> None:
        dev = self._fasttext_torch_device
        if dev is None:
            return

        # Batch token vectors and aggregate on GPU. This keeps the same semantics:
        # mean of L2-normalized token vectors, then L2-normalize value vector.
        batched_rows: List[np.ndarray] = []
        lengths: List[int] = []
        valid_values: List[str] = []
        for value in pending:
            tokens = self._tokenize(value)
            if not tokens:
                self._value_cache[value] = None
                for idx in pending_map[value]:
                    results[idx] = None
                continue

            token_vecs: List[np.ndarray] = []
            for tok in tokens:
                vec = self._token_vector(tok)
                if vec is not None:
                    token_vecs.append(vec)

            if not token_vecs:
                self._value_cache[value] = None
                for idx in pending_map[value]:
                    results[idx] = None
                continue

            arr = np.stack(token_vecs, axis=0).astype(np.float32, copy=False)
            batched_rows.append(arr)
            lengths.append(arr.shape[0])
            valid_values.append(value)

        if not batched_rows:
            return

        concat = np.concatenate(batched_rows, axis=0).astype(np.float32, copy=False)
        token_tensor = torch.from_numpy(concat).to(dev, non_blocking=True)
        len_tensor = torch.as_tensor(lengths, device=dev, dtype=torch.int64)
        val_idx = torch.repeat_interleave(
            torch.arange(len(valid_values), device=dev, dtype=torch.int64),
            len_tensor,
        )

        sums = torch.zeros(
            (len(valid_values), token_tensor.shape[1]),
            device=dev,
            dtype=token_tensor.dtype,
        )
        sums.index_add_(0, val_idx, token_tensor)
        means = sums / len_tensor.to(token_tensor.dtype).unsqueeze(1)
        norms = torch.linalg.norm(means, dim=1, keepdim=True)
        valid = norms.squeeze(1) > 1e-12
        means = means / torch.clamp(norms, min=1e-12)

        out_np = means.detach().cpu().numpy().astype(np.float32, copy=False)
        valid_np = valid.detach().cpu().numpy()
        for row_idx, value in enumerate(valid_values):
            if not bool(valid_np[row_idx]):
                self._value_cache[value] = None
                vec = None
            else:
                vec = out_np[row_idx]
                self._value_cache[value] = vec
            for idx in pending_map[value]:
                results[idx] = vec

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
                    LOGGER.info("loaded %s fasttext vectors", len(vectors))

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
            LOGGER.warning("fastText archive has no .bin; using .vec vectors from %s", out_path)
        return kind, out_path

    def _download_file(self, url: str, dest: Path) -> None:
        LOGGER.info("downloading fastText vectors to %s: %s", MODEL_DIR, url)
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
                        LOGGER.info("download %.0f%%", pct)


def _normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def _sample_unique_values(
    values: Set[str],
    max_unique_values: int,
    seed: int,
    key: str,
) -> Tuple[Set[str], int]:
    cap = int(max(0, max_unique_values))
    original_size = len(values)
    if cap == 0 or original_size <= cap:
        return values, original_size

    value_list = sorted(values)
    key_salt = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
    local_seed = (int(seed) + key_salt) % (2 ** 32)
    rng = np.random.default_rng(local_seed)
    selected_idx = rng.choice(original_size, size=cap, replace=False)
    sampled = {value_list[int(i)] for i in selected_idx.tolist()}
    return sampled, original_size


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
    *,
    max_unique_values: int = 0,
    sample_seed: int = 0,
    sample_scope: str = "",
    log_sampling: bool = False,
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

    if int(max_unique_values) > 0:
        for col in selected_columns:
            scope = sample_scope or "column"
            sampled, original_size = _sample_unique_values(
                values[col],
                max_unique_values=max_unique_values,
                seed=sample_seed,
                key=f"{scope}|{csv_path}|{col}",
            )
            if original_size > len(sampled):
                log_fn = LOGGER.info if log_sampling else LOGGER.debug
                log_fn(
                    "sampled unique values for %s.%s [%s]: %s -> %s",
                    csv_path.name,
                    col,
                    scope,
                    original_size,
                    len(sampled),
                )
            values[col] = sampled
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
            # AutoFJ-style groundtruth files contain left/right table pairs without
            # explicit column names. Using right_table as query is typically better
            # aligned with containment-style retrieval (query is often the smaller set).
            table_key = "right_table" if "right_table" in header_lower else "left_table"
            table_idx = header_lower.index(table_key)
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
                LOGGER.warning("query table not found: %s", table_ref)
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


def _map_points_to_pivots(
    points: np.ndarray,
    pivots: np.ndarray,
    gpu_device: Optional[torch.device] = None,
) -> np.ndarray:
    if points.size == 0 or pivots.size == 0:
        return np.empty((points.shape[0], pivots.shape[0]), dtype=np.float64)
    if gpu_device is not None:
        points_t = torch.as_tensor(points, device=gpu_device, dtype=torch.float32)
        pivots_t = torch.as_tensor(pivots, device=gpu_device, dtype=torch.float32)
        distances = torch.cdist(points_t, pivots_t, p=2)
        return distances.detach().cpu().numpy().astype(np.float64, copy=False)

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
    max_unique_values_per_attr: int,
) -> Dict:
    LOGGER.info("Building PEXESO index from %s", datalake_dir)
    column_items: List[Tuple[str, str]] = []
    column_sets: List[Set[int]] = []
    column_raw_values: List[Set[str]] = []
    value_to_id: Dict[str, int] = {}
    value_embeddings: List[np.ndarray] = []

    csv_files = sorted(datalake_dir.glob("*.csv"))
    LOGGER.info("datalake csv files: %s", len(csv_files))
    for idx, csv_path in enumerate(csv_files):
        selected_cols = _discover_selected_columns(csv_path, column_name)
        if not selected_cols:
            continue
        values_by_col = _read_columns(
            csv_path,
            selected_cols,
            max_unique_values=max_unique_values_per_attr,
            sample_seed=seed,
            sample_scope="index",
            log_sampling=False,
        )
        for col in selected_cols:
            ids: Set[int] = set()
            raw_values = set(values_by_col.get(col, set()))
            values = list(raw_values)
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
                column_raw_values.append(raw_values)
        if (idx + 1) % 500 == 0 or idx + 1 == len(csv_files):
            LOGGER.info("indexed %s/%s tables", idx + 1, len(csv_files))

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
    finalize_leaf_arrays(index_grid.root)
    inverted_index = InvertedIndex()
    for col_id, ids in enumerate(column_sets):
        for value_id in ids:
            grid = id_to_grid[value_id]
            assert grid.vec_id_to_local is not None
            local_idx = grid.vec_id_to_local[value_id]
            inverted_index.add(grid, col_id, local_idx=local_idx)
    inverted_index.finalize()

    LOGGER.info(
        "index ready: %s columns, %s unique values, %s pivots",
        len(column_items),
        embedding_matrix.shape[0],
        pivots.shape[0],
    )

    return {
        "meta": {
            "embedding_mode": embedder.mode,
            "embedding_dim": embedder.dim,
            "embedding_pickle": str(embedder.embedding_pickle) if embedder.embedding_pickle else None,
            "fasttext_url": FASTTEXT_DEFAULT_URL if embedder.mode == "fasttext" else None,
            "fasttext_cache_dir": str(MODEL_DIR),
            "mpnet_model": str(embedder.mpnet_model) if embedder.mpnet_model else None,
            "device": embedder.device if embedder.mode == "mpnet" else None,
            "num_pivots": num_pivots,
            "num_layers": num_layers,
            "pivot_method": pivot_method,
            "column_name": column_name,
            "seed": seed,
            "max_index_unique_values": int(max_unique_values_per_attr),
        },
        "column_items": column_items,
        "column_sets": column_sets,
        "column_raw_values": column_raw_values,
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


def _load_query_values(
    csv_path: Path,
    query_column: str,
    max_unique_values_per_attr: int,
    seed: int,
) -> Tuple[Optional[str], List[str]]:
    selected_cols = _discover_selected_columns(csv_path, query_column)
    if not selected_cols:
        return None, []
    canonical = selected_cols[0]
    values_by_col = _read_columns(
        csv_path,
        [canonical],
        max_unique_values=max_unique_values_per_attr,
        sample_seed=seed,
        sample_scope="query",
        log_sampling=True,
    )
    values = sorted(values_by_col.get(canonical, set()))
    return canonical, values


def _containment_score(query_values: Set[str], candidate_values: Set[str]) -> float:
    if not query_values:
        return 0.0
    inter = len(query_values.intersection(candidate_values))
    return float(inter) / float(len(query_values))


def run_baseline(args: argparse.Namespace) -> int:
    log_cuda_runtime()
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
    LOGGER.info("dataset_dir=%s", dataset_dir)
    LOGGER.info("datalake_dir=%s", datalake_dir)

    global_max_unique_values = max(0, int(args.max_unique_values))
    max_index_unique_values = (
        int(args.max_index_unique_values)
        if int(args.max_index_unique_values) > 0
        else global_max_unique_values
    )
    max_query_unique_values = (
        int(args.max_query_unique_values)
        if int(args.max_query_unique_values) > 0
        else global_max_unique_values
    )

    query_source = Path(args.query_source) if args.query_source else None
    query_specs, query_source_desc = _discover_query_specs(
        dataset_dir=dataset_dir,
        datalake_dir=datalake_dir,
        query_source=query_source,
        default_query_column=args.default_query_column,
    )
    if not query_specs:
        raise ValueError("No query columns discovered")
    LOGGER.info("Query source: %s (%s query columns)", query_source_desc, len(query_specs))
    LOGGER.info(
        "run config: embedding_mode=%s, column_name=%s, tau=%s, "
        "containment_threshold=%s, num_pivots=%s, num_layers=%s, "
        "time_threshold=%s, top_k=%s, score_mode=%s, prune_multiplier=%s, "
        "max_index_unique_values=%s, max_query_unique_values=%s",
        args.embedding_mode,
        args.column_name,
        args.tau,
        args.containment_threshold,
        args.num_pivots,
        args.num_layers,
        args.time_threshold,
        args.top_k,
        args.score_mode,
        args.prune_multiplier,
        max_index_unique_values,
        max_query_unique_values,
    )

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
        device=args.device,
    )
    LOGGER.info("Device selected: %s", embedder.device)
    if embedder.mode == "glove":
        LOGGER.info("embedding_mode=glove runs on CPU")
    elif embedder.mode == "fasttext":
        if embedder._fasttext_torch_device is not None:
            LOGGER.info(
                "embedding_mode=fasttext uses CUDA for token aggregation on %s",
                embedder._fasttext_torch_device,
            )
        else:
            LOGGER.info("embedding_mode=fasttext runs on CPU")

    query_gpu_runtime: Optional[QueryGpuRuntime] = None
    query_map_device: Optional[torch.device] = None
    if embedder.device.startswith("cuda") and torch.cuda.is_available():
        query_map_device = torch.device(embedder.device)
        query_gpu_runtime = QueryGpuRuntime(
            device=query_map_device,
            grid_tensor_cache={},
        )
        LOGGER.info("query-time GPU path enabled on %s", query_map_device)
    else:
        LOGGER.info("query-time GPU path disabled; using CPU for query verify/map")

    payload: Optional[Dict] = None
    offline_generation_seconds = 0.0
    used_cached_index = False
    if args.index_cache and Path(args.index_cache).is_file() and not args.rebuild_index:
        cache_path = Path(args.index_cache)
        LOGGER.info("Loading index cache: %s", cache_path)
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
            "device": embedder.device if embedder.mode == "mpnet" else None,
            "num_pivots": args.num_pivots,
            "num_layers": args.num_layers,
            "pivot_method": args.pivot_method,
            "column_name": args.column_name,
            "seed": args.seed,
            "max_index_unique_values": int(max_index_unique_values),
        }
        if not isinstance(meta, dict):
            LOGGER.warning("index cache missing metadata; rebuilding to avoid mismatched embeddings")
            payload = None
        else:
            for key, value in expected_meta.items():
                if meta.get(key) != value:
                    LOGGER.warning("index cache mismatch for %s; rebuilding", key)
                    payload = None
                    break
        if payload is not None:
            used_cached_index = True
    if payload is None:
        offline_start = time.perf_counter()
        payload = _build_index_payload(
            datalake_dir=datalake_dir,
            column_name=args.column_name,
            embedder=embedder,
            num_pivots=args.num_pivots,
            num_layers=args.num_layers,
            seed=args.seed,
            pivot_method=args.pivot_method,
            max_unique_values_per_attr=max_index_unique_values,
        )
        offline_generation_seconds = time.perf_counter() - offline_start
        if args.index_cache:
            cache_path = Path(args.index_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(payload, f)
            LOGGER.info("Saved index cache: %s", cache_path)
    else:
        LOGGER.info("Offline datalake embedding generation skipped (loaded index cache).")

    column_items: List[Tuple[str, str]] = payload["column_items"]
    column_sets: List[Set[int]] = payload["column_sets"]
    column_raw_values: Optional[List[Set[str]]] = payload.get("column_raw_values")
    value_to_id: Dict[str, int] = payload["value_to_id"]
    pivots: np.ndarray = payload["pivots"]
    x_min: float = payload["x_min"]
    x_max: float = payload["x_max"]
    num_layers: int = payload["num_layers"]
    index_grid: HierarchicalGrid = payload["index_grid"]
    inverted_index: InvertedIndex = payload["inverted_index"]
    ensure_index_runtime_structures(index_grid, inverted_index, column_sets)
    if column_raw_values is None:
        id_to_value = {vid: val for val, vid in value_to_id.items()}
        column_raw_values = []
        for ids in column_sets:
            column_raw_values.append({id_to_value[i] for i in ids if i in id_to_value})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped = 0
    total_embed_sec = 0.0
    total_block_sec = 0.0
    total_verify_sec = 0.0
    total_score_sec = 0.0
    online_start = time.perf_counter()
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
            canonical_col, query_values = _load_query_values(
                spec.csv_path,
                spec.query_column,
                max_unique_values_per_attr=max_query_unique_values,
                seed=args.seed,
            )
            if not query_values or canonical_col is None:
                skipped += 1
                continue
            query_value_set = set(query_values)

            query_embeddings: List[np.ndarray] = []
            query_known_ids: Set[int] = set()
            t_embed_start = time.perf_counter()
            query_vecs = embedder.embed_values(query_values)
            for value, vec in zip(query_values, query_vecs):
                if vec is None:
                    continue
                query_embeddings.append(vec)
                value_id = value_to_id.get(value)
                if value_id is not None:
                    query_known_ids.add(value_id)
            t_embed = time.perf_counter() - t_embed_start
            total_embed_sec += t_embed

            if not query_embeddings:
                skipped += 1
                continue

            query_matrix = np.asarray(query_embeddings, dtype=np.float32)
            query_points = _map_points_to_pivots(
                query_matrix,
                pivots,
                gpu_device=query_map_device,
            )
            query_grid, _, _ = build_hierarchical_grid(
                query_points,
                query_matrix,
                n_layers=num_layers,
                x_min=x_min,
                x_max=x_max,
            )

            pairs: Dict[int, Pair] = {}
            t_block_start = time.perf_counter()
            block(query_grid.root, index_grid.root, pairs, args.tau)
            t_block = time.perf_counter() - t_block_start
            total_block_sec += t_block
            if args.debug:
                matched_grid_count = sum(len(p.matched_grids) for p in pairs.values())
                candidate_grid_count = sum(len(p.candidate_grids) for p in pairs.values())
                LOGGER.info(
                    "%s.%s: query_values=%s, embedded=%s, known_ids=%s, pairs=%s, "
                    "matched_grids=%s, candidate_grids=%s, t_embed=%.3fs, t_block=%.3fs",
                    spec.query_table,
                    canonical_col,
                    len(query_values),
                    len(query_embeddings),
                    len(query_known_ids),
                    len(pairs),
                    matched_grid_count,
                    candidate_grid_count,
                    t_embed,
                    t_block,
                )

            # Paper-faithful threshold: T * |query raw distinct values|.
            threshold_count = args.containment_threshold * len(query_values)
            t_verify_start = time.perf_counter()
            candidate_indices, match_count, over_time = verify(
                pairs=pairs,
                inverted_index=inverted_index,
                tau=args.tau,
                threshold_count=threshold_count,
                index_sets=column_sets,
                # Paper code's filter7 uses qlen=len(query_embs).
                query_size=len(query_embeddings),
                time_threshold=args.time_threshold,
                gpu_runtime=query_gpu_runtime,
                prune_multiplier=args.prune_multiplier,
            )
            t_verify = time.perf_counter() - t_verify_start
            total_verify_sec += t_verify
            if args.debug:
                top_match = max(match_count) if match_count else 0
                LOGGER.info(
                    "%s.%s: threshold_count=%.3f, candidate_indices=%s, max_match_count=%s, "
                    "t_verify=%.3fs",
                    spec.query_table,
                    canonical_col,
                    threshold_count,
                    len(candidate_indices),
                    top_match,
                    t_verify,
                )
            if over_time:
                LOGGER.warning(
                    "verify timed out for %s.%s (>%ss); partial candidates kept",
                    spec.query_table,
                    canonical_col,
                    args.time_threshold,
                )

            scored: List[Tuple[str, str, float]] = []
            t_score_start = time.perf_counter()
            for col_idx in candidate_indices:
                cand_table, cand_col = column_items[col_idx]
                if not args.include_same_table and cand_table == spec.query_table:
                    continue
                if column_raw_values is not None and col_idx < len(column_raw_values):
                    containment = _containment_score(query_value_set, column_raw_values[col_idx])
                else:
                    # Fallback for old caches without raw value sets.
                    cand_ids = {str(v) for v in column_sets[col_idx]}
                    query_ids = {str(v) for v in query_known_ids}
                    containment = _containment_score(query_ids, cand_ids)
                if args.score_mode == "hybrid":
                    pexeso_score = float(match_count[col_idx]) / float(max(1, len(query_embeddings)))
                    score = max(containment, pexeso_score)
                else:
                    score = containment
                scored.append((cand_table, cand_col, score))
            t_score = time.perf_counter() - t_score_start
            total_score_sec += t_score

            scored.sort(key=lambda x: x[2], reverse=True)
            if args.top_k > 0:
                scored = scored[: args.top_k]
            if args.debug:
                top_score = scored[0][2] if scored else 0.0
                LOGGER.info(
                    "%s.%s: scored_after_filters=%s, top_score=%.4f, t_score=%.3fs",
                    spec.query_table,
                    canonical_col,
                    len(scored),
                    top_score,
                    t_score,
                )
            for cand_table, cand_col, score in scored:
                writer.writerow(
                    [spec.query_table, canonical_col, cand_table, cand_col, f"{score:.6f}"]
                )
                rows_written += 1

            if (idx + 1) % 20 == 0 or idx + 1 == len(query_specs):
                LOGGER.info("processed %s/%s queries", idx + 1, len(query_specs))

    online_elapsed = time.perf_counter() - online_start
    LOGGER.info(
        "Done. Wrote %s rows to %s (skipped %s queries, elapsed %.1fs)",
        rows_written,
        out_csv,
        skipped,
        online_elapsed,
    )
    if used_cached_index:
        LOGGER.info(
            "[TIMING] offline_datalake_embedding_seconds=0.000 (loaded cached index)"
        )
    else:
        LOGGER.info(
            "[TIMING] offline_datalake_embedding_seconds=%.3f",
            offline_generation_seconds,
        )
    LOGGER.info(
        "[TIMING] online_query_seconds=%.3f",
        online_elapsed,
    )
    LOGGER.info(
        "timing summary: embed=%.1fs, block=%.1fs, verify=%.1fs, score=%.1fs",
        total_embed_sec,
        total_block_sec,
        total_verify_sec,
        total_score_sec,
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
        default=30,
        help="Per-query verification timeout (seconds).",
    )
    parser.add_argument(
        "--score_mode",
        choices=["containment", "hybrid"],
        default="containment",
        help="Final ranking score mode (paper-faithful: containment).",
    )
    parser.add_argument(
        "--prune_multiplier",
        type=float,
        default=2.0,
        help="Lemma-7 pruning multiplier (paper-faithful: 2.0).",
    )
    parser.add_argument(
        "--max_unique_values",
        type=int,
        default=0,
        help="Global cap for unique values per attribute (0 disables sampling).",
    )
    parser.add_argument(
        "--max_index_unique_values",
        type=int,
        default=0,
        help="Index-time cap for unique values per attribute (overrides --max_unique_values when >0).",
    )
    parser.add_argument(
        "--max_query_unique_values",
        type=int,
        default=0,
        help="Query-time cap for unique values per attribute (overrides --max_unique_values when >0).",
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device for mpnet mode (auto, cpu, cuda).",
    )
    parser.add_argument("--seed", type=int, default=128, help="Random seed.")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed per-query diagnostics for block/verify/scoring.",
    )
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
    setup_logging(args.log_level)
    try:
        return run_baseline(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
