#!/usr/bin/env python3
"""Snoopy pretrained-inference baseline integrated for SemSketch datasets."""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fasttext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
TARGET_TABLE_RE = re.compile(r"^target_(\d+)\.csv$", re.IGNORECASE)
QUERY_TABLE_RE = re.compile(r"^query_(\d+)\.csv$", re.IGNORECASE)


@dataclass
class QuerySpec:
    query_table: str
    query_column: str
    csv_path: Path


@dataclass
class PrecomputedStats:
    target_reused: int = 0
    query_reused: int = 0
    fallback: int = 0
    target_miss: int = 0
    query_miss: int = 0


class Scorpion(nn.Module):
    """Inference-only copy of Snoopy's Scorpion encoder."""

    def __init__(self, n_proxy_sets: int, n_elements: int, d: int, device: torch.device):
        super().__init__()
        self.n_proxy_sets = n_proxy_sets
        self.n_elements = n_elements
        self.d = d
        self.device = device
        self.proxy_sets = nn.Parameter(torch.empty(self.n_proxy_sets, self.n_elements, self.d))

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).to(self.device)
        sim_mat = torch.matmul(self.proxy_sets, x)
        t, _ = torch.max(sim_mat, dim=1)
        splits = torch.split(t, index.tolist(), dim=1)
        pooled = [torch.sum(split, dim=1) for split in splits]
        out = torch.stack(pooled, dim=0).to(torch.float32)
        out = F.normalize(out, p=2, dim=1)
        return out


def setup_logging(level_name: str) -> None:
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            LOGGER.warning("MPS requested but unavailable; falling back to CPU.")
            return "cpu"
    return requested


def _has_csv_files(path: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() == ".csv" for p in path.iterdir())
    except OSError:
        return False


def resolve_datalake_dir(path: Path) -> Path:
    nested = path / "datalake"
    if nested.is_dir() and _has_csv_files(nested):
        return nested
    if path.is_dir() and _has_csv_files(path):
        return path
    return nested if nested.is_dir() else path


def read_csv_header(csv_path: Path) -> List[str]:
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except OSError:
        return []
    return header or []


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

    col_norm = (column_name or "title").strip().lower()
    if col_norm in {"*", "all"}:
        out: List[str] = []
        for name in header:
            if name.strip().lower() in {"id", "index"}:
                continue
            out.append(name)
        return out

    matched = canonical_column_name(header, column_name)
    return [matched] if matched else []


def read_selected_column_values(csv_path: Path, selected_columns: Sequence[str]) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {c: [] for c in selected_columns}
    if not selected_columns:
        return values

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return values

        idx_map: Dict[str, int] = {}
        for col in selected_columns:
            for i, name in enumerate(header):
                if name == col:
                    idx_map[col] = i
                    break

        for row in reader:
            for col, idx in idx_map.items():
                if idx < len(row):
                    values[col].append(row[idx])
                else:
                    values[col].append("")

    return values


def _resolve_table_path(table_ref: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    ref = (table_ref or "").strip()
    if not ref:
        return None

    p = Path(ref)
    if p.is_file():
        return p

    candidates: List[Path] = []
    if p.suffix.lower() == ".csv":
        candidates.append(p)
    else:
        candidates.append(p.with_suffix(".csv"))

    for base in search_dirs:
        for cand in candidates:
            joined = base / cand
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
        for t_key, c_key in key_pairs:
            if t_key in header_lower and c_key in header_lower:
                indices = (header_lower.index(t_key), header_lower.index(c_key))
                break

        if indices is None and "left_table" in header_lower:
            table_key = "right_table" if "right_table" in header_lower else "left_table"
            t_idx = header_lower.index(table_key)
            seen = set()
            for row in reader:
                if len(row) <= t_idx:
                    continue
                table = row[t_idx].strip()
                if table and table not in seen:
                    pairs.append((table, default_query_column))
                    seen.add(table)
            return pairs

        if indices is None:
            for row in reader:
                if len(row) >= 2:
                    table = row[0].strip()
                    col = row[1].strip() or default_query_column
                    if table:
                        pairs.append((table, col))
            return pairs

        t_idx, c_idx = indices
        for row in reader:
            if len(row) <= max(t_idx, c_idx):
                continue
            table = row[t_idx].strip()
            col = row[c_idx].strip() or default_query_column
            if table:
                pairs.append((table, col))
    return pairs


def discover_query_specs(
    dataset_dir: Path,
    datalake_dir: Path,
    query_source: Optional[Path],
    default_query_column: str,
) -> Tuple[List[QuerySpec], str]:
    queries_dir = dataset_dir / "queries"
    search_dirs: List[Path] = []
    if queries_dir.is_dir():
        search_dirs.append(queries_dir)
    search_dirs.append(datalake_dir)
    search_dirs.append(dataset_dir)

    source = query_source
    if source is None:
        precedence = [
            dataset_dir / "query_columns.csv",
            dataset_dir / "autofj_query_columns.csv",
            dataset_dir / "gdc_breakdown_query_columns.csv",
            dataset_dir / "groundtruth-joinable.csv",
        ]
        source = None
        for cand in precedence:
            if cand.is_file():
                source = cand
                break
        if source is None and queries_dir.is_dir() and _has_csv_files(queries_dir):
            source = queries_dir
        if source is None:
            source = datalake_dir

    if source.is_file():
        specs: List[QuerySpec] = []
        seen = set()
        for table_ref, query_col in _parse_query_pairs(source, default_query_column):
            table_path = _resolve_table_path(table_ref, search_dirs=search_dirs)
            if table_path is None:
                LOGGER.warning("Query table not found: %s", table_ref)
                continue
            key = (table_path.name, query_col)
            if key in seen:
                continue
            seen.add(key)
            specs.append(QuerySpec(query_table=table_path.name, query_column=query_col, csv_path=table_path))
        return specs, f"file:{source}"

    if source.is_dir():
        specs = []
        for csv_path in sorted(source.glob("*.csv")):
            selected = discover_selected_columns(csv_path, default_query_column)
            for col in selected:
                specs.append(QuerySpec(query_table=csv_path.name, query_column=col, csv_path=csv_path))
        return specs, f"dir:{source}"

    raise ValueError(f"Query source does not exist: {source}")


def file_signature(path: Optional[Path]) -> Dict[str, Optional[object]]:
    if path is None:
        return {"path": None, "mtime_ns": None, "size": None}
    p = path.resolve()
    if not p.is_file():
        return {"path": str(p), "mtime_ns": None, "size": None}
    stat = p.stat()
    return {"path": str(p), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def compare_meta(expected: Dict[str, object], got: Dict[str, object]) -> List[str]:
    mismatches: List[str] = []
    for key, value in expected.items():
        if got.get(key) != value:
            mismatches.append(key)
    return mismatches


class PrecomputedLookup:
    def __init__(self, enabled: bool, target_npy: Optional[Path], query_npy: Optional[Path]):
        self.enabled = bool(enabled)
        self.target_npy = target_npy
        self.query_npy = query_npy
        self._target = None
        self._query = None

        if not self.enabled:
            return

        self._target = self._safe_load_npy(target_npy, "target")
        self._query = self._safe_load_npy(query_npy, "query")

    @staticmethod
    def _safe_load_npy(path: Optional[Path], label: str):
        if path is None:
            return None
        if not path.is_file():
            LOGGER.warning("Precomputed %s npy not found: %s", label, path)
            return None
        try:
            return np.load(path, allow_pickle=True)
        except Exception as exc:
            LOGGER.warning("Failed to load precomputed %s npy (%s): %s", label, path, exc)
            return None

    @staticmethod
    def _array_row_to_matrix(
        arr_obj,
        expected_dim: Optional[int],
    ) -> Optional[np.ndarray]:
        try:
            mat = np.asarray(arr_obj, dtype=np.float32)
        except Exception:
            return None
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
            return None
        if expected_dim is not None and mat.shape[1] != expected_dim:
            return None
        return mat

    def maybe_get(
        self,
        table_name: str,
        column_name: str,
        stats: PrecomputedStats,
        expected_dim: Optional[int],
    ) -> Optional[np.ndarray]:
        if not self.enabled or column_name.strip().lower() != "title":
            return None

        t_match = TARGET_TABLE_RE.match(table_name)
        if t_match:
            idx = int(t_match.group(1))
            if self._target is None or idx < 0 or idx >= len(self._target):
                stats.target_miss += 1
                stats.fallback += 1
                return None
            mat = self._array_row_to_matrix(self._target[idx], expected_dim=expected_dim)
            if mat is None:
                stats.target_miss += 1
                stats.fallback += 1
                return None
            stats.target_reused += 1
            return mat

        q_match = QUERY_TABLE_RE.match(table_name)
        if q_match:
            idx = int(q_match.group(1))
            if self._query is None or idx < 0 or idx >= len(self._query):
                stats.query_miss += 1
                stats.fallback += 1
                return None
            mat = self._array_row_to_matrix(self._query[idx], expected_dim=expected_dim)
            if mat is None:
                stats.query_miss += 1
                stats.fallback += 1
                return None
            stats.query_reused += 1
            return mat

        return None


class FastTextEmbedder:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"fastText model not found: {model_path}")
        LOGGER.info("Loading fastText model: %s", model_path)
        self.model = fasttext.load_model(str(model_path))
        self.dim = int(self.model.get_dimension())

    def embed_values(self, values: Sequence[str]) -> np.ndarray:
        if not values:
            return np.zeros((1, self.dim), dtype=np.float32)

        out = np.empty((len(values), self.dim), dtype=np.float32)
        for i, val in enumerate(values):
            text = "" if val is None else str(val)
            out[i] = self.model.get_sentence_vector(text.replace("\n", " "))
        return out


def _extract_proxy_tensor(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "proxy_sets" in state_dict:
        return state_dict["proxy_sets"]
    for key, value in state_dict.items():
        if key.endswith("proxy_sets"):
            return value
    raise KeyError("Checkpoint does not contain proxy_sets")


def load_scorpion_model(checkpoint_path: Path, device: torch.device) -> Tuple[Scorpion, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    proxy_tensor = _extract_proxy_tensor(state_dict)
    if proxy_tensor.ndim != 3:
        raise ValueError(f"proxy_sets must be 3D, got shape={tuple(proxy_tensor.shape)}")

    n_proxy_sets, n_elements, d = map(int, proxy_tensor.shape)
    model = Scorpion(n_proxy_sets=n_proxy_sets, n_elements=n_elements, d=d, device=device).to(device)

    clean_state = {"proxy_sets": proxy_tensor}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        raise RuntimeError("Checkpoint missing required model parameters: " + ", ".join(missing))
    if unexpected:
        LOGGER.warning("Ignoring unexpected checkpoint parameters: %s", ", ".join(unexpected))

    model.eval()
    return model, d


def ensure_matrix(matrix: np.ndarray, expected_dim: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
    if arr.shape[0] == 0:
        arr = np.zeros((1, expected_dim), dtype=np.float32)
    if arr.shape[1] != expected_dim:
        raise ValueError(f"Matrix dim mismatch: expected {expected_dim}, got {arr.shape[1]}")
    return arr


def encode_batch(model: Scorpion, matrices: Sequence[np.ndarray], device: torch.device) -> np.ndarray:
    concat = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    index = torch.tensor([m.shape[0] for m in matrices], dtype=torch.long, device=device)
    data = torch.from_numpy(concat).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        emb = model(data, index)
    return emb.detach().cpu().numpy().astype(np.float32, copy=False)


def encode_matrices(
    model: Scorpion,
    matrices: Sequence[np.ndarray],
    device: torch.device,
    batch_cols: int,
    max_cells_per_batch: int,
) -> np.ndarray:
    if not matrices:
        raise ValueError("No matrices to encode")

    chunks: List[np.ndarray] = []
    pending: List[np.ndarray] = []
    pending_cells = 0

    for mat in matrices:
        pending.append(mat)
        pending_cells += mat.shape[0]
        if len(pending) >= batch_cols or pending_cells >= max_cells_per_batch:
            chunks.append(encode_batch(model, pending, device=device))
            pending = []
            pending_cells = 0

    if pending:
        chunks.append(encode_batch(model, pending, device=device))

    return np.concatenate(chunks, axis=0)


def build_index(
    datalake_dir: Path,
    column_name: str,
    model: Scorpion,
    expected_dim: int,
    embedder: FastTextEmbedder,
    precomputed: PrecomputedLookup,
    device: torch.device,
    batch_cols: int,
    max_cells_per_batch: int,
) -> Tuple[List[Tuple[str, str]], np.ndarray, PrecomputedStats]:
    files = sorted(datalake_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in datalake: {datalake_dir}")

    items: List[Tuple[str, str]] = []
    encoded_chunks: List[np.ndarray] = []
    pending_mats: List[np.ndarray] = []
    pending_items: List[Tuple[str, str]] = []
    pending_cells = 0
    stats = PrecomputedStats()

    for i, csv_path in enumerate(files):
        selected = discover_selected_columns(csv_path, column_name)
        if not selected and column_name.strip().lower() not in {"*", "all"}:
            selected = discover_selected_columns(csv_path, "*")
        if not selected:
            continue

        precomputed_mats: Dict[str, np.ndarray] = {}
        fallback_cols: List[str] = []
        for col in selected:
            mat = precomputed.maybe_get(
                table_name=csv_path.name,
                column_name=col,
                stats=stats,
                expected_dim=expected_dim,
            )
            if mat is None:
                fallback_cols.append(col)
            else:
                precomputed_mats[col] = mat

        fallback_values: Dict[str, List[str]] = {}
        if fallback_cols:
            fallback_values = read_selected_column_values(csv_path, fallback_cols)

        for col in selected:
            if col in precomputed_mats:
                mat = precomputed_mats[col]
            else:
                values = fallback_values.get(col, [])
                mat = embedder.embed_values(values)
            mat = ensure_matrix(mat, expected_dim=expected_dim)

            pending_mats.append(mat)
            pending_items.append((csv_path.name, col))
            pending_cells += mat.shape[0]

            if len(pending_mats) >= batch_cols or pending_cells >= max_cells_per_batch:
                encoded_chunks.append(encode_batch(model, pending_mats, device=device))
                items.extend(pending_items)
                pending_mats = []
                pending_items = []
                pending_cells = 0

        if (i + 1) % 2000 == 0 or (i + 1) == len(files):
            LOGGER.info("indexed %s/%s tables", i + 1, len(files))

    if pending_mats:
        encoded_chunks.append(encode_batch(model, pending_mats, device=device))
        items.extend(pending_items)

    if not items:
        raise ValueError(f"No datalake columns were indexed from {datalake_dir}")

    embeddings = np.concatenate(encoded_chunks, axis=0)
    LOGGER.info("index ready: %s columns", len(items))
    LOGGER.info(
        "precomputed reuse (index): target_reused=%s, query_reused=%s, fallback=%s",
        stats.target_reused,
        stats.query_reused,
        stats.fallback,
    )
    return items, embeddings, stats


def encode_queries(
    query_specs: Sequence[QuerySpec],
    model: Scorpion,
    expected_dim: int,
    embedder: FastTextEmbedder,
    precomputed: PrecomputedLookup,
    device: torch.device,
    batch_cols: int,
    max_cells_per_batch: int,
) -> Tuple[List[Tuple[str, str]], np.ndarray, int, PrecomputedStats]:
    query_items: List[Tuple[str, str]] = []
    matrices: List[np.ndarray] = []
    skipped = 0
    stats = PrecomputedStats()

    for spec in query_specs:
        selected = discover_selected_columns(spec.csv_path, spec.query_column)
        if not selected:
            LOGGER.warning(
                "Query column '%s' not found in %s; skipping.",
                spec.query_column,
                spec.csv_path,
            )
            skipped += 1
            continue

        canonical_col = selected[0]
        mat = precomputed.maybe_get(
            table_name=spec.query_table,
            column_name=canonical_col,
            stats=stats,
            expected_dim=expected_dim,
        )
        if mat is None:
            values = read_selected_column_values(spec.csv_path, [canonical_col]).get(canonical_col, [])
            mat = embedder.embed_values(values)
        mat = ensure_matrix(mat, expected_dim=expected_dim)

        matrices.append(mat)
        query_items.append((spec.query_table, canonical_col))

    if not matrices:
        raise ValueError("No valid query columns found after parsing query source")

    query_embeddings = encode_matrices(
        model=model,
        matrices=matrices,
        device=device,
        batch_cols=batch_cols,
        max_cells_per_batch=max_cells_per_batch,
    )
    LOGGER.info(
        "precomputed reuse (query): target_reused=%s, query_reused=%s, fallback=%s",
        stats.target_reused,
        stats.query_reused,
        stats.fallback,
    )
    return query_items, query_embeddings, skipped, stats


def run_baseline(args: argparse.Namespace) -> int:
    requested_device = resolve_device(args.device)
    device = torch.device(requested_device)

    if args.datalake_dir:
        datalake_dir = resolve_datalake_dir(Path(args.datalake_dir).expanduser().resolve())
        dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else datalake_dir.parent
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        datalake_dir = resolve_datalake_dir(dataset_dir)
    else:
        raise ValueError("Provide either --dataset_dir or --datalake_dir")

    if not datalake_dir.is_dir():
        raise ValueError(f"Datalake directory not found: {datalake_dir}")

    query_source = Path(args.query_source).expanduser().resolve() if args.query_source else None

    fasttext_model_path = Path(args.fasttext_model_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    precomputed_target_npy = (
        Path(args.precomputed_target_npy).expanduser().resolve() if args.precomputed_target_npy else None
    )
    precomputed_query_npy = (
        Path(args.precomputed_query_npy).expanduser().resolve() if args.precomputed_query_npy else None
    )

    LOGGER.info("dataset_dir=%s", dataset_dir)
    LOGGER.info("datalake_dir=%s", datalake_dir)
    LOGGER.info("Device selected: %s", requested_device)

    model, model_dim = load_scorpion_model(checkpoint_path=checkpoint_path, device=device)
    LOGGER.info("Loaded checkpoint: %s (embedding dim=%s)", checkpoint_path, model_dim)

    embedder = FastTextEmbedder(model_path=fasttext_model_path)
    if embedder.dim != model_dim:
        raise ValueError(
            "fastText dimension does not match checkpoint model dimension: "
            f"fastText={embedder.dim}, checkpoint={model_dim}"
        )

    precomputed = PrecomputedLookup(
        enabled=args.reuse_precomputed,
        target_npy=precomputed_target_npy,
        query_npy=precomputed_query_npy,
    )

    cache_meta = {
        "version": 1,
        "datalake_dir": str(datalake_dir),
        "column_name": args.column_name,
        "checkpoint": file_signature(checkpoint_path),
        "reuse_precomputed": bool(args.reuse_precomputed),
        "precomputed_target": file_signature(precomputed_target_npy),
        "precomputed_query": file_signature(precomputed_query_npy),
        "fasttext_model": file_signature(fasttext_model_path),
        "batch_cols": int(args.batch_cols),
        "max_cells_per_batch": int(args.max_cells_per_batch),
    }

    items: Optional[List[Tuple[str, str]]] = None
    embeddings: Optional[np.ndarray] = None
    offline_seconds = 0.0
    used_cached_index = False

    cache_path = Path(args.index_cache).expanduser().resolve() if args.index_cache else None
    if cache_path and cache_path.is_file() and not args.rebuild_index:
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            got_meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            mismatches = compare_meta(cache_meta, got_meta)
            if mismatches:
                LOGGER.info("Index cache metadata mismatch (%s); rebuilding.", ", ".join(mismatches))
            else:
                loaded_items = payload.get("items")
                loaded_embeddings = payload.get("embeddings")
                if (
                    isinstance(loaded_items, list)
                    and isinstance(loaded_embeddings, np.ndarray)
                    and loaded_embeddings.ndim == 2
                    and loaded_embeddings.shape[0] == len(loaded_items)
                ):
                    items = loaded_items
                    embeddings = loaded_embeddings.astype(np.float32, copy=False)
                    used_cached_index = True
                    LOGGER.info("Loaded index cache: %s", cache_path)
                else:
                    LOGGER.info("Index cache payload invalid; rebuilding.")
        except Exception as exc:
            LOGGER.info("Failed to read index cache (%s); rebuilding.", exc)

    if items is None or embeddings is None:
        t0 = time.perf_counter()
        items, embeddings, _ = build_index(
            datalake_dir=datalake_dir,
            column_name=args.column_name,
            model=model,
            expected_dim=model_dim,
            embedder=embedder,
            precomputed=precomputed,
            device=device,
            batch_cols=args.batch_cols,
            max_cells_per_batch=args.max_cells_per_batch,
        )
        offline_seconds = time.perf_counter() - t0

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump({"items": items, "embeddings": embeddings, "meta": cache_meta}, f)
            LOGGER.info("Saved index cache: %s", cache_path)

    query_specs, query_source_desc = discover_query_specs(
        dataset_dir=dataset_dir,
        datalake_dir=datalake_dir,
        query_source=query_source,
        default_query_column=args.default_query_column,
    )
    if not query_specs:
        raise ValueError("No query columns discovered")
    LOGGER.info("Query source: %s (%s query columns)", query_source_desc, len(query_specs))

    t_online = time.perf_counter()
    query_items, query_embeddings, skipped_queries, _ = encode_queries(
        query_specs=query_specs,
        model=model,
        expected_dim=model_dim,
        embedder=embedder,
        precomputed=precomputed,
        device=device,
        batch_cols=args.batch_cols,
        max_cells_per_batch=args.max_cells_per_batch,
    )

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    corpus_embeddings = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    query_embedding_tensor = torch.from_numpy(query_embeddings).to(device=device, dtype=torch.float32)
    sim = torch.matmul(query_embedding_tensor, corpus_embeddings.t())

    table_to_indices: Dict[str, List[int]] = {}
    for idx, (cand_table, _) in enumerate(items):
        table_to_indices.setdefault(cand_table, []).append(idx)

    neg_inf = torch.finfo(sim.dtype).min
    if not args.include_same_table:
        for q_idx, (q_table, _) in enumerate(query_items):
            idxs = table_to_indices.get(q_table)
            if idxs:
                sim[q_idx, idxs] = neg_inf

    rows_written = 0
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

        n_candidates = sim.shape[1]
        for q_idx, (q_table, q_col) in enumerate(query_items):
            row_scores = sim[q_idx]
            if args.top_k > 0:
                k = min(args.top_k, n_candidates)
                top_scores, top_indices = torch.topk(row_scores, k=k)
            else:
                top_scores, top_indices = torch.sort(row_scores, descending=True)

            for score, cand_idx in zip(top_scores.tolist(), top_indices.tolist()):
                if not np.isfinite(score):
                    continue
                cand_table, cand_col = items[cand_idx]
                if not args.include_same_table and cand_table == q_table:
                    continue
                writer.writerow([q_table, q_col, cand_table, cand_col, f"{score:.6f}"])
                rows_written += 1

    online_seconds = time.perf_counter() - t_online
    LOGGER.info(
        "Done. Wrote %s rows to %s (queries=%s, skipped=%s)",
        rows_written,
        out_csv,
        len(query_items),
        skipped_queries,
    )

    if used_cached_index:
        LOGGER.info("[TIMING] offline_datalake_embedding_seconds=0.000 (loaded cached index)")
    else:
        LOGGER.info("[TIMING] offline_datalake_embedding_seconds=%.3f", offline_seconds)
    LOGGER.info("[TIMING] online_query_seconds=%.3f", online_seconds)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Snoopy pretrained inference baseline and emit evaluator-compatible CSV."
    )
    parser.add_argument("--dataset_dir", help="Dataset root (contains datalake/ and query metadata).")
    parser.add_argument(
        "--datalake_dir",
        help="Path to datalake directory (or parent that contains datalake/).",
    )
    parser.add_argument(
        "--query_source",
        help=(
            "Optional query source: CSV file (query columns / ground truth) or directory of query tables. "
            "If omitted, auto-discovery precedence is: query_columns.csv > autofj_query_columns.csv > "
            "gdc_breakdown_query_columns.csv > groundtruth-joinable.csv > queries/ > datalake/."
        ),
    )
    parser.add_argument("--out_csv", required=True, help="Output CSV path.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to Snoopy pretrained checkpoint (.pth).")
    parser.add_argument(
        "--column_name",
        default="title",
        help="Datalake column name to index (default: title). Use '*' or 'all' for all columns.",
    )
    parser.add_argument(
        "--default_query_column",
        default="title",
        help="Fallback query column when query metadata only provides table names.",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k candidates per query; <=0 returns all.")
    parser.add_argument(
        "--include_same_table",
        action="store_true",
        help="Include candidates from the same table as the query.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--index_cache", help="Optional pickle cache for offline datalake embeddings.")
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Force rebuilding index even if --index_cache exists.",
    )
    parser.add_argument(
        "--fasttext_model_path",
        default=None,
        help="Path to fastText model .bin (default: baselines/snoopy/model/cc.en.300.bin).",
    )
    parser.add_argument(
        "--reuse_precomputed",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Reuse Snoopy precomputed .npy for target_<id>.csv/query_<id>.csv when available "
            "(disable with --no-reuse_precomputed)."
        ),
    )
    parser.add_argument(
        "--precomputed_target_npy",
        default=None,
        help="Override precomputed target npy path (default: baselines/snoopy/precomputed/WDC/target.npy).",
    )
    parser.add_argument(
        "--precomputed_query_npy",
        default=None,
        help="Override precomputed query npy path (default: baselines/snoopy/precomputed/WDC/query.npy).",
    )
    parser.add_argument(
        "--batch_cols",
        type=int,
        default=256,
        help="Maximum number of columns per encoding batch.",
    )
    parser.add_argument(
        "--max_cells_per_batch",
        type=int,
        default=200000,
        help="Maximum total cells per encoding batch.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    baseline_root = Path(__file__).resolve().parent
    if args.fasttext_model_path is None:
        args.fasttext_model_path = str(baseline_root / "model" / "cc.en.300.bin")
    if args.precomputed_target_npy is None:
        args.precomputed_target_npy = str(baseline_root / "precomputed" / "WDC" / "target.npy")
    if args.precomputed_query_npy is None:
        args.precomputed_query_npy = str(baseline_root / "precomputed" / "WDC" / "query.npy")

    try:
        return run_baseline(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
