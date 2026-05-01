#!/usr/bin/env python3
"""Build KOIOS input sets and its SQLite embedding database.

KOIOS expects a data lake directory where each file is one set and each line is
one set element. Its semantic C++ path also expects a local SQLite database named
``ft.sqlite3`` with a table ``wv(word TEXT, vec BLOB)`` where every vector is a
300-dimensional float32 blob.

This script adapts MosaicJoin CSV columns to that shape without modifying KOIOS:

* selected CSV columns are exported as one-value-per-line KOIOS set files;
* the vocabulary from those set files is embedded with a SentenceTransformer
  model;
* embeddings are L2-normalized and projected/padded to KOIOS's 300 dimensions;
* rows are written to an SQLite database compatible with KOIOS's C++ code.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger("koios-db")
KOIOS_DIM = 300
EPS = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_SETS_DIR = "work/sets"
DEFAULT_OUT_DB = "work/ft.sqlite3"
DEFAULT_MANIFEST_JSON = "work/sets_manifest.json"
SUPPORTED_DATASETS = ("autofj", "autofj-wdc", "freyja", "freyja-wdc", "wt", "wt-wdc")


@dataclass
class ColumnSpec:
    table_name: str
    column_name: str
    csv_path: str
    set_file: str
    value_count: int


def setup_logging(level_name: str) -> None:
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


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
            return next(csv.reader(f), []) or []
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
        return [name for name in header if name.strip().lower() not in {"id", "index"}]

    matched = canonical_column_name(header, column_name)
    return [matched] if matched else []


def normalize_text(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return ""
    return " ".join(value.split())


def safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._")
    return safe or "column"


def set_filename(table_name: str, column_name: str) -> str:
    return f"{safe_component(table_name)}__{safe_component(column_name)}"


def read_unique_column_values(
    csv_path: Path,
    column_name: str,
    max_values: int,
) -> List[str]:
    values: List[str] = []
    seen: Set[str] = set()

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return values

        try:
            col_idx = header.index(column_name)
        except ValueError:
            return values

        for row in reader:
            raw = row[col_idx] if col_idx < len(row) else ""
            norm = normalize_text(raw)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            values.append(norm)
            if max_values > 0 and len(values) >= max_values:
                break

    return values


def iter_set_values(sets_dir: Path) -> Iterator[str]:
    for set_path in sorted(sets_dir.iterdir()):
        if not set_path.is_file():
            continue
        with set_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                value = normalize_text(line)
                if value:
                    yield value


def export_koios_sets(
    datalake_dir: Path,
    sets_dir: Path,
    column_name: str,
    max_datalake_tables: int,
    max_values_per_column: int,
) -> Tuple[List[ColumnSpec], Set[str]]:
    sets_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(datalake_dir.glob("*.csv"))
    if max_datalake_tables > 0:
        csv_files = csv_files[:max_datalake_tables]
    if not csv_files:
        raise ValueError(f"No CSV files found in datalake: {datalake_dir}")

    manifest: List[ColumnSpec] = []
    vocab: Set[str] = set()

    for idx, csv_path in enumerate(csv_files, start=1):
        selected_columns = discover_selected_columns(csv_path, column_name)
        if not selected_columns:
            continue

        for selected in selected_columns:
            values = read_unique_column_values(
                csv_path=csv_path,
                column_name=selected,
                max_values=max_values_per_column,
            )
            if not values:
                continue

            out_name = set_filename(csv_path.name, selected)
            out_path = sets_dir / out_name
            with out_path.open("w", encoding="utf-8", newline="\n") as f:
                for value in values:
                    f.write(value)
                    f.write("\n")
                    vocab.add(value)

            manifest.append(
                ColumnSpec(
                    table_name=csv_path.name,
                    column_name=selected,
                    csv_path=str(csv_path),
                    set_file=out_name,
                    value_count=len(values),
                )
            )

        if idx % 500 == 0:
            LOGGER.info(
                "exported through %s/%s CSVs, set_files=%s, vocab=%s",
                idx,
                len(csv_files),
                len(manifest),
                len(vocab),
            )

    if not manifest:
        raise ValueError(
            f"No columns were exported from {datalake_dir}; check --column_name={column_name!r}"
        )
    return manifest, vocab


def default_model_name() -> str:
    candidates = [
        Path("/scratch/yfw215/.huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots"),
        Path.home() / ".cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots",
        Path.home() / ".huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots",
    ]
    for snapshots_dir in candidates:
        if not snapshots_dir.is_dir():
            continue
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return "BAAI/bge-base-en-v1.5"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_sentence_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_device(device)
    LOGGER.info("loading SentenceTransformer model=%s device=%s", model_name, resolved_device)
    return SentenceTransformer(model_name, device=resolved_device)


def encode_batch(model, texts: List[str], batch_size: int, encode_mode: str):
    kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "batch_size": batch_size,
        "show_progress_bar": False,
    }

    def call_encode(fn):
        try:
            return fn(texts, **kwargs)
        except TypeError:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("show_progress_bar", None)
            return fn(texts, **fallback_kwargs)

    if encode_mode == "document" and hasattr(model, "encode_document"):
        return call_encode(model.encode_document)
    if encode_mode == "query" and hasattr(model, "encode_query"):
        return call_encode(model.encode_query)
    return call_encode(model.encode)


def normalize_rows(matrix):
    import numpy as np

    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, EPS)


def build_projection(source_dim: int, target_dim: int, seed: int):
    import numpy as np

    if source_dim <= target_dim:
        return None
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(target_dim)
    return rng.normal(0.0, scale, size=(source_dim, target_dim)).astype(np.float32)


def to_koios_vectors(embeddings, projection, target_dim: int):
    import numpy as np

    arr = normalize_rows(embeddings)
    source_dim = arr.shape[1]

    if source_dim > target_dim:
        if projection is None:
            raise ValueError("projection matrix is required when source_dim > target_dim")
        arr = arr @ projection
    elif source_dim < target_dim:
        padded = np.zeros((arr.shape[0], target_dim), dtype=np.float32)
        padded[:, :source_dim] = arr
        arr = padded

    arr = normalize_rows(arr)
    return np.asarray(arr, dtype="<f4")


def init_database(db_path: Path, overwrite: bool) -> Tuple[sqlite3.Connection, Path]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not overwrite:
        raise FileExistsError(f"Database already exists: {db_path}; use --overwrite")
    tmp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(str(tmp_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE wv (word TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return conn, tmp_path


def write_metadata(conn: sqlite3.Connection, metadata: Dict[str, object]) -> None:
    rows = [(str(key), json.dumps(value, sort_keys=True)) for key, value in metadata.items()]
    conn.executemany("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", rows)


def build_embedding_db(
    vocab: Sequence[str],
    db_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    projection_seed: int,
    encode_mode: str,
    overwrite: bool,
    metadata: Dict[str, object],
) -> None:
    import numpy as np

    model = load_sentence_model(model_name=model_name, device=device)
    conn, tmp_path = init_database(db_path=db_path, overwrite=overwrite)
    projection = None
    source_dim: Optional[int] = None
    inserted = 0
    started = time.perf_counter()

    try:
        for start in range(0, len(vocab), batch_size):
            batch_words = list(vocab[start : start + batch_size])
            embeddings = encode_batch(
                model=model,
                texts=batch_words,
                batch_size=batch_size,
                encode_mode=encode_mode,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if source_dim is None:
                source_dim = int(embeddings.shape[1])
                projection = build_projection(source_dim, KOIOS_DIM, projection_seed)
                LOGGER.info("source_dim=%s, koios_dim=%s", source_dim, KOIOS_DIM)

            vectors = to_koios_vectors(embeddings, projection, KOIOS_DIM)
            rows = [
                (word, sqlite3.Binary(vector.tobytes(order="C")))
                for word, vector in zip(batch_words, vectors)
            ]
            conn.executemany("INSERT OR REPLACE INTO wv(word, vec) VALUES (?, ?)", rows)
            inserted += len(rows)

            if inserted % max(batch_size * 10, 1000) == 0 or inserted == len(vocab):
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "embedded %s/%s values (%.1f values/s)",
                    inserted,
                    len(vocab),
                    inserted / elapsed if elapsed > 0 else 0.0,
                )

        write_metadata(
            conn,
            {
                **metadata,
                "model_name": model_name,
                "source_dim": source_dim,
                "koios_dim": KOIOS_DIM,
                "projection": "identity_or_padding" if source_dim and source_dim <= KOIOS_DIM else "gaussian_random_projection",
                "projection_seed": projection_seed,
                "encode_mode": encode_mode,
                "vocab_size": len(vocab),
                "created_at_unix": int(time.time()),
            },
        )
        conn.commit()
    except Exception:
        conn.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        conn.close()
        os.replace(tmp_path, db_path)


def load_vocab_from_sets(sets_dir: Path) -> Set[str]:
    vocab = set(iter_set_values(sets_dir))
    if not vocab:
        raise ValueError(f"No values found in existing KOIOS sets dir: {sets_dir}")
    return vocab


def install_database(db_path: Path, koios_repo: Path, overwrite: bool) -> Path:
    dest = koios_repo / "ft.sqlite3"
    if dest.exists() and not overwrite:
        raise FileExistsError(f"KOIOS database already exists: {dest}; use --overwrite")
    shutil.copy2(db_path, dest)
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MosaicJoin CSV columns and build KOIOS-compatible ft.sqlite3."
    )
    parser.add_argument(
        "--dataset_name",
        choices=SUPPORTED_DATASETS,
        help="Shortcut for datasets/<name> and baselines/koios/work/<name> outputs.",
    )
    parser.add_argument("--dataset_dir", help="Dataset root containing datalake/.")
    parser.add_argument("--datalake_dir", help="Datalake directory or parent containing datalake/.")
    parser.add_argument(
        "--sets_dir",
        default=DEFAULT_SETS_DIR,
        help="Output KOIOS set directory, or existing set directory with --skip_set_export.",
    )
    parser.add_argument(
        "--out_db",
        default=DEFAULT_OUT_DB,
        help="Output SQLite database path.",
    )
    parser.add_argument(
        "--column_name",
        default="title",
        help="CSV column to export. Use '*' or 'all' to export all non-id columns.",
    )
    parser.add_argument(
        "--max_datalake_tables",
        type=int,
        default=0,
        help="Optional cap on CSV files to export (0 = all).",
    )
    parser.add_argument(
        "--max_values_per_column",
        type=int,
        default=0,
        help="Optional cap on unique values per column (0 = all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--skip_set_export",
        action="store_true",
        help="Build the database from an existing --sets_dir instead of reading CSVs.",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="SentenceTransformer model name or local path. Defaults to cached BGE if present.",
    )
    parser.add_argument("--device", default="auto", help="SentenceTransformer device: auto/cpu/cuda/etc.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--projection_seed",
        type=int,
        default=13,
        help="Seed for deterministic projection to KOIOS's 300 dimensions.",
    )
    parser.add_argument(
        "--encode_mode",
        choices=["standard", "document", "query"],
        default="document",
        help="Use encode_document/encode_query when the model exposes those methods.",
    )
    parser.add_argument(
        "--manifest_json",
        default=DEFAULT_MANIFEST_JSON,
        help="Where to write the KOIOS set manifest.",
    )
    parser.add_argument(
        "--koios_repo",
        default="/scratch/yfw215/tmp/koios-semantic-search",
        help="KOIOS source directory used by --install.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Copy --out_db to <koios_repo>/ft.sqlite3 after building.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output database and installed ft.sqlite3.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.dataset_name:
        if not args.dataset_dir and not args.datalake_dir:
            args.dataset_dir = str(REPO_ROOT / "datasets" / args.dataset_name)
        if args.sets_dir == DEFAULT_SETS_DIR:
            args.sets_dir = str(BASELINE_ROOT / "work" / args.dataset_name / "sets")
        if args.out_db == DEFAULT_OUT_DB:
            args.out_db = str(BASELINE_ROOT / "work" / args.dataset_name / "ft.sqlite3")
        if args.manifest_json == DEFAULT_MANIFEST_JSON:
            args.manifest_json = str(BASELINE_ROOT / "work" / args.dataset_name / "sets_manifest.json")

    sets_dir = Path(args.sets_dir).expanduser().resolve()
    out_db = Path(args.out_db).expanduser().resolve()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    model_name = args.model_name or default_model_name()
    offline_start = time.perf_counter()

    if args.skip_set_export:
        LOGGER.info("loading vocabulary from existing KOIOS sets: %s", sets_dir)
        manifest: List[ColumnSpec] = []
        vocab = load_vocab_from_sets(sets_dir)
    else:
        if args.datalake_dir:
            datalake_dir = resolve_datalake_dir(Path(args.datalake_dir).expanduser().resolve())
        elif args.dataset_dir:
            datalake_dir = resolve_datalake_dir(Path(args.dataset_dir).expanduser().resolve())
        else:
            raise ValueError("Provide --dataset_dir, --datalake_dir, or --skip_set_export.")

        LOGGER.info("datalake_dir=%s", datalake_dir)
        LOGGER.info("sets_dir=%s", sets_dir)
        manifest, vocab = export_koios_sets(
            datalake_dir=datalake_dir,
            sets_dir=sets_dir,
            column_name=args.column_name,
            max_datalake_tables=args.max_datalake_tables,
            max_values_per_column=args.max_values_per_column,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps([asdict(spec) for spec in manifest], indent=2),
            encoding="utf-8",
        )
        LOGGER.info("wrote manifest: %s (%s set files)", manifest_path, len(manifest))

    sorted_vocab = sorted(vocab)
    LOGGER.info("unique values to embed: %s", len(sorted_vocab))
    if not sorted_vocab:
        raise ValueError("Empty vocabulary; nothing to embed.")

    build_embedding_db(
        vocab=sorted_vocab,
        db_path=out_db,
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
        projection_seed=args.projection_seed,
        encode_mode=args.encode_mode,
        overwrite=args.overwrite,
        metadata={
            "sets_dir": str(sets_dir),
            "manifest_json": str(manifest_path) if manifest else "",
            "column_name": args.column_name,
            "max_datalake_tables": args.max_datalake_tables,
            "max_values_per_column": args.max_values_per_column,
        },
    )
    offline_seconds = time.perf_counter() - offline_start
    LOGGER.info("wrote KOIOS embedding database: %s", out_db)
    LOGGER.info("[TIMING] offline_datalake_embedding_seconds=%.3f", offline_seconds)

    if args.install:
        installed = install_database(
            db_path=out_db,
            koios_repo=Path(args.koios_repo).expanduser().resolve(),
            overwrite=args.overwrite,
        )
        LOGGER.info("installed database for KOIOS: %s", installed)

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
