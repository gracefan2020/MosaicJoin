#!/usr/bin/env python3
"""
Split CSV tables into id/title two-column CSVs.

Example (single file):
  python scripts/breakup_datalake.py datasets/autofj-gdc/datalake/cao.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _is_id_title(header: List[str]) -> bool:
    cols = [c.strip().lower() for c in header]
    return cols == ["id", "title"]


def _detect_id_column(
    header: List[str], explicit: Optional[str] = None, src: str = ""
) -> Tuple[int, str]:
    if explicit:
        for idx, col in enumerate(header):
            if col.strip().lower() == explicit.strip().lower():
                return idx, col
        raise ValueError(f"ID column '{explicit}' not found in {src}")

    for idx, col in enumerate(header):
        if col.strip().lower() == "id":
            return idx, col

    candidates = [
        (idx, col)
        for idx, col in enumerate(header)
        if col.strip().lower().endswith("_id")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(
            f"Warning: multiple *_id columns in {src}; using '{candidates[0][1]}'",
            file=sys.stderr,
        )
        return candidates[0]

    print(f"Warning: no obvious id column in {src}; using '{header[0]}'", file=sys.stderr)
    return 0, header[0]


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    safe = safe.strip("._-")
    return safe or "column"


def _unique_base(base: str, used: Set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}__{i}" in used:
        i += 1
    unique = f"{base}__{i}"
    used.add(unique)
    return unique


def _iter_csvs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    yield from sorted(p for p in input_path.glob("*.csv") if p.is_file())


def _split_csv(
    csv_path: Path,
    out_base: Path,
    id_column: Optional[str],
    include_id_title: bool,
    prefix_table: bool,
    overwrite: bool,
    dry_run: bool,
) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"Skipping empty file: {csv_path}")
            return 0

        if _is_id_title(header) and not include_id_title:
            return 0

        id_idx, id_name = _detect_id_column(header, explicit=id_column, src=str(csv_path))
        data_indices = [i for i in range(len(header)) if i != id_idx]

        os.makedirs(out_base, exist_ok=True)

        used_names: Set[str] = set()
        writers: Dict[int, csv.writer] = {}
        handles = []
        table_prefix = f"{csv_path.stem}__" if prefix_table else ""
        table_suffix = _sanitize_filename(csv_path.stem)

        for i in data_indices:
            col_name = header[i]
            base = _sanitize_filename(f"{table_prefix}{col_name}")
            if table_suffix:
                base = f"{base}_{table_suffix}"
            base = _unique_base(base, used_names)
            out_path = out_base / f"{base}.csv"
            if out_path.exists() and not overwrite:
                print(f"Skipping existing file: {out_path}", file=sys.stderr)
                continue
            if dry_run:
                print(f"Would write: {out_path} (id: {id_name}, title: {col_name})")
                continue
            handle = out_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(["id", "title"])
            writers[i] = writer
            handles.append(handle)

        if dry_run or not writers:
            for handle in handles:
                handle.close()
            return 0

        for row in reader:
            id_val = row[id_idx] if id_idx < len(row) else ""
            for i, writer in writers.items():
                val = row[i] if i < len(row) else ""
                writer.writerow([id_val, val])

        for handle in handles:
            handle.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Break up CSV tables into id/title two-column CSVs."
    )
    parser.add_argument(
        "input",
        type=str,
        help="CSV file or directory containing CSVs to split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (defaults to <input>_split for files or <input>/split_id_title for dirs).",
    )
    parser.add_argument(
        "--layout",
        choices=["flat", "per-table"],
        default=None,
        help="Output layout. flat writes all outputs into one folder; "
        "per-table writes into subfolders per source table.",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        help="Explicit id column name to use (case-insensitive).",
    )
    parser.add_argument(
        "--include_id_title",
        action="store_true",
        help="Also process files already in id/title format.",
    )
    parser.add_argument(
        "--prefix_table",
        action="store_true",
        help="Prefix output filenames with the source table name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    layout = args.layout
    if layout is None:
        layout = "flat" if input_path.is_file() else "per-table"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            input_path.parent / f"{input_path.stem}_split"
            if input_path.is_file()
            else input_path / "split_id_title"
        )

    if input_path.is_dir() and layout == "flat" and not args.prefix_table:
        print(
            "Warning: flat layout without --prefix_table may cause filename collisions.",
            file=sys.stderr,
        )

    for csv_path in _iter_csvs(input_path):
        if layout == "per-table":
            out_base = output_dir / csv_path.stem
        else:
            out_base = output_dir
        _split_csv(
            csv_path,
            out_base,
            id_column=args.id_column,
            include_id_title=args.include_id_title,
            prefix_table=args.prefix_table,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
