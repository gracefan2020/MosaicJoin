#!/usr/bin/env python3
"""
Convert triplet-style benchmark files to AutoFJ-style dataset layout.

Input triplet files:
- query.csv: each row is one query column (cell values)
- target.csv: each row is one target/datalake column (cell values)
- index.csv: each row is ranked target indices for the query row

Output AutoFJ-style files:
- <output_root>/datalake/*.csv        (id,title tables)
- <output_root>/groundtruth-joinable.csv
- <output_root>/join_col_groundtruth.csv
- <output_root>/autofj_query_columns.csv
- <output_root>/query_table_mapping.csv
- <output_root>/target_table_mapping.csv

Example:
  python3 src/convert_triplet_to_autofj.py \
    --input_root datasets/Lake/opendata \
    --output_root /Users/yifanwu/Desktop/VIDA/tmp/SemSketch/datasets/opendata-autofj \
    --index_offset 0 \
    --groundtruth_mode all
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List

from paths import resolve_path


def read_csv_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_index_row(row: List[str], row_idx: int) -> List[int]:
    values: List[int] = []
    for cell in row:
        token = cell.strip()
        if not token:
            continue
        parts = token.split() if any(ch.isspace() for ch in token) else [token]
        for part in parts:
            if not part:
                continue
            try:
                values.append(int(part))
            except ValueError as e:
                raise ValueError(
                    f"Invalid integer '{part}' at index.csv row {row_idx + 1}"
                ) from e
    return values


def read_index_rows(path: Path) -> List[List[int]]:
    rows: List[List[int]] = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            rows.append(parse_index_row(row, row_idx))
    return rows


def write_id_title_table(
    out_path: Path,
    values: List[str],
    id_start: int,
    drop_empty_cells: bool,
) -> int:
    num_rows = 0
    next_id = id_start
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title"])
        for value in values:
            text = "" if value is None else str(value)
            if drop_empty_cells and text == "":
                continue
            writer.writerow([next_id, text])
            next_id += 1
            num_rows += 1
    return num_rows


def write_datalake_tables(
    columns: List[List[str]],
    datalake_dir: Path,
    prefix: str,
    width: int,
    id_start: int,
    drop_empty_cells: bool,
) -> tuple[List[str], List[int]]:
    table_files: List[str] = []
    value_counts: List[int] = []

    total = len(columns)
    for i, values in enumerate(columns):
        file_name = f"{prefix}{i:0{width}d}.csv"
        out_path = datalake_dir / file_name
        n_values = write_id_title_table(out_path, values, id_start, drop_empty_cells)
        table_files.append(file_name)
        value_counts.append(n_values)
        if (i + 1) % 1000 == 0 or i + 1 == total:
            print(f"  wrote {i + 1}/{total} tables to {datalake_dir}")
    return table_files, value_counts


def normalize_index_rows(
    raw_rows: List[List[int]],
    num_queries: int,
    num_targets: int,
    index_offset: int,
    deduplicate: bool,
    index_topk: int | None,
) -> tuple[List[List[int]], int]:
    normalized_rows: List[List[int]] = []
    dropped = 0
    for q_idx in range(num_queries):
        raw = raw_rows[q_idx] if q_idx < len(raw_rows) else []
        seen = set()
        cleaned: List[int] = []
        for raw_id in raw:
            target_id = raw_id - index_offset
            if target_id < 0 or target_id >= num_targets:
                dropped += 1
                continue
            if deduplicate and target_id in seen:
                continue
            seen.add(target_id)
            cleaned.append(target_id)
            if index_topk is not None and len(cleaned) >= index_topk:
                break
        normalized_rows.append(cleaned)
    return normalized_rows, dropped


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert triplet CSVs (query/target/index) to AutoFJ-style dataset."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Directory containing query.csv, target.csv, index.csv.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory in AutoFJ style.",
    )
    parser.add_argument(
        "--query_csv",
        type=str,
        default=None,
        help="Optional explicit query.csv path (overrides --input_root/query.csv).",
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        default=None,
        help="Optional explicit target.csv path (overrides --input_root/target.csv).",
    )
    parser.add_argument(
        "--index_csv",
        type=str,
        default=None,
        help="Optional explicit index.csv path (overrides --input_root/index.csv).",
    )
    parser.add_argument(
        "--query_prefix",
        type=str,
        default="query_",
        help="Filename prefix for query tables in datalake/.",
    )
    parser.add_argument(
        "--target_prefix",
        type=str,
        default="target_",
        help="Filename prefix for target tables in datalake/.",
    )
    parser.add_argument(
        "--name_width",
        type=int,
        default=6,
        help="Zero-padding width for generated table filenames.",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default="pair_",
        help="Prefix for dataset IDs in groundtruth-joinable.csv.",
    )
    parser.add_argument(
        "--id_start",
        type=int,
        default=0,
        help="Starting id for each generated id,title table.",
    )
    parser.add_argument(
        "--index_offset",
        type=int,
        default=0,
        help=(
            "Offset to subtract from raw index IDs (0 for zero-based, "
            "1 for one-based index.csv)."
        ),
    )
    parser.add_argument(
        "--index_topk",
        type=int,
        default=None,
        help="Keep only top-k targets per query from index.csv.",
    )
    parser.add_argument(
        "--groundtruth_mode",
        choices=["all", "top1"],
        default="all",
        help="Use all valid index targets per query, or only top1.",
    )
    parser.add_argument(
        "--keep_duplicate_index",
        action="store_true",
        help="Keep duplicate target IDs if they appear in one index row.",
    )
    parser.add_argument(
        "--drop_empty_cells",
        action="store_true",
        help="Drop empty cells when writing id,title rows.",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Delete existing output_root before writing new files.",
    )
    args = parser.parse_args()

    input_root = Path(resolve_path(args.input_root))
    output_root = Path(resolve_path(args.output_root))

    query_csv = Path(resolve_path(args.query_csv)) if args.query_csv else input_root / "query.csv"
    target_csv = Path(resolve_path(args.target_csv)) if args.target_csv else input_root / "target.csv"
    index_csv = Path(resolve_path(args.index_csv)) if args.index_csv else input_root / "index.csv"

    for required_path in [query_csv, target_csv, index_csv]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing file: {required_path}")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    datalake_dir = output_root / "datalake"
    os.makedirs(datalake_dir, exist_ok=True)

    print("Loading input files...")
    query_columns = read_csv_rows(query_csv)
    target_columns = read_csv_rows(target_csv)
    raw_index_rows = read_index_rows(index_csv)

    num_queries = len(query_columns)
    num_targets = len(target_columns)
    if len(raw_index_rows) != num_queries:
        print(
            f"Warning: index.csv rows ({len(raw_index_rows)}) != query.csv rows ({num_queries}). "
            "Missing rows will be treated as empty."
        )

    print(f"Loaded {num_queries} query columns")
    print(f"Loaded {num_targets} target columns")
    print(f"Loaded {len(raw_index_rows)} index rows")

    name_width = max(
        args.name_width,
        len(str(max(num_queries - 1, 0))),
        len(str(max(num_targets - 1, 0))),
    )

    print("Writing query tables...")
    query_table_files, query_value_counts = write_datalake_tables(
        query_columns,
        datalake_dir=datalake_dir,
        prefix=args.query_prefix,
        width=name_width,
        id_start=args.id_start,
        drop_empty_cells=args.drop_empty_cells,
    )

    print("Writing target tables...")
    target_table_files, target_value_counts = write_datalake_tables(
        target_columns,
        datalake_dir=datalake_dir,
        prefix=args.target_prefix,
        width=name_width,
        id_start=args.id_start,
        drop_empty_cells=args.drop_empty_cells,
    )

    normalized_index_rows, dropped_refs = normalize_index_rows(
        raw_rows=raw_index_rows,
        num_queries=num_queries,
        num_targets=num_targets,
        index_offset=args.index_offset,
        deduplicate=not args.keep_duplicate_index,
        index_topk=args.index_topk,
    )
    if dropped_refs > 0:
        print(f"Warning: dropped {dropped_refs} out-of-range target references from index.csv")

    pair_rows: List[List[str]] = []
    join_col_rows: List[List[str]] = []
    query_spec_rows: List[List[str]] = []

    for q_idx, left_table in enumerate(query_table_files):
        query_spec_rows.append([left_table, "title"])
        targets = normalized_index_rows[q_idx]
        if args.groundtruth_mode == "top1":
            targets = targets[:1]
        for rank, t_idx in enumerate(targets, start=1):
            right_table = target_table_files[t_idx]
            if args.groundtruth_mode == "top1":
                dataset_id = f"{args.dataset_prefix}{q_idx:0{name_width}d}"
            else:
                dataset_id = f"{args.dataset_prefix}{q_idx:0{name_width}d}_r{rank:03d}"
            pair_rows.append([dataset_id, left_table, right_table])
            join_col_rows.append([left_table, right_table, "title", "title"])

    write_csv(
        output_root / "groundtruth-joinable.csv",
        ["dataset", "left_table", "right_table"],
        pair_rows,
    )
    write_csv(
        output_root / "join_col_groundtruth.csv",
        ["source_table", "target_table", "source_column", "target_column"],
        join_col_rows,
    )
    write_csv(
        output_root / "autofj_query_columns.csv",
        ["target_ds", "target_attr"],
        query_spec_rows,
    )

    query_mapping_rows = [
        [str(i), query_table_files[i], str(query_value_counts[i])] for i in range(num_queries)
    ]
    target_mapping_rows = [
        [str(i), target_table_files[i], str(target_value_counts[i])] for i in range(num_targets)
    ]
    write_csv(
        output_root / "query_table_mapping.csv",
        ["query_row_index", "table_file", "num_values"],
        query_mapping_rows,
    )
    write_csv(
        output_root / "target_table_mapping.csv",
        ["target_row_index", "table_file", "num_values"],
        target_mapping_rows,
    )

    print("Done.")
    print(f"Output root: {output_root}")
    print(f"Datalake tables: {num_queries + num_targets}")
    print(f"groundtruth-joinable rows: {len(pair_rows)}")
    print(f"join_col_groundtruth rows: {len(join_col_rows)}")
    print(f"query specs: {len(query_spec_rows)}")


if __name__ == "__main__":
    main()
