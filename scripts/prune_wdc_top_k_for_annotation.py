#!/usr/bin/env python3
"""Prepare WDC top-k recommendations for manual annotation.

Workflow:
1. Read recommendation CSVs under scripts/wdc_top_k (e.g., deepjoin_ft_wt-wdc.csv).
2. Keep only WDC candidates (candidate_table starts with target_ by default).
3. Keep top-k per (query_table, query_column) within each method file.
4. Union unique (query_table, query_column, candidate_table, candidate_column)
   pairs across methods per benchmark.
5. Optionally apply lexical filtering on sampled unique values.
6. Export annotation-ready CSVs with sampled unique values (64 by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz  # type: ignore
except Exception:
    rapidfuzz_fuzz = None


TOPK_FILE_RE = re.compile(r"^(?P<method>.+?)_ft_(?P<benchmark>.+)\.csv$")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Prune WDC top-k recommendations and export annotation CSVs."
    )
    parser.add_argument(
        "--topk-dir",
        type=Path,
        default=script_dir / "wdc_top_k",
        help="Directory containing method top-k CSVs.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=project_root / "datasets",
        help="Dataset root that contains <benchmark>/datalake.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "wdc_top_k_annotation",
        help="Where to write annotation CSVs.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional benchmark IDs to process (e.g., autofj-wdc wt-wdc).",
    )
    parser.add_argument(
        "--target-prefix",
        default="target_",
        help="Keep candidates whose table starts with this prefix.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Per method and per query attribute, keep top-k WDC candidates before union. "
            "Use <= 0 to disable this cap."
        ),
    )
    parser.add_argument(
        "--max-export-values",
        type=int,
        default=64,
        help="Max number of unique values to export per query/candidate column.",
    )
    parser.add_argument(
        "--max-load-values",
        type=int,
        default=512,
        help="Max unique values to load per column (used for export and lexical score).",
    )
    parser.add_argument(
        "--lexical-filter",
        action="store_true",
        help="Enable lexical filtering based on query/candidate value similarity.",
    )
    parser.add_argument(
        "--lexical-threshold",
        type=float,
        default=0.20,
        help="Keep rows with lexical score >= threshold (0-1).",
    )
    parser.add_argument(
        "--max-lexical-values",
        type=int,
        default=128,
        help="Max sampled unique values per side used for lexical scoring.",
    )
    parser.add_argument(
        "--disable-rapidfuzz",
        action="store_true",
        help="Do not use rapidfuzz even if installed.",
    )
    parser.add_argument(
        "--rapidfuzz-value-limit",
        type=int,
        default=32,
        help="Max sampled values per side for pairwise rapidfuzz comparisons.",
    )
    return parser.parse_args()


def discover_topk_files(topk_dir: Path) -> Dict[str, List[Tuple[str, Path]]]:
    grouped: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for csv_path in sorted(topk_dir.glob("*.csv")):
        match = TOPK_FILE_RE.match(csv_path.name)
        if not match:
            continue
        method = match.group("method")
        benchmark = match.group("benchmark")
        grouped[benchmark].append((method, csv_path))
    return grouped


def read_recommendation_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"query_table", "query_column", "candidate_table", "candidate_column"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            query_table = (row.get("query_table") or "").strip()
            query_column = (row.get("query_column") or "").strip()
            candidate_table = (row.get("candidate_table") or "").strip()
            candidate_column = (row.get("candidate_column") or "").strip()
            if not query_table or not candidate_table:
                continue

            sim_raw = (row.get("similarity_score") or "").strip()
            similarity_score: Optional[float]
            try:
                similarity_score = float(sim_raw) if sim_raw else None
            except ValueError:
                similarity_score = None

            yield {
                "query_table": query_table,
                "query_column": query_column,
                "candidate_table": candidate_table,
                "candidate_column": candidate_column,
                "similarity_score": similarity_score,
            }


def resolve_column_name(fieldnames: Sequence[str], requested: str) -> Optional[str]:
    if not fieldnames:
        return None
    if requested in fieldnames:
        return requested

    lower_to_name = {name.lower(): name for name in fieldnames}
    lowered = requested.lower()
    if lowered in lower_to_name:
        return lower_to_name[lowered]

    if len(fieldnames) == 1:
        return fieldnames[0]

    if "title" in lower_to_name:
        return lower_to_name["title"]

    return None


class ValueCache:
    def __init__(self, datalake_dir: Path, max_load_values: int):
        self.datalake_dir = datalake_dir
        self.max_load_values = max_load_values
        self._cache: Dict[Tuple[str, str], List[str]] = {}

    def _resolve_table_path(self, table_name: str) -> Optional[Path]:
        direct = self.datalake_dir / table_name
        if direct.is_file():
            return direct
        if not table_name.endswith(".csv"):
            alt = self.datalake_dir / f"{table_name}.csv"
            if alt.is_file():
                return alt
        return None

    def get_unique_values(self, table_name: str, column_name: str) -> List[str]:
        key = (table_name, column_name)
        if key in self._cache:
            return self._cache[key]

        table_path = self._resolve_table_path(table_name)
        if table_path is None:
            self._cache[key] = []
            return self._cache[key]

        values: List[str] = []
        seen = set()
        try:
            with table_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    self._cache[key] = []
                    return self._cache[key]
                resolved_col = resolve_column_name(reader.fieldnames, column_name)
                if resolved_col is None:
                    self._cache[key] = []
                    return self._cache[key]

                for row in reader:
                    raw = row.get(resolved_col)
                    if raw is None:
                        continue
                    val = str(raw).strip()
                    if not val or val in seen:
                        continue
                    seen.add(val)
                    values.append(val)
                    if len(values) >= self.max_load_values:
                        break
        except OSError:
            values = []

        self._cache[key] = values
        return values


def token_set(values: Sequence[str], max_values: int) -> set:
    out = set()
    for value in values[:max_values]:
        for tok in TOKEN_RE.findall(value.lower()):
            out.add(tok)
    return out


def token_overlap_score(query_values: Sequence[str], candidate_values: Sequence[str], max_values: int) -> float:
    q_tokens = token_set(query_values, max_values=max_values)
    c_tokens = token_set(candidate_values, max_values=max_values)
    if not q_tokens or not c_tokens:
        return 0.0

    inter = len(q_tokens & c_tokens)
    union = len(q_tokens | c_tokens)
    min_size = min(len(q_tokens), len(c_tokens))
    jaccard = inter / union if union else 0.0
    containment = inter / min_size if min_size else 0.0
    return max(jaccard, containment)


def rapidfuzz_score(
    query_values: Sequence[str],
    candidate_values: Sequence[str],
    max_per_side: int,
) -> float:
    if rapidfuzz_fuzz is None:
        return 0.0

    best = 0.0
    for qv in query_values[:max_per_side]:
        for cv in candidate_values[:max_per_side]:
            score = float(rapidfuzz_fuzz.token_set_ratio(qv, cv)) / 100.0
            if score > best:
                best = score
                if best >= 1.0:
                    return 1.0
    return best


def lexical_similarity(
    query_values: Sequence[str],
    candidate_values: Sequence[str],
    max_lexical_values: int,
    use_rapidfuzz: bool,
    rapidfuzz_value_limit: int,
) -> float:
    if not query_values or not candidate_values:
        return 0.0

    score = token_overlap_score(query_values, candidate_values, max_values=max_lexical_values)
    if use_rapidfuzz:
        score = max(
            score,
            rapidfuzz_score(
                query_values[:max_lexical_values],
                candidate_values[:max_lexical_values],
                max_per_side=rapidfuzz_value_limit,
            ),
        )
    return score


def aggregate_candidates(
    benchmark: str,
    method_files: Sequence[Tuple[str, Path]],
    target_prefix: str,
    top_k: int,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    stats = {
        "input_rows": 0,
        "wdc_rows": 0,
        "wdc_rows_after_topk": 0,
    }
    merged: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for method, path in method_files:
        rows_by_query_attr: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in read_recommendation_rows(path):
            stats["input_rows"] += 1
            candidate_table = str(row["candidate_table"])
            if not candidate_table.startswith(target_prefix):
                continue
            stats["wdc_rows"] += 1
            qkey = (str(row["query_table"]), str(row["query_column"]))
            rows_by_query_attr[qkey].append(row)

        selected_rows: List[Dict[str, object]] = []
        for rows in rows_by_query_attr.values():
            if top_k > 0 and len(rows) > top_k:
                rows = sorted(
                    rows,
                    key=lambda r: (
                        isinstance(r.get("similarity_score"), float),
                        float(r["similarity_score"]) if isinstance(r.get("similarity_score"), float) else float("-inf"),
                    ),
                    reverse=True,
                )[:top_k]
            selected_rows.extend(rows)

        stats["wdc_rows_after_topk"] += len(selected_rows)

        for row in selected_rows:
            candidate_table = str(row["candidate_table"])
            key = (
                str(row["query_table"]),
                str(row["query_column"]),
                candidate_table,
                str(row["candidate_column"]),
            )
            if key not in merged:
                merged[key] = {
                    "benchmark": benchmark,
                    "query_table": row["query_table"],
                    "query_column": row["query_column"],
                    "candidate_table": row["candidate_table"],
                    "candidate_column": row["candidate_column"],
                    "methods": set(),
                    "method_scores": {},
                }

            entry = merged[key]
            methods = entry["methods"]
            assert isinstance(methods, set)
            methods.add(method)

            sim = row.get("similarity_score")
            if isinstance(sim, float):
                scores = entry["method_scores"]
                assert isinstance(scores, dict)
                prev = scores.get(method)
                if prev is None or sim > prev:
                    scores[method] = sim

    records = [merged[k] for k in sorted(merged.keys())]
    stats["unique_pairs"] = len(records)
    return records, stats


def write_output_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "benchmark",
        "query_table",
        "query_column",
        "candidate_table",
        "candidate_column",
        "methods",
        "method_scores",
        "lexical_score",
        "query_unique_values",
        "candidate_unique_values",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    topk_dir = args.topk_dir.expanduser().resolve()
    datasets_root = args.datasets_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not topk_dir.is_dir():
        raise FileNotFoundError(f"top-k directory not found: {topk_dir}")
    if not datasets_root.is_dir():
        raise FileNotFoundError(f"datasets root not found: {datasets_root}")

    grouped = discover_topk_files(topk_dir)
    if not grouped:
        raise ValueError(f"No top-k files found in {topk_dir}")

    selected_benchmarks = set(args.benchmarks) if args.benchmarks else None
    if selected_benchmarks:
        grouped = {k: v for k, v in grouped.items() if k in selected_benchmarks}
        missing = selected_benchmarks - set(grouped.keys())
        if missing:
            print(f"[WARN] Requested benchmarks not found in top-k dir: {sorted(missing)}")

    if not grouped:
        raise ValueError("No benchmarks selected for processing.")

    use_rapidfuzz = args.lexical_filter and (not args.disable_rapidfuzz) and rapidfuzz_fuzz is not None
    if args.lexical_filter and not use_rapidfuzz:
        print("[INFO] rapidfuzz unavailable/disabled; lexical filter will use token overlap only.")

    max_load_values = max(args.max_load_values, args.max_export_values, args.max_lexical_values)

    for benchmark in sorted(grouped.keys()):
        method_files = grouped[benchmark]
        dataset_dir = datasets_root / benchmark
        datalake_dir = dataset_dir / "datalake"
        if not datalake_dir.is_dir():
            print(f"[WARN] Skip benchmark {benchmark}: datalake not found at {datalake_dir}")
            continue

        records, stats = aggregate_candidates(
            benchmark=benchmark,
            method_files=method_files,
            target_prefix=args.target_prefix,
            top_k=args.top_k,
        )
        value_cache = ValueCache(datalake_dir=datalake_dir, max_load_values=max_load_values)

        exported: List[Dict[str, object]] = []
        lexical_dropped = 0
        missing_value_rows = 0

        for rec in records:
            query_table = str(rec["query_table"])
            query_column = str(rec["query_column"])
            candidate_table = str(rec["candidate_table"])
            candidate_column = str(rec["candidate_column"])

            query_values = value_cache.get_unique_values(query_table, query_column)
            candidate_values = value_cache.get_unique_values(candidate_table, candidate_column)
            if not query_values or not candidate_values:
                missing_value_rows += 1

            lexical_score = ""
            if args.lexical_filter:
                score = lexical_similarity(
                    query_values,
                    candidate_values,
                    max_lexical_values=args.max_lexical_values,
                    use_rapidfuzz=use_rapidfuzz,
                    rapidfuzz_value_limit=args.rapidfuzz_value_limit,
                )
                lexical_score = f"{score:.4f}"
                if score < args.lexical_threshold:
                    lexical_dropped += 1
                    continue

            methods = sorted(str(m) for m in rec["methods"])
            method_scores = rec["method_scores"]
            if isinstance(method_scores, dict):
                score_payload = {k: method_scores[k] for k in sorted(method_scores.keys())}
            else:
                score_payload = {}

            exported.append(
                {
                    "benchmark": benchmark,
                    "query_table": query_table,
                    "query_column": query_column,
                    "candidate_table": candidate_table,
                    "candidate_column": candidate_column,
                    "methods": ";".join(methods),
                    "method_scores": json.dumps(score_payload, ensure_ascii=False, sort_keys=True),
                    "lexical_score": lexical_score,
                    "query_unique_values": json.dumps(
                        query_values[: args.max_export_values], ensure_ascii=False
                    ),
                    "candidate_unique_values": json.dumps(
                        candidate_values[: args.max_export_values], ensure_ascii=False
                    ),
                }
            )

        out_path = output_dir / f"{benchmark}_annotation_candidates.csv"
        write_output_csv(out_path, exported)

        print(
            f"[{benchmark}] input_rows={stats['input_rows']} "
            f"wdc_rows={stats['wdc_rows']} wdc_rows_after_topk={stats['wdc_rows_after_topk']} "
            f"unique_pairs={stats['unique_pairs']} "
            f"lexical_dropped={lexical_dropped} missing_value_pairs={missing_value_rows} "
            f"exported={len(exported)} output={out_path}"
        )


if __name__ == "__main__":
    main()
