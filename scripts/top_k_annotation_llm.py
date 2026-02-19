#!/usr/bin/env python3
"""Batch LLM annotation for top-k WDC candidates.

This script consumes annotation candidates produced by:
  scripts/prune_wdc_top_k_for_annotation.py

Rows are grouped and sent to the LLM by query table.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import pandas as pd


INPUT_COLUMNS = [
    "query_table",
    "query_column",
    "candidate_table",
    "candidate_column",
    "query_unique_values",
    "candidate_unique_values",
]

OUTPUT_COLUMNS = [
    "query_table",
    "candidate_table",
    "joinability",
    "confidence",
    "analysis",
]

MAX_TOKENS = 65536
THINKING_BUDGET = 10000
MAX_RETRIES = 2
SLEEP_SECONDS = 0.0
OUTPUT_SUFFIX = "_annotation_llm.csv"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Batch LLM annotation for pruned WDC top-k candidates."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "wdc_top_k_annotation",
        help="Input CSV file or directory containing CSVs to scan for required headers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "wdc_top_k_annotation_llm",
        help="Directory for annotated output files.",
    )
    parser.add_argument(
        "--provider",
        default="portkey",
        choices=["portkey"],
        help="LLM provider backend.",
    )
    parser.add_argument(
        "--model",
        default="@vertexai/gemini-3-pro-preview",
        help="Model name passed to provider.",
    )
    parser.add_argument(
        "--query-tables",
        nargs="*",
        default=None,
        help="Optional subset of query tables to process.",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="Ignore existing checkpoint and start from scratch.",
    )
    return parser.parse_args()


def _csv_has_required_headers(path: Path) -> bool:
    try:
        df = pd.read_csv(path, nrows=0, dtype=str, keep_default_na=False)
    except Exception:
        return False
    return all(col in df.columns for col in INPUT_COLUMNS)


def discover_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    csv_files = sorted(p.resolve() for p in input_path.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise ValueError(f"No CSV files found under {input_path}")

    matched: List[Path] = []
    for csv_path in csv_files:
        if _csv_has_required_headers(csv_path):
            matched.append(csv_path)
        else:
            print(f"[SKIP] {csv_path.name}: missing required input headers")

    if not matched:
        raise ValueError(
            f"No CSV files with required headers found under {input_path}. "
            f"Required: {INPUT_COLUMNS}"
        )
    return matched


def _get_prompt(candidates: pd.DataFrame) -> Tuple[str, str]:
    system_prompt = """
You are an annotation assistant for identifying joinable attributes.

I will provide rows (CSV) with these fields:
query_table, query_column, candidate_table, candidate_column, query_unique_values, candidate_unique_values

Task:
For each row, read query_table/query_column and query_unique_values, then compare against candidate_unique_values (and candidate_table/candidate_column as context). Decide whether the two attributes are joinable.

Return a CSV with exactly this header:
query_table,candidate_table,joinability,confidence,analysis

Where:
- joinability in {semantic, equijoin, not joinable}
  - equijoin: values represent the same identifier/value space and can match directly (allowing minor normalization like case/whitespace/punctuation).
  - semantic: values are not usually identical but have a clear relationship that enables a meaningful join via mapping or transformation (e.g., city->state, county->state, code->name).
  - not joinable: no stable direct match or meaningful semantic mapping is supported by the values.
- confidence: a number in [0, 1]
- analysis: one short, precise sentence explaining the key evidence (e.g., overlap, same entity type, hierarchy, incompatible domains).

Output requirements:
- Return only CSV text (no markdown, no explanations).
- Produce one output row per input row, in the same order.
- Be conservative: choose "not joinable" if evidence is weak or ambiguous.
""".strip()

    payload = candidates[INPUT_COLUMNS].to_csv(index=False)
    prompt = f"Candidates CSV:\n{payload}"
    return system_prompt, prompt


def _extract_chunk_text(chunk: object) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""

    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""

    content = None
    if isinstance(delta, dict):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _ask_llm(
    user_prompt: str,
    system_message: str,
    provider: str,
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> Iterator[str]:
    if provider != "portkey":
        raise ValueError(f"Unsupported provider: {provider}")

    try:
        from portkey_ai import Portkey
    except ImportError as exc:
        raise ImportError(
            "portkey_ai is required for provider=portkey. Install it with `pip install portkey-ai`."
        ) from exc

    portkey = Portkey(
        base_url = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
        api_key=os.getenv("PORTKEY_API_KEY"),
        strict_open_ai_compliance=False,
        metadata={"_user": "yfw215"},
    )

    response = portkey.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        thinking={
            "type": "enabled",
            "budget_tokens": thinking_budget,
        },
        stream=True,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    for chunk in response:
        text = _extract_chunk_text(chunk)
        if text:
            yield text


def _strip_code_fences(text: str) -> str:
    body = text.strip()
    if "```" not in body:
        return body

    matches = re.findall(r"```(?:csv)?\s*(.*?)```", body, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[0].strip()
    return body.replace("```", "").strip()


def _extract_csv_block(text: str) -> str:
    cleaned = _strip_code_fences(text)
    lines = cleaned.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        lowered = line.strip().lower().replace(" ", "")
        if "query_table,candidate_table,joinability,confidence,analysis" in lowered:
            header_idx = idx
            break
    if header_idx is None:
        return cleaned.strip()
    return "\n".join(lines[header_idx:]).strip()


def _parse_result_csv(raw_response: str) -> pd.DataFrame:
    """Convert LLM CSV text into normalized annotation rows."""
    csv_text = _extract_csv_block(raw_response)
    if not csv_text:
        raise ValueError("LLM response is empty.")

    try:
        parsed = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    parsed.columns = [str(c).strip().lower() for c in parsed.columns]
    missing = [c for c in OUTPUT_COLUMNS if c not in parsed.columns]
    if missing:
        raise ValueError(f"Missing expected output columns: {missing}")

    out = parsed[OUTPUT_COLUMNS].copy()
    for col in out.columns:
        out[col] = out[col].astype(str).str.strip()
    out["joinability"] = out["joinability"].str.lower()

    confidence = pd.to_numeric(out["confidence"], errors="coerce")
    confidence = confidence.clip(lower=0.0, upper=1.0)
    out["confidence"] = confidence.map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def _checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.checkpoint.csv")


def _append_checkpoint_rows(path: Path, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _validate_input_columns(df: pd.DataFrame, source: Path) -> None:
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def annotate_file(
    input_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    query_tables: Optional[Sequence[str]],
    overwrite_checkpoint: bool,
) -> None:
    source_df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    _validate_input_columns(source_df, input_path)
    source_df = source_df.copy()
    source_df["_row_id"] = source_df.index.astype(int)

    if query_tables:
        allow = set(query_tables)
        source_df = source_df[source_df["query_table"].isin(allow)].copy()

    if source_df.empty:
        print(f"[SKIP] No rows to process in {input_path}")
        return

    checkpoint_path = _checkpoint_path_for(output_path)
    if overwrite_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    done_df = _load_checkpoint(checkpoint_path)
    done_ids = set()
    if not done_df.empty and "_row_id" in done_df.columns:
        done_ids = set(pd.to_numeric(done_df["_row_id"], errors="coerce").dropna().astype(int).tolist())

    pending_df = source_df[~source_df["_row_id"].isin(done_ids)].copy()
    if pending_df.empty:
        print(f"[DONE] All rows already annotated for {input_path.name}")
    else:
        grouped = list(pending_df.groupby("query_table", sort=True, dropna=False))

        total_batches = len(grouped)
        for batch_idx, (query_table, batch_df) in enumerate(grouped, start=1):
            batch_payload = batch_df[INPUT_COLUMNS].reset_index(drop=True)
            system_prompt, user_prompt = _get_prompt(batch_payload)

            parsed: Optional[pd.DataFrame] = None
            last_err: Optional[Exception] = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    raw_response = "".join(
                        _ask_llm(
                            user_prompt=user_prompt,
                            system_message=system_prompt,
                            provider=provider,
                            model=model,
                            max_tokens=MAX_TOKENS,
                            thinking_budget=THINKING_BUDGET,
                        )
                    )
                    parsed = _parse_result_csv(raw_response)
                    if len(parsed) != len(batch_payload):
                        raise ValueError(
                            f"Row count mismatch for query_table={query_table}: "
                            f"input={len(batch_payload)} output={len(parsed)}"
                        )
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt >= MAX_RETRIES:
                        raise RuntimeError(
                            f"Failed batch query_table={query_table} after {MAX_RETRIES + 1} attempts"
                        ) from last_err
                    print(
                        f"[WARN] Retry {attempt + 1}/{MAX_RETRIES} for query_table={query_table}: {exc}"
                    )
                    time.sleep(max(0.0, SLEEP_SECONDS))

            assert parsed is not None
            out_batch = batch_df.copy().reset_index(drop=True)
            out_batch["joinability"] = parsed["joinability"]
            out_batch["confidence"] = parsed["confidence"]
            out_batch["analysis"] = parsed["analysis"]
            _append_checkpoint_rows(checkpoint_path, out_batch)

            print(
                f"[{input_path.name}] batch={batch_idx}/{total_batches} "
                f"query_table={query_table} rows={len(batch_df)} done={len(done_ids) + len(out_batch)}"
            )
            done_ids.update(out_batch["_row_id"].astype(int).tolist())

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    final_df = _load_checkpoint(checkpoint_path)
    if final_df.empty:
        print(f"[SKIP] No annotated rows produced for {input_path.name}")
        return

    final_df["_row_id"] = pd.to_numeric(final_df["_row_id"], errors="coerce")
    final_df = final_df.dropna(subset=["_row_id"]).copy()
    final_df["_row_id"] = final_df["_row_id"].astype(int)

    final_df = final_df.sort_values("_row_id", kind="stable")
    final_df = final_df.drop_duplicates(subset=["_row_id"], keep="last")

    source_cols = [c for c in source_df.columns if c != "_row_id"]
    keep_cols = source_cols + ["joinability", "confidence", "analysis"]
    keep_cols = [c for c in keep_cols if c in final_df.columns]
    final_df = final_df[keep_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[DONE] output={output_path} rows={len(final_df)} checkpoint={checkpoint_path}")


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.provider == "portkey" and not os.getenv("PORTKEY_API_KEY"):
        raise EnvironmentError("PORTKEY_API_KEY is not set.")

    input_files = discover_input_files(input_path)
    for src in input_files:
        stem = src.stem
        if stem.endswith("_annotation_candidates"):
            stem = stem[: -len("_annotation_candidates")]
        output_name = stem + OUTPUT_SUFFIX
        dst = output_dir / output_name
        annotate_file(
            input_path=src,
            output_path=dst,
            provider=args.provider,
            model=args.model,
            query_tables=args.query_tables,
            overwrite_checkpoint=args.overwrite_checkpoint,
        )


if __name__ == "__main__":
    main()
