#!/usr/bin/env python3
"""Verify joinability annotations with one LLM call per row."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "datasets/verified_joinable_200_samples.csv"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "datasets/verified_joinable_200_samples_llm.csv"
)
DEFAULT_BASE_URL = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"
DEFAULT_MAX_TOKENS = 3000
MODEL = os.getenv("PORTKEY_MODEL", "@vertexai/gemini-3.5-flash")
MAX_VALUES = 40
REASONING_EFFORT = "none"
SLEEP_SECONDS = 0.0
LIMIT = None
CONFIDENCE_COLUMN = "llm_confidence"
REASON_COLUMN = "llm_reason"


SYSTEM_PROMPT = """
You are a mix-of-experts verifier for joinability annotations.

Use three internal expert views before answering:
1. Equijoin expert: can the two columns be joined by direct value equality after light normalization?
2. Semantic expert: can values be cast, mapped, parsed, normalized, or linked by a stable domain relation?
3. Skeptic: is the proposed label unsupported, too broad, or ambiguous?

Return only JSON:
{"confidence": 0.0, "reason": "short reason"}

The confidence is the probability that the existing joinability label is correct.
0 means definitely incorrect; 1 means definitely correct.
The reason must be one short sentence.
For semantic joins, score high when there is a plausible reusable mapping, even if it is not a direct equijoin.
Do not penalize granularity differences, partial overlap, one-to-many or many-to-one mappings, subtype/supertype
relations, entity-to-attribute mappings, parsed components, aliases, abbreviations, units, dates, locations,
or sampled values that omit many true matches. Partial but systematic evidence is enough for high confidence.
Only score low when the relation is arbitrary, unsupported by the values, or would require unreliable outside knowledge.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    return parser.parse_args()


def load_values(text: str, max_values: int) -> list[str]:
    values = json.loads(text)
    if not isinstance(values, list):
        values = [values]
    return [str(value) for value in values[:max_values]]


def row_prompt(row: dict[str, str], max_values: int) -> str:
    payload = {
        "benchmark": row["benchmark"],
        "query_table": row["query_table"],
        "query_column": row["query_column"],
        "candidate_table": row["candidate_table"],
        "candidate_column": row["candidate_column"],
        "query_unique_values": load_values(row["query_unique_values"], max_values),
        "candidate_unique_values": load_values(row["candidate_unique_values"], max_values),
        "claimed_joinability": row["joinability"],
        "claimed_analysis": row["analysis"],
    }
    return json.dumps(payload, ensure_ascii=False)


def message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def parse_response(text: str) -> tuple[float, str]:
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.IGNORECASE | re.DOTALL)
    try:
        data = json.loads(body)
        value = data["confidence"]
        reason = str(data.get("reason", "")).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", body)
        if not match:
            raise ValueError(f"Could not parse confidence from response: {text!r}")
        value = match.group(0)
        reason = body.replace(match.group(0), "", 1).strip(" :-\n")
    return max(0.0, min(1.0, float(value))), reason


def make_client():
    from portkey_ai import Portkey

    api_key = os.getenv("PORTKEY_API_KEY")
    if not api_key:
        raise EnvironmentError("PORTKEY_API_KEY is not set.")

    kwargs = {
        "api_key": api_key,
        "base_url": DEFAULT_BASE_URL,
    }
    return Portkey(**kwargs)


def score_row(client: Any, model: str, row: dict[str, str]) -> tuple[float, str]:
    if model.startswith("@vertexai/"):
        request = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0,
            "reasoning_effort": REASONING_EFFORT,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row_prompt(row, MAX_VALUES)},
            ],
        }
    else:
        request = {
            "model": model,
            "reasoning_effort": REASONING_EFFORT,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row_prompt(row, MAX_VALUES)},
            ],
        }


    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    text = message_text(choice.message)
    if not text.strip():
        raise ValueError(
            f"LLM returned no content; finish_reason={getattr(choice, 'finish_reason', None)!r}. "
            "Set REASONING_EFFORT='none' or increase DEFAULT_MAX_TOKENS."
        )
    return parse_response(text)


def main() -> None:
    args = parse_args()
    client = make_client()

    with DEFAULT_INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if LIMIT is not None:
        rows = rows[:LIMIT]

    fieldnames = [
        name for name in rows[0] if name not in {CONFIDENCE_COLUMN, REASON_COLUMN}
    ] + [CONFIDENCE_COLUMN, REASON_COLUMN]
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with DEFAULT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            score, reason = score_row(client, args.model, row)
            row[CONFIDENCE_COLUMN] = f"{score:.4f}"
            row[REASON_COLUMN] = reason
            writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            print(f"[{idx}/{len(rows)}] {CONFIDENCE_COLUMN}={score:.4f}")
            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
