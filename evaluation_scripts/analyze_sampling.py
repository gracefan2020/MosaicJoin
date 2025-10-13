#!/usr/bin/env python3
"""
Analyze how a SentenceTransformer model tokenizes and embeds values for a given
table.column, and which values are selected by the k-closest-to-origin rule.

Outputs a CSV with columns:
  - value: original string value
  - token_count: number of wordpiece tokens
  - tokens: token list (space-joined)
  - embedding_norm: L2 norm of the embedding vector
  - selected_k: 1 if chosen among k closest-to-origin, else 0
  - rank_by_norm: 1-based rank after sorting by norm asc (1 means closest)

Usage example:
  python scripts/analyze_sampling.py \
    --datalake-dir datasets/freyja-semantic-join/datalake \
    --table pte_has_property --column Arg0 \
    --embedder all-MiniLM-L6-v2 --k 16 \
    --limit 10000 \
    --out-csv analyses-disagreements/pte_has_property_Arg0_sampling.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    raise SystemExit(
        "SentenceTransformers is required. Install with: pip install sentence-transformers"
    ) from exc


def load_column_values(datalake_dir: str, table: str, column: str, limit: int) -> List[str]:
    csv_path = os.path.join(datalake_dir, f"{table}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    series = pd.read_csv(csv_path, usecols=[column], dtype=str)[column]
    # Normalize lightly: strip, drop empty, keep order, unique
    values: List[str] = []
    seen = set()
    for v in series.dropna().astype(str):
        sv = v.strip()
        if not sv:
            continue
        if sv in seen:
            continue
        seen.add(sv)
        values.append(sv)
        if len(values) >= limit:
            break
    return values


def tokenize_values(model: SentenceTransformer, values: List[str]) -> Tuple[List[List[str]], List[int]]:
    # Access underlying HF tokenizer
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        # Fallback: treat the whole value as a single token
        tokens = [[v] for v in values]
        counts = [1 for _ in values]
        return tokens, counts
    tokens: List[List[str]] = []
    counts: List[int] = []
    for v in values:
        enc = tok(v, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=False)
        ids = enc["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        tokens.append(toks)
        counts.append(len(toks))
    return tokens, counts


def embed_values(model: SentenceTransformer, values: List[str], batch_size: int = 64) -> np.ndarray:
    # normalize_embeddings=True gives unit vectors for cosine similarity usage
    embeddings = model.encode(
        values,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    # If normalized, all norms ~1 unless something degenerate; for analysis,
    # optionally compute non-normalized embeddings by re-encoding without normalization
    return embeddings


def compute_k_closest(norms: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = max(0, min(k, len(norms)))
    if k == 0:
        return np.array([], dtype=int), norms
    # argpartition for efficiency; then sort the first k by norm asc for rank
    idx = np.argpartition(norms, k - 1)[:k]
    return idx, norms


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze tokenization/embeddings and k-closest sampling")
    ap.add_argument("--datalake-dir", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    print("Loading values...")
    values = load_column_values(args.datalake_dir, args.table, args.column, args.limit)
    if not values:
        raise SystemExit("No values loaded. Check table/column names.")
    print(f"Loaded {len(values)} unique values")

    print(f"Loading model: {args.embedder}")
    model = SentenceTransformer(args.embedder)

    print("Tokenizing values...")
    tokens, counts = tokenize_values(model, values)

    print("Embedding values...")
    embeddings = embed_values(model, values, batch_size=args.batch_size)
    norms = np.linalg.norm(embeddings, axis=1)

    print("Selecting k closest to origin by L2 norm...")
    selected_idx, _ = compute_k_closest(norms, args.k)
    selected_set = set(map(int, selected_idx))

    # Rank by norm ascending for extra interpretability
    order = np.argsort(norms)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)

    print(f"Writing CSV: {args.out_csv}")
    rows = []
    for i, v in enumerate(values):
        rows.append({
            "value": v,
            "token_count": int(counts[i]),
            "tokens": " ".join(tokens[i]),
            "embedding_norm": float(norms[i]),
            "selected_k": 1 if i in selected_set else 0,
            "rank_by_norm": int(rank[i]),
        })
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    # Small summary
    selected_preview = [values[i] for i in order[: min(args.k, len(values))]]
    print("Preview of k closest values:")
    for j, v in enumerate(selected_preview, start=1):
        print(f"  {j:2d}. {v}")


if __name__ == "__main__":
    main()


