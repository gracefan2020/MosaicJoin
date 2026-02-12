#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import sys

import pandas as pd
import torch
import nltk
from sentence_transformers import SentenceTransformer, util

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# FROM https://github.com/mutong184/deepjoin
def analyze_column_values(df, column_name):

    # 获取指定列的所有不同的列值数据和它们的频率
    value_counts = df[column_name].astype(str).value_counts()

    # 按照频率由高到低对列值进行排序
    sorted_values = value_counts.index.tolist()

    n = len(sorted_values)
    # 以逗号分隔列值
    col = ', '.join(sorted_values)

    # 统计列值的最大、最小和平均长度
    lengths = [len(str(value)) for value in sorted_values]
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    tokens = f"{column_name} contains {str(n)} values ({str(max_len)}, {str(min_len)}, {str(avg_len)}): {col}"
    # 返回结果

    tokens = nltk.word_tokenize(tokens)
    truncated_tokens = tokens[:512]
    truncated_sentence = ' '.join(truncated_tokens)
    return truncated_sentence


def _has_csv_files(dir_path):
    try:
        return any(
            fname.endswith(".csv")
            and os.path.isfile(os.path.join(dir_path, fname))
            for fname in os.listdir(dir_path)
        )
    except (FileNotFoundError, NotADirectoryError):
        return False


def resolve_datalake_dir(datalake_dir):
    nested = os.path.join(datalake_dir, "datalake")
    if os.path.isdir(nested) and _has_csv_files(nested):
        return nested
    if _has_csv_files(datalake_dir):
        return datalake_dir
    return datalake_dir


def resolve_query_dir(query_dir, datalake_dir):
    if query_dir:
        if os.path.isfile(query_dir):
            return query_dir
        nested = os.path.join(query_dir, "queries")
        if os.path.isdir(nested) and _has_csv_files(nested):
            return nested
        if _has_csv_files(query_dir):
            return query_dir
        return query_dir

    parent = os.path.dirname(datalake_dir)
    candidate = os.path.join(parent, "queries")
    if os.path.isdir(candidate) and _has_csv_files(candidate):
        return candidate
    if _has_csv_files(datalake_dir):
        return datalake_dir
    return None


def _table_title_from_filename(fname):
    base = os.path.splitext(fname)[0]
    return base.replace("_", " ").strip()


def _load_query_pairs(query_file):
    pairs = []
    with open(query_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pairs
        header_lower = [h.strip().lower() for h in header]
        candidates = [
            ("target_ds", "target_attr"),
            ("query_table", "query_column"),
            ("table", "column"),
            ("table_name", "column_name"),
            ("source_table", "source_column"),
            ("target_table", "target_column"),
        ]
        indices = None
        for table_key, col_key in candidates:
            if table_key in header_lower and col_key in header_lower:
                indices = (header_lower.index(table_key), header_lower.index(col_key))
                break

        if indices is None and "left_table" in header_lower:
            t_idx = header_lower.index("left_table")
            for row in reader:
                if len(row) <= t_idx:
                    continue
                table = row[t_idx].strip()
                if table:
                    pairs.append((table, "title"))
            return pairs

        if indices is None:
            if len(header) >= 2:
                pairs.append((header[0].strip(), header[1].strip()))
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0].strip(), row[1].strip()))
            return pairs

        t_idx, c_idx = indices
        for row in reader:
            if len(row) <= max(t_idx, c_idx):
                continue
            table = row[t_idx].strip()
            column = row[c_idx].strip()
            if table and column:
                pairs.append((table, column))
    return pairs


def _resolve_table_path(datalake_dir, table_ref):
    if not table_ref:
        return None
    table_ref = table_ref.strip()
    ref_path = os.path.abspath(table_ref) if os.path.isabs(table_ref) else table_ref
    if os.path.isfile(ref_path):
        return ref_path
    if not os.path.isabs(table_ref):
        candidate = os.path.join(datalake_dir, table_ref)
        if os.path.isfile(candidate):
            return candidate
        if not table_ref.lower().endswith(".csv"):
            candidate = os.path.join(datalake_dir, f"{table_ref}.csv")
            if os.path.isfile(candidate):
                return candidate
    return None


def collect_query_columns_from_file(query_file, datalake_dir, use_table_name=False):
    rows = []
    texts = []
    pairs = _load_query_pairs(query_file)
    for table_ref, column_ref in pairs:
        path = _resolve_table_path(datalake_dir, table_ref)
        if not path:
            print(f"Query table not found: {table_ref}")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            print(f"Failed to read query table: {path}")
            continue
        target_col = None
        for col in df.columns:
            if col.strip().lower() == column_ref.strip().lower():
                target_col = col
                break
        if not target_col:
            print(f"Query column '{column_ref}' not found in {path}")
            continue
        try:
            text = analyze_column_values(df, target_col)
        except Exception:
            continue
        table_title = _table_title_from_filename(os.path.basename(path)) if use_table_name else ""
        if table_title:
            text = f"{table_title}. {text}"
        rows.append((os.path.basename(path), target_col))
        texts.append(text)
    return rows, texts


def iter_table_columns(table_dir, column_name="title", use_table_name=False):
    for fname in sorted(os.listdir(table_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(table_dir, fname)
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        table_title = _table_title_from_filename(fname) if use_table_name else ""
        for col in df.columns:
            if column_name and col.strip().lower() != column_name.lower():
                continue
            try:
                text = analyze_column_values(df, col)
            except Exception:
                continue
            if table_title:
                text = f"{table_title}. {text}"
            yield fname, col, text


def collect_table_columns(table_dir, column_name="title", use_table_name=False):
    rows = []
    texts = []
    for table_name, col_name, text in iter_table_columns(
        table_dir, column_name=column_name, use_table_name=use_table_name
    ):
        rows.append((table_name, col_name))
        texts.append(text)
    return rows, texts


def load_index(index_path):
    with open(index_path, "rb") as f:
        payload = pickle.load(f)
    items = payload.get("items", [])
    embeddings = payload.get("embeddings")
    if embeddings is None:
        raise ValueError(f"Index file missing embeddings: {index_path}")
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    meta = payload.get("meta", {})
    return items, tensor, meta


def save_index(index_path, items, embeddings, meta):
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    payload = {
        "items": items,
        "embeddings": embeddings.detach().cpu().numpy(),
        "meta": meta,
    }
    with open(index_path, "wb") as f:
        pickle.dump(payload, f)


def semantic_search(query_embeddings, corpus_embeddings, top_k):
    if top_k <= 0:
        results = []
        for q_idx in range(query_embeddings.shape[0]):
            scores = util.cos_sim(query_embeddings[q_idx], corpus_embeddings).squeeze(0)
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            hits = [
                {"corpus_id": int(idx), "score": float(score)}
                for idx, score in zip(sorted_indices, sorted_scores)
            ]
            results.append(hits)
        return results

    try:
        return util.semantic_search(
            query_embeddings, corpus_embeddings, top_k=top_k, score_function=util.cos_sim
        )
    except TypeError:
        return util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalake_dir", required=True, help="Directory containing datalake CSVs")
    parser.add_argument(
        "--query_dir",
        help=(
            "Directory containing query CSVs or a query columns CSV file "
            "(defaults to sibling queries/)."
        ),
    )
    parser.add_argument("--out_csv", help="Output CSV with predicted scores")
    parser.add_argument(
        "--model_name",
        default="baselines/deepjoin/all-mpnet-base-v2",
        help=(
            "Model name/path (defaults to baselines/deepjoin/all-mpnet-base-v2, "
            "same as deepjoin_infer.py)"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--column_name",
        default="title",
        help="Column name to analyze (default: title). Use '*' or 'all' for all columns.",
    )
    parser.add_argument(
        "--use_table_name",
        action="store_true",
        help="Prefix each column text with the table name (filename without extension).",
    )
    parser.add_argument(
        "--index_path",
        help="Optional index file to save/load datalake embeddings (pickle).",
    )
    parser.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Rebuild index even if --index_path already exists.",
    )
    parser.add_argument(
        "--build_index_only",
        action="store_true",
        help="Only build/save the datalake index, then exit.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of candidates per query column (<=0 to return all).",
    )
    parser.add_argument(
        "--include_same_table",
        action="store_true",
        help="Include candidates from the same table as the query.",
    )
    parser.add_argument(
        "--with_header",
        action="store_true",
        help="Write a CSV header row.",
    )
    args = parser.parse_args()

    datalake_dir = resolve_datalake_dir(args.datalake_dir)
    if datalake_dir != args.datalake_dir:
        print(f"Using datalake dir: {datalake_dir}")

    query_dir = resolve_query_dir(args.query_dir, datalake_dir)
    if args.build_index_only:
        query_dir = None
    if query_dir and args.query_dir and not os.path.exists(query_dir):
        print(f"Query path not found: {query_dir}")
        return 1
    if not args.build_index_only and not query_dir:
        print("No query directory found. Provide --query_dir or ensure a sibling queries/ exists.")
        return 1
    if query_dir and args.query_dir and query_dir != args.query_dir:
        print(f"Using query dir: {query_dir}")

    column_name = args.column_name
    if not column_name or column_name.strip().lower() in {"*", "all"}:
        column_name = None

    if not os.path.exists(args.model_name):
        print(
            f"Model path not found locally: {args.model_name}. "
            "Provide a local path to model weights (no online fetch)."
        )
        return 1
    try:
        model = SentenceTransformer(args.model_name, local_files_only=True)
    except TypeError:
        model = SentenceTransformer(args.model_name)

    items = None
    embeddings = None
    if args.index_path and os.path.isfile(args.index_path) and not args.rebuild_index:
        items, embeddings, meta = load_index(args.index_path)
        if not items:
            print(f"No items found in index {args.index_path}; rebuilding.")
            items = None
            embeddings = None
    if items is None:
        items, texts = collect_table_columns(
            datalake_dir, column_name=column_name, use_table_name=args.use_table_name
        )
        if not items and column_name:
            print(
                f"No columns matched '{column_name}' in {datalake_dir}; falling back to all columns."
            )
            items, texts = collect_table_columns(
                datalake_dir, column_name=None, use_table_name=args.use_table_name
            )
        if not items:
            print(f"No columns found in {datalake_dir}.")
            return 1
        embeddings = model.encode(texts, convert_to_tensor=True, batch_size=args.batch_size)
        if args.index_path:
            meta = {
                "datalake_dir": datalake_dir,
                "column_name": column_name,
                "use_table_name": args.use_table_name,
                "model_name": args.model_name,
            }
            save_index(args.index_path, items, embeddings, meta)

    if args.build_index_only:
        return 0

    if not args.out_csv:
        print("Missing --out_csv for query inference.")
        return 1

    if query_dir and os.path.isfile(query_dir):
        query_items, query_texts = collect_query_columns_from_file(
            query_dir, datalake_dir, use_table_name=args.use_table_name
        )
    else:
        query_items, query_texts = collect_table_columns(
            query_dir, column_name=column_name, use_table_name=args.use_table_name
        )
        if not query_items and column_name:
            print(
                f"No query columns matched '{column_name}' in {query_dir}; falling back to all columns."
            )
            query_items, query_texts = collect_table_columns(
                query_dir, column_name=None, use_table_name=args.use_table_name
            )
    if not query_items:
        print(f"No query columns found in {query_dir}.")
        return 1

    query_embeddings = model.encode(
        query_texts, convert_to_tensor=True, batch_size=args.batch_size
    )
    if embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(embeddings.device)

    hits = semantic_search(query_embeddings, embeddings, args.top_k)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        if args.with_header:
            writer.writerow(
                [
                    "query_table",
                    "query_column",
                    "candidate_table",
                    "candidate_column",
                    "similarity_score",
                ]
            )
        for q_idx, hit_list in enumerate(hits):
            q_table, q_col = query_items[q_idx]
            if hit_list:
                hit_list = sorted(hit_list, key=lambda h: h["score"], reverse=True)
            kept = 0
            for hit in hit_list:
                cand_idx = hit["corpus_id"]
                score = hit["score"]
                cand_table, cand_col = items[cand_idx]
                if not args.include_same_table and q_table == cand_table:
                    continue
                writer.writerow([q_table, q_col, cand_table, cand_col, f"{score:.6f}"])
                kept += 1
                if args.top_k > 0 and kept >= args.top_k:
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
