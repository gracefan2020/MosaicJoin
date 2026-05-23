#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time

TOPJOIN_ROOT = os.path.join(os.path.dirname(__file__), "ContextAwareJoin")
TOPJOIN_SRC = os.path.join(TOPJOIN_ROOT, "src")
BASELINES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASELINES_ROOT not in sys.path:
    sys.path.insert(0, BASELINES_ROOT)
if TOPJOIN_SRC not in sys.path:
    sys.path.insert(0, TOPJOIN_SRC)

from myutils.logging_util import setup_logger
from resource_monitor import (
    ResourceMonitor,
    add_resource_monitor_args,
    default_resource_log_path,
    log_resource_summary,
)
from topjoin import create_index
from topjoin.query_helper import Joinable_QueryHelper


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "topjoin_config.json")


def strip_csv(name):
    name = os.path.basename(str(name).strip())
    return os.path.splitext(name)[0] if name.lower().endswith(".csv") else name


def resolve_datalake_dir(path):
    nested = os.path.join(path, "datalake")
    return nested if os.path.isdir(nested) else path


def load_query_columns(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {name.lower(): name for name in reader.fieldnames or []}
        table_key = fields.get("target_ds") or fields.get("source_table") or fields.get("table")
        column_key = fields.get("target_attr") or fields.get("source_column") or fields.get("column")
        if not table_key or not column_key:
            raise ValueError(f"Unsupported query column schema: {path}")
        return [(strip_csv(row[table_key]), row[column_key].strip()) for row in reader]


def result_column(table, column_id):
    prefix = f"{table}."
    return column_id[len(prefix):] if column_id.startswith(prefix) else column_id


def load_topjoin_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_model(path):
    if not os.path.exists(path):
        return
    weight_path = os.path.join(path, "pytorch_model.bin")
    if not os.path.isfile(weight_path):
        return
    with open(weight_path, "rb") as f:
        header = f.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError(
            f"{weight_path} is a Git LFS pointer, not model weights. "
            "Install git-lfs and run `git lfs pull` in the MosaicJoin repo, "
            "or pass MODEL_PATH to a complete local SentenceTransformer model."
        )


def write_results(config, queries, out_csv, include_same_table):
    query_helper = Joinable_QueryHelper(config)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
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
        for query_table, query_column in queries:
            response = query_helper.query_joinability(query_table, query_column)
            if not response:
                continue
            ranked_columns, scores = response
            for column_id, score in zip(ranked_columns, scores):
                candidate_table = ".".join(column_id.split(".")[:-1])
                candidate_column = result_column(candidate_table, column_id)
                if not include_same_table and (query_table, query_column) == (
                    candidate_table,
                    candidate_column,
                ):
                    continue
                writer.writerow(
                    [
                        query_table,
                        query_column,
                        candidate_table,
                        candidate_column,
                        -float(score),
                    ]
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalake_dir", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="topjoin")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--candidate_k", type=int, default=100)
    parser.add_argument("--include_same_table", action="store_true")
    parser.add_argument("--log_dir", default=os.path.join(os.path.dirname(__file__), "results"))
    add_resource_monitor_args(parser)
    args = parser.parse_args()
    validate_model(args.model)

    config = {
        "method": "topjoin",
        "benchmark": args.benchmark,
        "datalake_dir": resolve_datalake_dir(args.datalake_dir),
        "file_format": ".csv",
        "metadata_dir": None,
        "metadata_suffix": None,
        "model": args.model,
        "embedding_indexer": "NN",
        "minhash_indexer": "LSH_FOREST",
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
    }
    config.update(load_topjoin_config(args.config))

    setup_logger(
        f"topjoin_{args.benchmark}",
        variant=config,
        exp_id=os.getpid(),
        base_log_dir=args.log_dir,
        git_infos=[],
    )
    config["index_path"] = create_index(config)
    resource_log_csv = default_resource_log_path(args.out_csv, args.resource_log_csv)
    with ResourceMonitor(resource_log_csv, args.resource_sample_interval) as resources:
        online_start = time.perf_counter()
        write_results(config, load_query_columns(args.query_file), args.out_csv, args.include_same_table)
        online_seconds = time.perf_counter() - online_start
    print(f"[TIMING] online_query_seconds={online_seconds:.3f}")
    log_resource_summary(resources.summary(), print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
