#!/usr/bin/env python3
"""Minimal standalone Snoopy inference runner (no training).

Runs Scorpion checkpoint inference directly on Snoopy-style query/target .npy matrices.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class Scorpion(nn.Module):
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
        return F.normalize(out, p=2, dim=1)


def setup_logging(level_name: str) -> None:
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def resolve_device(requested: str) -> str:
    req = (requested or "auto").lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return req


def _extract_proxy_tensor(state_dict: dict) -> torch.Tensor:
    if "proxy_sets" in state_dict:
        return state_dict["proxy_sets"]
    for key, value in state_dict.items():
        if key.endswith("proxy_sets"):
            return value
    raise KeyError("Checkpoint missing proxy_sets")


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[Scorpion, int]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    proxy = _extract_proxy_tensor(state_dict)
    if proxy.ndim != 3:
        raise ValueError(f"proxy_sets must be rank-3, got shape={tuple(proxy.shape)}")

    n_proxy_sets, n_elements, d = map(int, proxy.shape)
    model = Scorpion(n_proxy_sets, n_elements, d, device).to(device)
    missing, unexpected = model.load_state_dict({"proxy_sets": proxy}, strict=False)
    if missing:
        raise RuntimeError("Checkpoint missing required parameters: " + ", ".join(missing))
    if unexpected:
        LOGGER.warning("Ignoring unexpected parameters: %s", ", ".join(unexpected))
    model.eval()
    return model, d


def normalize_matrix(mat_obj, expected_dim: int) -> np.ndarray:
    mat = np.asarray(mat_obj, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.ndim != 2:
        raise ValueError(f"Expected rank-2 matrix, got shape={mat.shape}")
    if mat.shape[0] == 0:
        mat = np.zeros((1, expected_dim), dtype=np.float32)
    if mat.shape[1] != expected_dim:
        raise ValueError(f"Embedding dim mismatch: expected {expected_dim}, got {mat.shape[1]}")
    return mat


def load_npy_matrices(npy_path: Path, expected_dim: int) -> List[np.ndarray]:
    arr = np.load(npy_path, allow_pickle=True)
    mats: List[np.ndarray] = []
    for obj in arr:
        mats.append(normalize_matrix(obj, expected_dim))
    if not mats:
        raise ValueError(f"No matrices found in {npy_path}")
    return mats


def encode_batch(model: Scorpion, mats: Sequence[np.ndarray], device: torch.device) -> np.ndarray:
    concat = np.concatenate(mats, axis=0).astype(np.float32, copy=False)
    index = torch.tensor([m.shape[0] for m in mats], dtype=torch.long, device=device)
    data = torch.from_numpy(concat).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        emb = model(data, index)
    return emb.detach().cpu().numpy().astype(np.float32, copy=False)


def encode_matrices(
    model: Scorpion,
    mats: Sequence[np.ndarray],
    device: torch.device,
    batch_cols: int,
    max_cells_per_batch: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    pending: List[np.ndarray] = []
    pending_cells = 0

    for mat in mats:
        pending.append(mat)
        pending_cells += mat.shape[0]
        if len(pending) >= batch_cols or pending_cells >= max_cells_per_batch:
            chunks.append(encode_batch(model, pending, device))
            pending = []
            pending_cells = 0

    if pending:
        chunks.append(encode_batch(model, pending, device))

    return np.concatenate(chunks, axis=0)


def read_gt_rows(path: Optional[Path], index_offset: int) -> Optional[List[List[int]]]:
    if path is None:
        return None
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            vals: List[int] = []
            for cell in row:
                cell = (cell or "").strip()
                if not cell:
                    continue
                parts = cell.split() if any(ch.isspace() for ch in cell) else [cell]
                for p in parts:
                    try:
                        vals.append(int(p) - index_offset)
                    except ValueError:
                        continue
            rows.append(vals)
    return rows


def compute_recall_at_k(pred_indices: np.ndarray, gt_rows: List[List[int]], k: int) -> float:
    if not gt_rows:
        return 0.0
    n = min(len(gt_rows), pred_indices.shape[0])
    if n == 0:
        return 0.0
    hit = 0
    for i in range(n):
        gt = set(gt_rows[i][:k])
        if not gt:
            continue
        pred = set(pred_indices[i, :k].tolist())
        hit += len(gt.intersection(pred))
    return hit / float(n * k)


def run(args: argparse.Namespace) -> int:
    device = torch.device(resolve_device(args.device))

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    target_npy = Path(args.target_npy).expanduser().resolve()
    query_npy = Path(args.query_npy).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not target_npy.is_file():
        raise FileNotFoundError(f"target.npy not found: {target_npy}")
    if not query_npy.is_file():
        raise FileNotFoundError(f"query.npy not found: {query_npy}")

    model, dim = load_model(checkpoint_path, device)
    LOGGER.info("Loaded model (dim=%s): %s", dim, checkpoint_path)

    target_mats = load_npy_matrices(target_npy, expected_dim=dim)
    query_mats = load_npy_matrices(query_npy, expected_dim=dim)
    LOGGER.info("Loaded target columns: %s", len(target_mats))
    LOGGER.info("Loaded query columns: %s", len(query_mats))

    target_emb = encode_matrices(
        model=model,
        mats=target_mats,
        device=device,
        batch_cols=args.batch_cols,
        max_cells_per_batch=args.max_cells_per_batch,
    )
    query_emb = encode_matrices(
        model=model,
        mats=query_mats,
        device=device,
        batch_cols=args.batch_cols,
        max_cells_per_batch=args.max_cells_per_batch,
    )

    target_tensor = torch.from_numpy(target_emb).to(device=device, dtype=torch.float32)
    query_tensor = torch.from_numpy(query_emb).to(device=device, dtype=torch.float32)

    sim = torch.matmul(query_tensor, target_tensor.t())
    k = min(int(args.top_k), target_tensor.shape[0])
    top_scores, top_indices = torch.topk(sim, k=k, dim=1)

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_idx", "rank", "candidate_idx", "score"])
            for qi in range(top_indices.shape[0]):
                for r in range(k):
                    writer.writerow(
                        [
                            qi,
                            r + 1,
                            int(top_indices[qi, r].item()),
                            f"{float(top_scores[qi, r].item()):.6f}",
                        ]
                    )
        LOGGER.info("Wrote ranked results: %s", out_path)

    gt_rows = read_gt_rows(Path(args.index_csv).expanduser().resolve(), args.index_offset) if args.index_csv else None
    if gt_rows is not None:
        recall = compute_recall_at_k(top_indices.cpu().numpy(), gt_rows, k=min(25, k))
        LOGGER.info("Recall@%s (index.csv): %.4f", min(25, k), recall)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal standalone Snoopy inference on query.npy/target.npy")
    parser.add_argument("--checkpoint_path", required=True, help="Path to Snoopy checkpoint (.pth)")
    parser.add_argument("--target_npy", required=True, help="Path to target.npy")
    parser.add_argument("--query_npy", required=True, help="Path to query.npy")
    parser.add_argument("--out_csv", default="", help="Optional CSV output for ranked predictions")
    parser.add_argument("--index_csv", default="", help="Optional Snoopy index.csv for recall calculation")
    parser.add_argument("--index_offset", type=int, default=0, help="Offset subtracted from index.csv ids")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k retrieval candidates")
    parser.add_argument("--batch_cols", type=int, default=256, help="Max columns per encoding batch")
    parser.add_argument(
        "--max_cells_per_batch",
        type=int,
        default=200000,
        help="Max total cells per encoding batch",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


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
    raise SystemExit(main())
