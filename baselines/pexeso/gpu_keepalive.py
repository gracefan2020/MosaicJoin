#!/usr/bin/env python3
"""
Lightweight GPU keep-alive worker.

Runs tiny CUDA compute bursts at a fixed interval to prevent prolonged 0% GPU
utilization windows in CPU-bound workloads.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import torch

RUNNING = True


def _stop_handler(signum, frame) -> None:  # noqa: ARG001
    global RUNNING
    RUNNING = False


def _resolve_device(requested: str) -> torch.device:
    req = (requested or "auto").lower()
    if req == "auto":
        req = "cuda"
    return torch.device(req)


def _reserve_gpu_memory(
    device: torch.device,
    target_mem_frac: float,
    reserve_mem_gb: float,
) -> torch.Tensor | None:
    free_b, total_b = torch.cuda.mem_get_info(device)
    total_b = int(total_b)
    free_b = int(free_b)

    target_b = int(max(0.0, min(1.0, target_mem_frac)) * total_b)
    reserve_b = int(max(0.0, reserve_mem_gb) * (1024 ** 3))
    alloc_b = min(target_b, max(0, free_b - reserve_b))
    if alloc_b <= 0:
        print(
            "[gpu-keepalive] memory reserve skipped "
            f"(free={free_b / (1024 ** 3):.2f}GB total={total_b / (1024 ** 3):.2f}GB)",
            flush=True,
        )
        return None

    # Allocate bytes directly so reservation size is predictable.
    step_b = 256 * 1024 * 1024  # back off in 256MB chunks if allocation fails.
    while alloc_b > 0:
        try:
            buf = torch.empty((alloc_b,), device=device, dtype=torch.uint8)
            actual_frac = float(buf.numel()) / float(total_b)
            print(
                "[gpu-keepalive] reserved "
                f"{buf.numel() / (1024 ** 3):.2f}GB "
                f"({actual_frac * 100:.1f}% of total {total_b / (1024 ** 3):.2f}GB)",
                flush=True,
            )
            return buf
        except RuntimeError:
            alloc_b -= step_b
    print("[gpu-keepalive] unable to reserve additional GPU memory", flush=True)
    return None


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print("[gpu-keepalive] CUDA unavailable, exiting.", flush=True)
        return 0

    device = _resolve_device(args.device)
    if device.type != "cuda":
        print(f"[gpu-keepalive] device={device} is not CUDA, exiting.", flush=True)
        return 0

    print(
        "[gpu-keepalive] start "
        f"device={device} interval_s={args.interval_sec} "
        f"work_ms={args.work_ms} matrix={args.matrix_size} "
        f"target_mem_frac={args.target_mem_frac}",
        flush=True,
    )

    mem_hold = _reserve_gpu_memory(
        device=device,
        target_mem_frac=args.target_mem_frac,
        reserve_mem_gb=args.reserve_mem_gb,
    )

    # Keep memory footprint small.
    a = torch.randn(
        (args.matrix_size, args.matrix_size),
        device=device,
        dtype=torch.float16,
    )
    b = torch.randn(
        (args.matrix_size, args.matrix_size),
        device=device,
        dtype=torch.float16,
    )

    while RUNNING:
        loop_start = time.perf_counter()
        work_deadline = loop_start + (args.work_ms / 1000.0)
        while RUNNING and time.perf_counter() < work_deadline:
            c = torch.matmul(a, b)
            c = torch.relu_(c)
            # Touch one value to prevent dead-code elimination assumptions.
            _ = c[0, 0].item()

        torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - loop_start
        sleep_for = max(0.0, args.interval_sec - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    # Keep reference alive until shutdown.
    _ = mem_hold
    print("[gpu-keepalive] stopped", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight CUDA keep-alive compute.")
    parser.add_argument("--device", type=str, default="auto", help="CUDA device (default: auto)")
    parser.add_argument(
        "--interval_sec",
        type=float,
        default=5.0,
        help="Seconds between keep-alive loop starts (default: 5).",
    )
    parser.add_argument(
        "--work_ms",
        type=float,
        default=4500.0,
        help="Target GPU work duration in each loop in milliseconds (default: 4500).",
    )
    parser.add_argument(
        "--matrix_size",
        type=int,
        default=4096,
        help="Square matrix size for keep-alive matmul workload (default: 4096).",
    )
    parser.add_argument(
        "--target_mem_frac",
        type=float,
        default=0.55,
        help="Target fraction of total GPU memory to reserve (default: 0.55).",
    )
    parser.add_argument(
        "--reserve_mem_gb",
        type=float,
        default=8.0,
        help="Host job memory safety reserve left unallocated on GPU in GB (default: 8).",
    )
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGINT, _stop_handler)
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[gpu-keepalive] error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
