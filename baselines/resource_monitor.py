from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Set, Tuple


class ResourceMonitor:
    def __init__(self, csv_path: Optional[str], interval: float = 1.0):
        self.csv_path = str(csv_path) if csv_path else ""
        self.interval = max(0.1, float(interval))
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rows = []
        self._last_cpu: Optional[Tuple[float, float]] = None
        self._start = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_ram_gb = 0.0
        self.peak_gpu_memory_gb = 0.0

    def __enter__(self) -> "ResourceMonitor":
        self._start = time.time()
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        self.write_csv()

    def start(self) -> "ResourceMonitor":
        return self.__enter__()

    def stop(self) -> None:
        self.__exit__(None, None, None)

    def summary(self) -> Dict[str, object]:
        return {
            "peak_cpu_percent": self.peak_cpu_percent,
            "peak_ram_gb": self.peak_ram_gb,
            "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
            "samples": len(self._rows),
            "csv_path": self.csv_path,
        }

    def sample(self) -> None:
        now = time.time()
        pids, cpu_seconds, rss_bytes = _process_tree_usage(self.pid)
        cpu_percent = self._cpu_percent(now, cpu_seconds)
        ram_gb = rss_bytes / float(1024 ** 3)
        gpu_gb = _gpu_memory_gb(pids)

        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
        self.peak_ram_gb = max(self.peak_ram_gb, ram_gb)
        self.peak_gpu_memory_gb = max(self.peak_gpu_memory_gb, gpu_gb)
        self._rows.append(
            {
                "timestamp": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                "elapsed_seconds": f"{now - self._start:.3f}",
                "cpu_percent": f"{cpu_percent:.3f}",
                "ram_gb": f"{ram_gb:.6f}",
                "gpu_memory_gb": f"{gpu_gb:.6f}",
                "pids": " ".join(str(pid) for pid in sorted(pids)),
            }
        )

    def write_csv(self) -> None:
        if not self.csv_path or not self._rows:
            return
        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0]))
            writer.writeheader()
            writer.writerows(self._rows)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def _cpu_percent(self, now: float, cpu_seconds: float) -> float:
        previous = self._last_cpu
        self._last_cpu = (now, cpu_seconds)
        if previous is None:
            return 0.0
        prev_time, prev_cpu = previous
        elapsed = now - prev_time
        if elapsed <= 0:
            return 0.0
        return max(0.0, (cpu_seconds - prev_cpu) / elapsed * 100.0)


def add_resource_monitor_args(parser) -> None:
    parser.add_argument(
        "--resource_log_csv",
        default="",
        help="Timestamped online resource samples CSV. Defaults to <out_csv>.resources.csv.",
    )
    parser.add_argument(
        "--resource_sample_interval",
        type=float,
        default=1.0,
        help="Seconds between online resource samples.",
    )


def default_resource_log_path(out_csv: str, requested: str = "") -> str:
    if requested:
        return requested
    if not out_csv:
        return ""
    path = Path(out_csv).expanduser()
    suffix = path.suffix + ".resources.csv" if path.suffix else ".resources.csv"
    return str(path.with_suffix(suffix))


def log_resource_summary(summary: Dict[str, object], emit: Callable[[str], None]) -> None:
    emit(
        "[RESOURCE] "
        f"peak_cpu_percent={summary['peak_cpu_percent']:.3f} "
        f"peak_ram_gb={summary['peak_ram_gb']:.3f} "
        f"peak_gpu_memory_gb={summary['peak_gpu_memory_gb']:.3f} "
        f"samples={summary['samples']} "
        f"resource_log_csv={summary['csv_path']}"
    )


def _process_tree_usage(root_pid: int) -> Tuple[Set[int], float, int]:
    stats = _proc_stats()
    children: Dict[int, Set[int]] = {}
    for pid, (ppid, _, _) in stats.items():
        children.setdefault(ppid, set()).add(pid)

    pids = {root_pid}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        for child in children.get(pid, ()):
            if child not in pids:
                pids.add(child)
                pending.append(child)

    cpu_ticks = 0
    rss_bytes = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for pid in pids:
        item = stats.get(pid)
        if item is None:
            continue
        _, ticks, rss_pages = item
        cpu_ticks += ticks
        rss_bytes += rss_pages * page_size
    return pids, cpu_ticks / float(os.sysconf("SC_CLK_TCK")), rss_bytes


def _proc_stats() -> Dict[int, Tuple[int, int, int]]:
    out: Dict[int, Tuple[int, int, int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        stat_path = f"/proc/{pid}/stat"
        statm_path = f"/proc/{pid}/statm"
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                stat = f.read()
            with open(statm_path, "r", encoding="utf-8") as f:
                statm = f.read().split()
        except OSError:
            continue

        close = stat.rfind(")")
        if close < 0:
            continue
        fields = stat[close + 2 :].split()
        if len(fields) < 13 or len(statm) < 2:
            continue
        try:
            ppid = int(fields[1])
            utime = int(fields[11])
            stime = int(fields[12])
            rss_pages = int(statm[1])
        except ValueError:
            continue
        out[pid] = (ppid, utime + stime, rss_pages)
    return out


def _gpu_memory_gb(pids: Iterable[int]) -> float:
    pid_set = set(pids)
    if not pid_set:
        return 0.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0

    total_mib = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if pid in pid_set:
            total_mib += used_mib
    return total_mib / 1024.0
