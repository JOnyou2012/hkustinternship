"""
Utility helpers: GPU-synchronized timing, peak memory tracking, cosine
similarity for correctness verification, and result formatting.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def get_peak_memory_mb(device: Optional[torch.device] = None) -> float:
    """Return peak allocated GPU memory in MiB since the last reset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak-memory statistics to get a clean reading."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


# ---------------------------------------------------------------------------
# GPU-synchronized timer
# ---------------------------------------------------------------------------

class CudaTimer:
    """GPU-synchronised timer using CUDA events; falls back to wall-clock on CPU."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._elapsed_ms: float = 0.0
        self._use_cuda = torch.cuda.is_available()

    def start(self) -> None:
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self._use_cuda:
            assert self._end_event is not None
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)  # type: ignore[union-attr]
        else:
            self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0

    def elapsed_ms(self) -> float:
        return self._elapsed_ms


@contextmanager
def cuda_timer(name: str = ""):
    """Context manager yielding a CudaTimer for GPU-synchronized timing."""
    ct = CudaTimer(name)
    ct.start()
    try:
        yield ct
    finally:
        ct.stop()


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Cosine similarity between two tensors (returns a Python float)."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    dot = torch.dot(a_flat, b_flat).item()
    norm_a = torch.norm(a_flat).item()
    norm_b = torch.norm(b_flat).item()
    return dot / (norm_a * norm_b + eps)


# ---------------------------------------------------------------------------
# Pretty-printed results
# ---------------------------------------------------------------------------

def format_results_table(
    seq_lengths: List[int],
    full_latencies_ms: List[float],
    quest_latencies_ms: List[float],
    full_memory_mb: List[float],
    quest_memory_mb: List[float],
    cosine_sims: Optional[List[float]] = None,
) -> str:
    """Return an ASCII table summarising the benchmark sweep."""
    header = (
        f"{'Seq Len':>7s} | "
        f"{'Full (ms)':>10s} | "
        f"{'Quest (ms)':>11s} | "
        f"{'Speedup':>8s} | "
        f"{'Full Mem':>9s} | "
        f"{'Quest Mem':>10s} | "
        f"{'Mem Saved':>10s}"
    )
    if cosine_sims is not None:
        header += f" | {'CosSim':>7s}"

    sep = "-" * len(header)
    rows: List[str] = [header, sep]

    for i, sl in enumerate(seq_lengths):
        speedup = full_latencies_ms[i] / max(quest_latencies_ms[i], 1e-6)
        mem_saved = (1.0 - quest_memory_mb[i] / max(full_memory_mb[i], 1e-6)) * 100.0
        row = (
            f"{sl:>7d} | "
            f"{full_latencies_ms[i]:>10.3f} | "
            f"{quest_latencies_ms[i]:>11.3f} | "
            f"{speedup:>7.2f}x | "
            f"{full_memory_mb[i]:>9.2f} | "
            f"{quest_memory_mb[i]:>10.2f} | "
            f"{mem_saved:>9.1f}%"
        )
        if cosine_sims is not None:
            row += f" | {cosine_sims[i]:>7.4f}"
        rows.append(row)

    return "\n".join(rows)


def format_phase2_results_table(
    seq_lengths: List[int],
    full_latencies_ms: List[float],
    hier_latencies_ms: List[float],
    full_memory_mb: List[float],
    hier_memory_mb: List[float],
    cosine_sims: Optional[List[float]] = None,
    quality_metrics: Optional[List[dict]] = None,
    token_budget: int = 256,
) -> str:
    """ASCII table for Phase 2 benchmark sweep including quality metrics."""
    header = (
        f"{'Seq Len':>7s} | "
        f"{'Full (ms)':>10s} | "
        f"{'Hier (ms)':>11s} | "
        f"{'Speedup':>8s} | "
        f"{'Full Mem':>9s} | "
        f"{'Hier Mem':>10s} | "
        f"{'Mem Saved':>10s}"
    )
    if cosine_sims is not None:
        header += f" | {'CosSim':>7s}"
    if quality_metrics is not None:
        header += f" | {'Rec@B':>7s} | {'Jac':>6s}"

    sep = "-" * len(header)
    rows: List[str] = [header, sep]

    for i, sl in enumerate(seq_lengths):
        speedup = full_latencies_ms[i] / max(hier_latencies_ms[i], 1e-6)
        mem_saved = (1.0 - hier_memory_mb[i] / max(full_memory_mb[i], 1e-6)) * 100.0
        row = (
            f"{sl:>7d} | "
            f"{full_latencies_ms[i]:>10.3f} | "
            f"{hier_latencies_ms[i]:>11.3f} | "
            f"{speedup:>7.2f}x | "
            f"{full_memory_mb[i]:>9.2f} | "
            f"{hier_memory_mb[i]:>10.2f} | "
            f"{mem_saved:>9.1f}%"
        )
        if cosine_sims is not None:
            row += f" | {cosine_sims[i]:>7.4f}"
        if quality_metrics is not None and i < len(quality_metrics):
            qm = quality_metrics[i]
            row += (
                f" | {qm['token_recall_mean']:>7.4f}"
                f" | {qm['attention_jaccard_mean']:>6.4f}"
            )
        rows.append(row)

    return "\n".join(rows)
