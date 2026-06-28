"""
Experiment harness — shared interface for benchmarking full vs. Quest attention.

Provides two benchmarking modes:
1. ``benchmark_decode_step`` — simulates the **decode phase** where a single
   new query token attends to a large KV cache.  This is Quest's target scenario.
2. ``run_sweep`` — full sweep across KV-cache sizes with latency, memory,
   and correctness reporting.
"""

from __future__ import annotations

import gc
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from utils import (
    CudaTimer,
    cuda_timer,
    cosine_similarity,
    get_peak_memory_mb,
    reset_peak_memory_stats,
    format_results_table,
)


def _to_dtype(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str)


# ---------------------------------------------------------------------------
# Decode-step benchmark — the core comparison
# ---------------------------------------------------------------------------

def benchmark_decode_step(
    *,
    kv_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_kv_heads: int = 32,
    page_size: int = 64,
    top_k: int = 4,
    num_warmup: int = 10,
    num_benchmark: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[float, float, float, float]:
    """Benchmark a single decode step — 1 query token vs a KV cache of ``kv_len``.

    This directly compares the two attention patterns on **identical Q/K/V
    tensors**, isolating the effect of page-wise sparsification.

    Returns:
        (full_latency_ms, quest_latency_ms, full_memory_mb, quest_memory_mb)
    """
    device_obj = torch.device(device)

    # ---- Build random Q, K, V representing the decode step ----
    # Q: (1, num_heads, 1, head_dim)  — single decode query
    # K: (1, num_heads, kv_len, head_dim)  — cached keys
    # V: (1, num_heads, kv_len, head_dim)  — cached values

    Q = torch.randn(1, num_heads, 1, head_dim, device=device_obj, dtype=dtype)
    K = torch.randn(1, num_kv_heads, kv_len, head_dim, device=device_obj, dtype=dtype)
    V = torch.randn(1, num_kv_heads, kv_len, head_dim, device=device_obj, dtype=dtype)

    # GQA broadcast
    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        K_bc = K.repeat_interleave(num_kv_groups, dim=1)
        V_bc = V.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_bc = K
        V_bc = V

    # ---- Pages & metadata for Quest (precompute once) ----
    # Pad to page boundary
    if kv_len % page_size != 0:
        pad_len = page_size - (kv_len % page_size)
        K_padded = F.pad(K, (0, 0, 0, pad_len))
        V_padded = F.pad(V, (0, 0, 0, pad_len))
        valid_mask = torch.ones(1, 1, kv_len, 1, device=device_obj, dtype=torch.bool)
        valid_mask = F.pad(valid_mask, (0, 0, 0, pad_len))
    else:
        K_padded = K
        V_padded = V
        valid_mask = torch.ones(1, 1, kv_len, 1, device=device_obj, dtype=torch.bool)
        pad_len = 0

    kv_padded = kv_len + pad_len
    num_pages = kv_padded // page_size

    K_paged = K_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)
    V_paged = V_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)
    pad_mask = valid_mask.view(1, 1, num_pages, page_size, 1)

    # Page metadata (min/max keys per page)
    K_min = K_paged.min(dim=3).values  # (1, H_kv, num_pages, d)
    K_max = K_paged.max(dim=3).values

    # Quest scoring function — separate so we can time it independently
    def _quest_score_and_select():
        if num_kv_groups > 1:
            K_min_bc = K_min.repeat_interleave(num_kv_groups, dim=1)
            K_max_bc = K_max.repeat_interleave(num_kv_groups, dim=1)
        else:
            K_min_bc = K_min
            K_max_bc = K_max

        Q_exp = Q.expand(-1, -1, num_pages, -1)
        prod_min = Q_exp * K_min_bc
        prod_max = Q_exp * K_max_bc
        combined = torch.max(prod_min, prod_max)
        scores = combined.sum(dim=-1)  # (1, num_heads, num_pages)
        effective_k = min(top_k, num_pages)
        _, indices = torch.topk(scores, k=effective_k, dim=-1)
        return indices  # (1, num_heads, top_k)

    def _quest_sparse_attention(page_indices):
        """Compute sparse attention over selected pages."""
        top_k_actual = page_indices.size(-1)
        H_q = Q.size(1)

        # Broadcast K/V paged for GQA
        if num_kv_groups > 1:
            K_pg = K_paged.repeat_interleave(num_kv_groups, dim=1)
            V_pg = V_paged.repeat_interleave(num_kv_groups, dim=1)
        else:
            K_pg = K_paged
            V_pg = V_paged

        # pad_mask: (1, 1, num_pages, page_size, 1) — broadcast to H_q
        pm = pad_mask.expand(-1, H_q, -1, -1, -1)

        # Gather selected pages
        idx = page_indices.view(1, H_q, top_k_actual, 1, 1).expand(
            -1, -1, -1, page_size, head_dim
        )
        K_sel = K_pg.gather(dim=2, index=idx).reshape(1, H_q, top_k_actual * page_size, head_dim)
        V_sel = V_pg.gather(dim=2, index=idx).reshape(1, H_q, top_k_actual * page_size, head_dim)

        idx_m = page_indices.view(1, H_q, top_k_actual, 1, 1).expand(-1, -1, -1, page_size, 1)
        m_sel = pm.gather(dim=2, index=idx_m).reshape(1, H_q, 1, top_k_actual * page_size)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q, K_sel.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~m_sel, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V_sel)  # (1, H, 1, d)

    def _full_attention():
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q, K_bc.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V_bc)

    # ---- Warmup ----
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = _full_attention()
            indices = _quest_score_and_select()
            _ = _quest_sparse_attention(indices)

    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    # ---- Memory: Full attention ----
    reset_peak_memory_stats(device_obj)
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        _ = _full_attention()
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    full_mem = get_peak_memory_mb(device_obj)

    # ---- Memory: Quest attention ----
    reset_peak_memory_stats(device_obj)
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        indices = _quest_score_and_select()
        _ = _quest_sparse_attention(indices)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    quest_mem = get_peak_memory_mb(device_obj)

    # ---- Latency: Full attention ----
    full_times: List[float] = []
    with torch.no_grad():
        for _ in range(num_benchmark):
            with cuda_timer("full") as timer:
                _ = _full_attention()
            full_times.append(timer.elapsed_ms())

    # ---- Latency: Quest (scoring + sparse attention) ----
    quest_times: List[float] = []
    with torch.no_grad():
        for _ in range(num_benchmark):
            with cuda_timer("quest") as timer:
                idx = _quest_score_and_select()
                _ = _quest_sparse_attention(idx)
            quest_times.append(timer.elapsed_ms())

    full_lat = sum(full_times) / len(full_times)
    quest_lat = sum(quest_times) / len(quest_times)

    return full_lat, quest_lat, full_mem, quest_mem


# ---------------------------------------------------------------------------
# Sweep across KV-cache sizes
# ---------------------------------------------------------------------------

def run_sweep(
    seq_lengths: List[int],
    num_heads: int = 32,
    head_dim: int = 128,
    num_kv_heads: int = 32,
    page_size: int = 64,
    top_k: int = 4,
    num_warmup: int = 10,
    num_benchmark: int = 50,
    device: str = "cuda",
    dtype_str: str = "float16",
    verify: bool = True,
    correctness_threshold: float = 0.99,
) -> Tuple[
    List[int],
    List[float],
    List[float],
    List[float],
    List[float],
    Optional[List[float]],
]:
    """Run the full benchmark sweep across KV-cache sizes.

    Each data point simulates a **decode step**: 1 query token attending to
    a KV cache of ``seq_len`` tokens.

    Returns:
        seq_lengths, full_latencies, quest_latencies,
        full_memories, quest_memories, cosine_sims (or None)
    """
    dtype = _to_dtype(dtype_str)
    device_obj = torch.device(device)

    full_latencies: List[float] = []
    quest_latencies: List[float] = []
    full_memories: List[float] = []
    quest_memories: List[float] = []
    cosine_sims: List[float] = []

    print(f"\n{'='*70}")
    print(f"  Quest Baseline — Decode-Step Benchmark")
    print(f"  Device: {device_obj}   Dtype: {dtype_str}")
    print(f"  num_heads={num_heads}  head_dim={head_dim}  num_kv_heads={num_kv_heads}")
    print(f"  page_size={page_size}  top_k={top_k}")
    print(f"  Warmup: {num_warmup}   Benchmark iters: {num_benchmark}")
    print(f"{'='*70}\n")

    for sl in seq_lengths:
        print(f"  KV cache = {sl:>6d} tokens ... ", end="", flush=True)

        fa_lat, qa_lat, fa_mem, qa_mem = benchmark_decode_step(
            kv_len=sl,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            top_k=top_k,
            num_warmup=num_warmup,
            num_benchmark=num_benchmark,
            device=device,
            dtype=dtype,
        )

        full_latencies.append(fa_lat)
        quest_latencies.append(qa_lat)
        full_memories.append(fa_mem)
        quest_memories.append(qa_mem)

        speedup = fa_lat / max(qa_lat, 1e-6)
        mem_saved = (1.0 - qa_mem / max(fa_mem, 1e-6)) * 100.0
        print(
            f"Full: {fa_lat:7.3f} ms | "
            f"Quest: {qa_lat:7.3f} ms | "
            f"Speedup: {speedup:.2f}x | "
            f"Mem saved: {mem_saved:.1f}%",
            end="",
        )

        # Correctness verification
        if verify:
            cs = _verify_decode_step(
                kv_len=sl,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                top_k=top_k,
                device=device,
                dtype=dtype,
            )
            cosine_sims.append(cs)
            status = " ✓" if cs >= correctness_threshold else " ✗ LOW"
            print(f"  CosSim: {cs:.5f}{status}")
        else:
            print()

    return (
        list(seq_lengths),
        full_latencies,
        quest_latencies,
        full_memories,
        quest_memories,
        cosine_sims if verify else None,
    )


# ---------------------------------------------------------------------------
# Correctness verification (decode step)
# ---------------------------------------------------------------------------

def _verify_decode_step(
    kv_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_kv_heads: int = 32,
    page_size: int = 64,
    top_k: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> float:
    """Compute cosine similarity between full and Quest sparse attention outputs.

    Uses identical Q/K/V tensors so the comparison isolates the effect of
    page-wise sparse attention.
    """
    device_obj = torch.device(device)
    generator = torch.Generator(device=device_obj).manual_seed(42)

    Q = torch.randn(1, num_heads, 1, head_dim,
                     device=device_obj, dtype=dtype, generator=generator)
    K = torch.randn(1, num_kv_heads, kv_len, head_dim,
                     device=device_obj, dtype=dtype, generator=generator)
    V = torch.randn(1, num_kv_heads, kv_len, head_dim,
                     device=device_obj, dtype=dtype, generator=generator)

    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        K_bc = K.repeat_interleave(num_kv_groups, dim=1)
        V_bc = V.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_bc = K
        V_bc = V

    # ---- Full attention output ----
    with torch.no_grad():
        scale = 1.0 / math.sqrt(head_dim)
        scores_full = torch.matmul(Q, K_bc.transpose(-2, -1)) * scale
        weights_full = F.softmax(scores_full, dim=-1)
        out_full = torch.matmul(weights_full, V_bc)

    # ---- Quest sparse attention output ----
    # Page construction
    if kv_len % page_size != 0:
        pad_len = page_size - (kv_len % page_size)
        K_padded = F.pad(K, (0, 0, 0, pad_len))
        V_padded = F.pad(V, (0, 0, 0, pad_len))
        valid_mask = torch.ones(1, 1, kv_len, 1, device=device_obj, dtype=torch.bool)
        valid_mask = F.pad(valid_mask, (0, 0, 0, pad_len))
    else:
        K_padded = K
        V_padded = V
        valid_mask = torch.ones(1, 1, kv_len, 1, device=device_obj, dtype=torch.bool)

    kv_padded = K_padded.size(2)
    num_pages = kv_padded // page_size

    K_paged = K_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)
    V_paged = V_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)
    pad_mask = valid_mask.view(1, 1, num_pages, page_size, 1)

    K_min = K_paged.min(dim=3).values
    K_max = K_paged.max(dim=3).values

    if num_kv_groups > 1:
        K_min_bc = K_min.repeat_interleave(num_kv_groups, dim=1)
        K_max_bc = K_max.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_min_bc = K_min
        K_max_bc = K_max

    with torch.no_grad():
        # Stage 1: Score pages
        Q_exp = Q.expand(-1, -1, num_pages, -1)
        prod_min = Q_exp * K_min_bc
        prod_max = Q_exp * K_max_bc
        combined = torch.max(prod_min, prod_max)
        scores = combined.sum(dim=-1)
        effective_k = min(top_k, num_pages)
        _, page_indices = torch.topk(scores, k=effective_k, dim=-1)

        # Stage 2: Sparse attention on selected pages
        H_q = Q.size(1)
        top_k_actual = page_indices.size(-1)
        if num_kv_groups > 1:
            K_pg = K_paged.repeat_interleave(num_kv_groups, dim=1)
            V_pg = V_paged.repeat_interleave(num_kv_groups, dim=1)
        else:
            K_pg = K_paged
            V_pg = V_paged

        # pad_mask: (1, 1, num_pages, page_size, 1) — broadcast to H_q
        pm = pad_mask.expand(-1, H_q, -1, -1, -1)

        idx = page_indices.view(1, H_q, top_k_actual, 1, 1).expand(
            -1, -1, -1, page_size, head_dim
        )
        K_sel = K_pg.gather(dim=2, index=idx).reshape(1, H_q, top_k_actual * page_size, head_dim)
        V_sel = V_pg.gather(dim=2, index=idx).reshape(1, H_q, top_k_actual * page_size, head_dim)

        idx_m = page_indices.view(1, H_q, top_k_actual, 1, 1).expand(-1, -1, -1, page_size, 1)
        m_sel = pm.gather(dim=2, index=idx_m).reshape(1, H_q, 1, top_k_actual * page_size)

        scores_q = torch.matmul(Q, K_sel.transpose(-2, -1)) * scale
        scores_q = scores_q.masked_fill(~m_sel, float("-inf"))
        weights_q = F.softmax(scores_q, dim=-1)
        out_quest = torch.matmul(weights_q, V_sel)

    out_full_f = out_full.to(device_obj).float()
    out_quest_f = out_quest.to(device_obj).float()

    return cosine_similarity(out_full_f, out_quest_f)
