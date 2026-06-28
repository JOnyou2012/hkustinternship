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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Hierarchical Token-Level Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_decode_step_phase2(
    *,
    kv_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_kv_heads: int = 32,
    page_size: int = 64,
    top_k: int = 4,
    macro_multiplier: int = 3,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 64,
    adaptive_budget: bool = True,
    num_warmup: int = 10,
    num_benchmark: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    return_quality_metrics: bool = False,
) -> Tuple[float, float, float, float] | Tuple[
    float, float, float, float, dict
]:
    """Benchmark Phase 2 hierarchical token-level attention.

    Compares full attention against the hierarchical sparse method on
    identical Q/K/V tensors.

    Returns:
        (full_lat_ms, hier_lat_ms, full_mem_mb, hier_mem_mb)
        — or with an extra ``dict`` of quality metrics if
        ``return_quality_metrics`` is True.
    """
    from hierarchical_attention import HierarchicalTokenAttention

    device_obj = torch.device(device)
    token_budget = top_k * page_size
    num_kv_groups = num_heads // num_kv_heads

    # ---- Random Q, K, V ----
    Q = torch.randn(1, num_heads, 1, head_dim, device=device_obj, dtype=dtype)
    K = torch.randn(1, num_kv_heads, kv_len, head_dim, device=device_obj, dtype=dtype)
    V = torch.randn(1, num_kv_heads, kv_len, head_dim, device=device_obj, dtype=dtype)

    # GQA broadcast for full attention
    if num_kv_groups > 1:
        K_bc = K.repeat_interleave(num_kv_groups, dim=1)
        V_bc = V.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_bc = K
        V_bc = V

    # ---- Page building for hierarchical ----
    if kv_len % page_size != 0:
        pad_len = page_size - (kv_len % page_size)
        K_padded = F.pad(K, (0, 0, 0, pad_len))
        V_padded = F.pad(V, (0, 0, 0, pad_len))
    else:
        K_padded = K
        V_padded = V

    kv_padded = K_padded.size(2)
    num_pages = kv_padded // page_size

    K_paged = K_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)
    V_paged = V_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)

    K_min = K_paged.min(dim=3).values
    K_max = K_paged.max(dim=3).values

    # ---- Macro page scoring function ----
    def _page_scoring():
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
        scores = combined.sum(dim=-1)  # (1, H, num_pages)

        # Adaptive M
        base_M = min(macro_multiplier * top_k, num_pages)
        if adaptive_budget and num_pages > 2:
            avg_scores = scores.mean(dim=(0, 1))
            sorted_s = avg_scores.sort(descending=True).values
            top2_sum = sorted_s[:2].sum()
            total_sum = sorted_s.sum() + 1e-8
            concentration = (top2_sum / total_sum).item()
            if concentration > 0.5:
                M_eff = max(base_M // 2, 2)
            elif concentration < 0.3:
                M_eff = min(base_M * 2, num_pages)
            else:
                M_eff = base_M
        else:
            M_eff = base_M
        M_eff = max(M_eff, top_k)
        M_eff = min(M_eff, num_pages)

        # Use the module's reusable macro page selection with sink/recent protection
        macro_pages = HierarchicalTokenAttention.select_macro_pages(
            page_scores=scores,
            M=M_eff,
            num_pages=num_pages,
            kv_len=kv_len,
            page_size=page_size,
            num_sink_tokens=num_sink_tokens,
            num_recent_tokens=num_recent_tokens,
        )
        return scores, macro_pages, M_eff

    # ---- Token-level scoring ----
    def _token_scoring(macro_pages):
        M = macro_pages.size(-1)
        H_q = Q.size(1)

        K_pg = K_paged
        if num_kv_groups > 1:
            K_pg = K_paged.repeat_interleave(num_kv_groups, dim=1)

        # Gather selected pages → flatten tokens
        idx_k = macro_pages.view(1, H_q, M, 1, 1).expand(-1, -1, -1, page_size, head_dim)
        K_sel = K_pg.gather(dim=2, index=idx_k).reshape(1, H_q, M * page_size, head_dim)

        # Valid positions
        valid_mask = torch.ones(1, 1, kv_len, 1, device=device_obj, dtype=torch.bool)
        if kv_len % page_size != 0:
            valid_mask = F.pad(valid_mask, (0, 0, 0, kv_padded - kv_len))
        valid_paged = valid_mask.view(1, 1, num_pages, page_size, 1)
        if valid_paged.size(1) != H_q:
            valid_paged = valid_paged.repeat_interleave(H_q, dim=1)

        idx_m = macro_pages.view(1, H_q, M, 1, 1).expand(-1, -1, -1, page_size, 1)
        validity = valid_paged.gather(dim=2, index=idx_m).reshape(1, H_q, M * page_size).bool()

        # Per-token exact scores
        scale = 1.0 / math.sqrt(head_dim)
        tk_scores = torch.matmul(Q, K_sel.transpose(-2, -1)).squeeze(-2) * scale
        tk_scores = tk_scores.masked_fill(~validity, float("-inf"))

        # Global positions
        base_pos = macro_pages * page_size
        offsets = torch.arange(page_size, device=device_obj).view(1, 1, 1, page_size)
        global_pos = (base_pos.unsqueeze(-1) + offsets).reshape(1, H_q, M * page_size)

        return tk_scores, validity, global_pos

    # ---- Consolidation ----
    def _consolidate(tk_scores, validity, global_pos):
        B, H, _ = tk_scores.shape
        B_target = token_budget

        eff_sink = min(num_sink_tokens, kv_len)
        eff_recent = min(num_recent_tokens, kv_len)

        is_sink = global_pos < eff_sink
        is_recent = global_pos >= (kv_len - eff_recent)
        is_protected = (is_sink | is_recent) & validity

        boosted = tk_scores.clone()
        if is_protected.sum() > B_target * B * H:
            boosted[is_recent & validity] = float("inf")
            boosted[is_sink & validity & ~is_recent] = float("inf") / 2
        else:
            boosted[is_protected] = float("inf")

        effective_B = min(B_target, tk_scores.size(-1))
        _, topk_idx = torch.topk(boosted, k=effective_B, dim=-1)

        sel_pos = global_pos.gather(dim=-1, index=topk_idx)
        sel_valid = validity.gather(dim=-1, index=topk_idx)

        return sel_pos, sel_valid

    # ---- Sparse attention on selected tokens ----
    def _token_sparse_attn(sel_pos, sel_valid):
        H_q = Q.size(1)
        B_target = sel_pos.size(-1)

        K_pg = K_padded
        V_pg = V_padded
        if num_kv_groups > 1:
            K_pg = K_padded.repeat_interleave(num_kv_groups, dim=1)
            V_pg = V_padded.repeat_interleave(num_kv_groups, dim=1)

        idx_g = sel_pos.unsqueeze(-1).expand(-1, -1, -1, head_dim).long()
        K_tk = K_pg.gather(dim=2, index=idx_g)
        V_tk = V_pg.gather(dim=2, index=idx_g)

        scale = 1.0 / math.sqrt(head_dim)
        attn_s = torch.matmul(Q, K_tk.transpose(-2, -1)) * scale
        attn_s = attn_s.masked_fill(~sel_valid.unsqueeze(-2), float("-inf"))
        attn_w = F.softmax(attn_s, dim=-1)
        return torch.matmul(attn_w, V_tk)

    # ---- Full attention ----
    def _full_attention():
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q, K_bc.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V_bc)

    # ── Warmup ────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = _full_attention()
            _, mp, _ = _page_scoring()
            ts, tv, gp = _token_scoring(mp)
            sp, sv = _consolidate(ts, tv, gp)
            _ = _token_sparse_attn(sp, sv)

    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    # ── Memory ────────────────────────────────────────────────────────
    reset_peak_memory_stats(device_obj)
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        _ = _full_attention()
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    full_mem = get_peak_memory_mb(device_obj)

    reset_peak_memory_stats(device_obj)
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        _, mp, _ = _page_scoring()
        ts, tv, gp = _token_scoring(mp)
        sp, sv = _consolidate(ts, tv, gp)
        _ = _token_sparse_attn(sp, sv)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    hier_mem = get_peak_memory_mb(device_obj)

    # ── Latency ───────────────────────────────────────────────────────
    full_times: List[float] = []
    with torch.no_grad():
        for _ in range(num_benchmark):
            with cuda_timer("full") as timer:
                _ = _full_attention()
            full_times.append(timer.elapsed_ms())

    hier_times: List[float] = []
    with torch.no_grad():
        for _ in range(num_benchmark):
            with cuda_timer("hierarchical") as timer:
                _, mp, _ = _page_scoring()
                ts, tv, gp_ = _token_scoring(mp)
                sp, sv = _consolidate(ts, tv, gp_)
                _ = _token_sparse_attn(sp, sv)
            hier_times.append(timer.elapsed_ms())

    full_lat = sum(full_times) / len(full_times)
    hier_lat = sum(hier_times) / len(hier_times)

    if return_quality_metrics:
        quality = _compute_quality_metrics(
            Q=Q,
            K_bc=K_bc,
            V_bc=V_bc,
            token_budget=token_budget,
            num_sink_tokens=num_sink_tokens,
            num_recent_tokens=num_recent_tokens,
            kv_len=kv_len,
            device_obj=device_obj,
            dtype=dtype,
            _page_scoring_fn=_page_scoring,
            _token_scoring_fn=_token_scoring,
            _consolidate_fn=_consolidate,
        )
        return full_lat, hier_lat, full_mem, hier_mem, quality

    return full_lat, hier_lat, full_mem, hier_mem


# ──────────────────────────────────────────────────────────────────────
# Quality metrics: token recall & attention overlap
# ──────────────────────────────────────────────────────────────────────

def _compute_quality_metrics(
    *,
    Q: torch.Tensor,
    K_bc: torch.Tensor,
    V_bc: torch.Tensor,
    token_budget: int,
    num_sink_tokens: int,
    num_recent_tokens: int,
    kv_len: int,
    device_obj: torch.device,
    dtype: torch.dtype,
    _page_scoring_fn,
    _token_scoring_fn,
    _consolidate_fn,
) -> dict:
    """Compute token recall and attention overlap against full attention.

    Token Recall@B
    ──────────────
    Fraction of the full-attention Top-B tokens (ranked by attention weight)
    that are also selected by the hierarchical sparse method.

    Attention Jaccard Overlap
    ─────────────────────────
    |S_sparse ∩ S_full| / |S_sparse ∪ S_full| where S_* is the set of
    selected token positions.
    """
    head_dim = Q.size(-1)
    H_q = Q.size(1)

    with torch.no_grad():
        # Full attention weights
        scale = 1.0 / math.sqrt(head_dim)
        scores_full = torch.matmul(Q, K_bc.transpose(-2, -1)) * scale
        weights_full = F.softmax(scores_full, dim=-1)  # (1, H, 1, kv_len)

        # Hierarchical selection
        _, macro_pages, _ = _page_scoring_fn()
        tk_scores, validity, gp = _token_scoring_fn(macro_pages)
        sel_pos, sel_valid = _consolidate_fn(tk_scores, validity, gp)
        # sel_pos: (1, H, B) — global positions

        # ---- Token Recall@B ----
        # Top-B tokens by full-attention weight
        _, full_top_idx = torch.topk(
            weights_full.squeeze(-2), k=min(token_budget, kv_len), dim=-1
        )  # (1, H, B)

        recall_per_head = []
        overlap_per_head = []
        for h in range(H_q):
            sparse_set = set(
                sel_pos[0, h, sel_valid[0, h]].long().cpu().tolist()
            )
            full_set = set(full_top_idx[0, h].long().cpu().tolist())

            # Recall: fraction of full top-B captured by sparse
            captured = len(sparse_set & full_set)
            recall = captured / max(len(full_set), 1)
            recall_per_head.append(recall)

            # Jaccard overlap
            union = len(sparse_set | full_set)
            jaccard = captured / max(union, 1)
            overlap_per_head.append(jaccard)

        recall_mean = sum(recall_per_head) / len(recall_per_head)
        overlap_mean = sum(overlap_per_head) / len(overlap_per_head)

    return {
        "token_recall_mean": recall_mean,
        "token_recall_per_head": recall_per_head,
        "attention_jaccard_mean": overlap_mean,
        "attention_jaccard_per_head": overlap_per_head,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Sweep (uses the same structure as run_sweep)
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep_phase2(
    seq_lengths: List[int],
    num_heads: int = 32,
    head_dim: int = 128,
    num_kv_heads: int = 32,
    page_size: int = 64,
    top_k: int = 4,
    macro_multiplier: int = 3,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 64,
    adaptive_budget: bool = True,
    num_warmup: int = 10,
    num_benchmark: int = 50,
    device: str = "cuda",
    dtype_str: str = "float16",
    verify: bool = True,
    correctness_threshold: float = 0.99,
    compute_quality: bool = True,
) -> Tuple[
    List[int],
    List[float],  # full latencies
    List[float],  # hierarchical latencies
    List[float],  # full memories
    List[float],  # hierarchical memories
    Optional[List[float]],  # cosine similarities
    Optional[List[dict]],   # quality metrics per length
]:
    """Run Phase 2 benchmark sweep across KV-cache sizes.

    Each data point simulates a **decode step**: 1 query token vs a large
    KV cache.  Compares full attention against the hierarchical token-level
    sparse method and reports both latency/throughput and quality metrics.

    Returns:
        seq_lengths, full_latencies, hier_latencies,
        full_memories, hier_memories, cosine_sims, quality_metrics
    """
    dtype = _to_dtype(dtype_str)
    device_obj = torch.device(device)
    token_budget = top_k * page_size

    full_latencies: List[float] = []
    hier_latencies: List[float] = []
    full_memories: List[float] = []
    hier_memories: List[float] = []
    cosine_sims: List[float] = []
    quality_metrics_all: List[dict] = []

    print(f"\n{'='*70}")
    print(f"  Phase 2 — Hierarchical Token-Level Attention")
    print(f"  iSING Lab, HKUST")
    print(f"  Device: {device_obj}   Dtype: {dtype_str}")
    print(f"  num_heads={num_heads}  head_dim={head_dim}  num_kv_heads={num_kv_heads}")
    print(f"  page_size={page_size}  top_k={top_k}  token_budget={token_budget}")
    print(f"  macro_multiplier={macro_multiplier}  sink={num_sink_tokens}  "
          f"recent={num_recent_tokens}  adaptive={adaptive_budget}")
    print(f"  Warmup: {num_warmup}   Benchmark iters: {num_benchmark}")
    print(f"{'='*70}\n")

    for sl in seq_lengths:
        print(f"  KV cache = {sl:>6d} tokens ... ", end="", flush=True)

        result = benchmark_decode_step_phase2(
            kv_len=sl,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            top_k=top_k,
            macro_multiplier=macro_multiplier,
            num_sink_tokens=num_sink_tokens,
            num_recent_tokens=num_recent_tokens,
            adaptive_budget=adaptive_budget,
            num_warmup=num_warmup,
            num_benchmark=num_benchmark,
            device=device,
            dtype=dtype,
            return_quality_metrics=compute_quality,
        )

        if compute_quality:
            fa_lat, qa_lat, fa_mem, qa_mem, quality = result
            quality_metrics_all.append(quality)
        else:
            fa_lat, qa_lat, fa_mem, qa_mem = result

        full_latencies.append(fa_lat)
        hier_latencies.append(qa_lat)
        full_memories.append(fa_mem)
        hier_memories.append(qa_mem)

        speedup = fa_lat / max(qa_lat, 1e-6)
        mem_saved = (1.0 - qa_mem / max(fa_mem, 1e-6)) * 100.0
        print(
            f"Full: {fa_lat:7.3f} ms | "
            f"Hier: {qa_lat:7.3f} ms | "
            f"Speedup: {speedup:.2f}x | "
            f"Mem saved: {mem_saved:.1f}%",
            end="",
        )

        # Quality metrics
        if compute_quality and quality_metrics_all:
            qm = quality_metrics_all[-1]
            print(
                f"  Rec@{token_budget}: {qm['token_recall_mean']:.4f}  "
                f"Jac: {qm['attention_jaccard_mean']:.4f}",
                end="",
            )

        # Cosine similarity verification
        if verify:
            cs = _verify_hierarchical_decode_step(
                kv_len=sl,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
                top_k=top_k,
                macro_multiplier=macro_multiplier,
                num_sink_tokens=num_sink_tokens,
                num_recent_tokens=num_recent_tokens,
                adaptive_budget=adaptive_budget,
                token_budget=token_budget,
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
        hier_latencies,
        full_memories,
        hier_memories,
        cosine_sims if verify else None,
        quality_metrics_all if compute_quality else None,
    )


# ──────────────────────────────────────────────────────────────────────
# Phase 2 correctness verification
# ──────────────────────────────────────────────────────────────────────

def _verify_hierarchical_decode_step(
    kv_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_kv_heads: int = 32,
    page_size: int = 64,
    top_k: int = 4,
    macro_multiplier: int = 3,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 64,
    adaptive_budget: bool = True,
    token_budget: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> float:
    """Cosine similarity between full and hierarchical attention outputs."""
    device_obj = torch.device(device)
    generator = torch.Generator(device=device_obj).manual_seed(42)

    if token_budget is None:
        token_budget = top_k * page_size

    num_kv_groups = num_heads // num_kv_heads

    Q = torch.randn(1, num_heads, 1, head_dim,
                    device=device_obj, dtype=dtype, generator=generator)
    K = torch.randn(1, num_kv_heads, kv_len, head_dim,
                    device=device_obj, dtype=dtype, generator=generator)
    V = torch.randn(1, num_kv_heads, kv_len, head_dim,
                    device=device_obj, dtype=dtype, generator=generator)

    if num_kv_groups > 1:
        K_bc = K.repeat_interleave(num_kv_groups, dim=1)
        V_bc = V.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_bc = K
        V_bc = V

    # ---- Full attention ----
    with torch.no_grad():
        scale = 1.0 / math.sqrt(head_dim)
        scores_full = torch.matmul(Q, K_bc.transpose(-2, -1)) * scale
        weights_full = F.softmax(scores_full, dim=-1)
        out_full = torch.matmul(weights_full, V_bc)

    # ---- Hierarchical attention (replicating the pipeline inline) ----
    if kv_len % page_size != 0:
        pad_len = page_size - (kv_len % page_size)
        K_padded = F.pad(K, (0, 0, 0, pad_len))
        V_padded = F.pad(V, (0, 0, 0, pad_len))
    else:
        K_padded = K
        V_padded = V

    kv_padded = K_padded.size(2)
    num_pages = kv_padded // page_size

    K_paged = K_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)
    V_paged = V_padded.view(1, num_kv_heads, num_pages, page_size, head_dim)

    K_min = K_paged.min(dim=3).values
    K_max = K_paged.max(dim=3).values

    with torch.no_grad():
        # Stage 1: Page scoring
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
        page_scores = combined.sum(dim=-1)

        base_M = min(macro_multiplier * top_k, num_pages)
        if adaptive_budget and num_pages > 2:
            avg_scores = page_scores.mean(dim=(0, 1))
            sorted_s = avg_scores.sort(descending=True).values
            top2_s = sorted_s[:2].sum()
            total_s = sorted_s.sum() + 1e-8
            concentration = (top2_s / total_s).item()
            if concentration > 0.5:
                M_eff = max(base_M // 2, 2)
            elif concentration < 0.3:
                M_eff = min(base_M * 2, num_pages)
            else:
                M_eff = base_M
        else:
            M_eff = base_M
        M_eff = max(M_eff, top_k)
        M_eff = min(M_eff, num_pages)

        from hierarchical_attention import HierarchicalTokenAttention
        macro_pages = HierarchicalTokenAttention.select_macro_pages(
            page_scores=page_scores,
            M=M_eff,
            num_pages=num_pages,
            kv_len=kv_len,
            page_size=page_size,
            num_sink_tokens=num_sink_tokens,
            num_recent_tokens=num_recent_tokens,
        )
        M = macro_pages.size(-1)
        H_q = Q.size(1)

        # Stage 2: Token scoring
        if num_kv_groups > 1:
            K_pg = K_paged.repeat_interleave(num_kv_groups, dim=1)
        else:
            K_pg = K_paged

        idx_k = macro_pages.view(1, H_q, M, 1, 1).expand(-1, -1, -1, page_size, head_dim)
        K_sel = K_pg.gather(dim=2, index=idx_k).reshape(1, H_q, M * page_size, head_dim)

        valid_mask = torch.ones(1, 1, kv_len, 1, device=device_obj, dtype=torch.bool)
        if kv_len % page_size != 0:
            valid_mask = F.pad(valid_mask, (0, 0, 0, kv_padded - kv_len))
        valid_paged = valid_mask.view(1, 1, num_pages, page_size, 1)
        if valid_paged.size(1) != H_q:
            valid_paged = valid_paged.repeat_interleave(H_q, dim=1)

        idx_m = macro_pages.view(1, H_q, M, 1, 1).expand(-1, -1, -1, page_size, 1)
        validity = valid_paged.gather(dim=2, index=idx_m).reshape(1, H_q, M * page_size).bool()

        tk_scores = torch.matmul(Q, K_sel.transpose(-2, -1)).squeeze(-2) * scale
        tk_scores = tk_scores.masked_fill(~validity, float("-inf"))

        base_pos = macro_pages * page_size
        offsets = torch.arange(page_size, device=device_obj).view(1, 1, 1, page_size)
        global_pos = (base_pos.unsqueeze(-1) + offsets).reshape(1, H_q, M * page_size)

        # Stage 3: Consolidation
        eff_sink = min(num_sink_tokens, kv_len)
        eff_recent = min(num_recent_tokens, kv_len)
        is_sink = global_pos < eff_sink
        is_recent = global_pos >= (kv_len - eff_recent)
        is_protected = (is_sink | is_recent) & validity

        boosted = tk_scores.clone()
        if is_protected.sum() > token_budget:
            boosted[is_recent & validity] = float("inf")
            boosted[is_sink & validity & ~is_recent] = float("inf") / 2
        else:
            boosted[is_protected] = float("inf")

        effective_B = min(token_budget, tk_scores.size(-1))
        _, topk_idx = torch.topk(boosted, k=effective_B, dim=-1)

        sel_pos = global_pos.gather(dim=-1, index=topk_idx)
        sel_valid = validity.gather(dim=-1, index=topk_idx)

        # Sparse attention
        if num_kv_groups > 1:
            K_pg2 = K_padded.repeat_interleave(num_kv_groups, dim=1)
            V_pg2 = V_padded.repeat_interleave(num_kv_groups, dim=1)
        else:
            K_pg2 = K_padded
            V_pg2 = V_padded

        idx_g = sel_pos.unsqueeze(-1).expand(-1, -1, -1, head_dim).long()
        K_tk = K_pg2.gather(dim=2, index=idx_g)
        V_tk = V_pg2.gather(dim=2, index=idx_g)

        attn_s = torch.matmul(Q, K_tk.transpose(-2, -1)) * scale
        attn_s = attn_s.masked_fill(~sel_valid.unsqueeze(-2), float("-inf"))
        attn_w = F.softmax(attn_s, dim=-1)
        out_hier = torch.matmul(attn_w, V_tk)

    out_full_f = out_full.to(device_obj).float()
    out_hier_f = out_hier.to(device_obj).float()

    return cosine_similarity(out_full_f, out_hier_f)
