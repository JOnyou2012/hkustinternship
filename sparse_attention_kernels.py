"""
Standalone sparse attention kernel functions.

These operate on **already-projected** Q/K/V tensors — no nn.Module
dependency, no internal projection layers.  This clean separation is the
dependency that unlocks HuggingFace model integration: we can intercept
Q/K/V from any HF model and route them through these kernels without
duplicating projection weights.

All kernels in this file accept Q/K/V with shape (B, heads, T, head_dim) and
return output of shape (B, heads, T_out, head_dim) plus any metadata.

Kernel                         Purpose
───────────────────────────────────────────────────────────────
quest_sparse_attention          Phase 1 — Quest page-wise sparse attention
hierarchical_sparse_attention   Phase 2 — 3-stage token-level sparse attention
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Page-building helper (shared by both kernels)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pages(
    K: torch.Tensor,
    V: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Partition K and V tensors into fixed-size pages.

    Args:
        K: Key tensor   of shape (B, num_kv_heads, T_kv, head_dim).
        V: Value tensor of shape (B, num_kv_heads, T_kv, head_dim).
        page_size: Tokens per page.

    Returns:
        K_paged:   (B, num_kv_heads, num_pages, page_size, head_dim)
        V_paged:   (B, num_kv_heads, num_pages, page_size, head_dim)
        pad_mask:  (B, 1, num_pages, page_size, 1) — True for valid tokens.
        num_pages: int
    """
    B, H, T, d = K.shape

    valid_mask = torch.ones(B, 1, T, 1, device=K.device, dtype=torch.bool)

    if T % page_size != 0:
        pad_len = page_size - (T % page_size)
        K = F.pad(K, (0, 0, 0, pad_len))
        V = F.pad(V, (0, 0, 0, pad_len))
        valid_mask = F.pad(valid_mask, (0, 0, 0, pad_len))
        T_padded = T + pad_len
    else:
        T_padded = T

    num_pages = T_padded // page_size

    K_paged = K.view(B, H, num_pages, page_size, d)
    V_paged = V.view(B, H, num_pages, page_size, d)
    pad_mask = valid_mask.view(B, 1, num_pages, page_size, 1).bool()

    return K_paged, V_paged, pad_mask, num_pages


# ═══════════════════════════════════════════════════════════════════════════════
# Quest — Page-wise sparse attention kernel
# ═══════════════════════════════════════════════════════════════════════════════

def quest_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    page_size: int,
    top_k: int,
    mask: Optional[torch.Tensor] = None,
    num_kv_groups: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quest page-wise sparse attention — standalone kernel.

    Full Quest pipeline:
      1. Build pages from K/V.
      2. Compute per-page element-wise min/max of keys.
      3. Score pages via Quest heuristic: Q ⊙ {K_min, K_max} →
         per-channel max → sum over head_dim.
      4. Select Top-K pages.
      5. Compute sparse attention over the selected pages only.

    When T_q > 1 (prefill), positions 0..T-2 use standard full attention
    and only the last position uses Quest sparse attention.  This matches
    the Quest decode-step semantics: full prefill, sparse decode.

    Args:
        Q:         (B, num_heads, T_q, head_dim)
        K:         (B, num_kv_heads, T_kv, head_dim)
        V:         (B, num_kv_heads, T_kv, head_dim)
        page_size: Tokens per page.
        top_k:     Number of pages to select for sparse attention.
        mask:      Optional causal/additive mask for the full-attention
                   prefill fallback (used when T_q > 1).
        num_kv_groups: num_heads // num_kv_heads (1 for MHA, >1 for GQA).

    Returns:
        output:        (B, num_heads, T_q, head_dim)
        page_indices:  (B, num_heads, top_k) — selected page indices.
    """
    B, H_q, T_q, d = Q.shape
    _, H_kv, T_kv, _ = K.shape
    device = Q.device

    # ---- Build pages ----
    K_paged, V_paged, pad_mask, num_pages = _build_pages(K, V, page_size)

    # ---- Page metadata (min/max) ----
    K_min = K_paged.min(dim=3).values  # (B, H_kv, num_pages, d)
    K_max = K_paged.max(dim=3).values

    # ---- GQA broadcast ----
    if num_kv_groups > 1:
        K_min_bc = K_min.repeat_interleave(num_kv_groups, dim=1)
        K_max_bc = K_max.repeat_interleave(num_kv_groups, dim=1)
        K_pg = K_paged.repeat_interleave(num_kv_groups, dim=1)
        V_pg = V_paged.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_min_bc = K_min
        K_max_bc = K_max
        K_pg = K_paged
        V_pg = V_paged
    # pad_mask: (B, 1, num_pages, page_size, 1) → (B, H_q, ...)
    pm = pad_mask.expand(-1, H_q, -1, -1, -1)

    # ---- Decode query (last token) ----
    if T_q == 1:
        Q_decode = Q
    else:
        Q_decode = Q[:, :, -1:, :]

    # ---- Stage 1: Score pages ----
    Q_exp = Q_decode.expand(-1, -1, num_pages, -1)
    prod_min = Q_exp * K_min_bc
    prod_max = Q_exp * K_max_bc
    combined = torch.max(prod_min, prod_max)
    scores = combined.sum(dim=-1)  # (B, H_q, num_pages)

    # ---- Select Top-K pages ----
    effective_k = min(top_k, num_pages)
    _, page_indices = torch.topk(scores, k=effective_k, dim=-1)
    top_k_actual = page_indices.size(-1)

    # ---- Stage 2: Sparse attention on selected pages ----
    idx = page_indices.view(B, H_q, top_k_actual, 1, 1).expand(
        -1, -1, -1, page_size, d
    )
    K_sel = K_pg.gather(dim=2, index=idx).reshape(
        B, H_q, top_k_actual * page_size, d
    )
    V_sel = V_pg.gather(dim=2, index=idx).reshape(
        B, H_q, top_k_actual * page_size, d
    )

    idx_m = page_indices.view(B, H_q, top_k_actual, 1, 1).expand(
        -1, -1, -1, page_size, 1
    )
    m_sel = pm.gather(dim=2, index=idx_m).reshape(
        B, H_q, 1, top_k_actual * page_size
    )

    scale = 1.0 / math.sqrt(d)
    attn_scores = torch.matmul(Q_decode, K_sel.transpose(-2, -1)) * scale
    attn_scores = attn_scores.masked_fill(~m_sel, float("-inf"))
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_weights, V_sel)  # (B, H_q, 1, d)

    # ---- Multi-token prefill: full attention for positions 0..T-2 ----
    if T_q > 1:
        K_bc = K
        V_bc = V
        if num_kv_groups > 1:
            K_bc = K.repeat_interleave(num_kv_groups, dim=1)
            V_bc = V.repeat_interleave(num_kv_groups, dim=1)

        Q_prefill = Q[:, :, :-1, :]
        attn_scores_pre = torch.matmul(Q_prefill, K_bc.transpose(-2, -1)) * scale

        if mask is not None:
            attn_scores_pre = attn_scores_pre + mask

        attn_weights_pre = F.softmax(attn_scores_pre, dim=-1)
        attn_out_pre = torch.matmul(attn_weights_pre, V_bc)
        attn_out = torch.cat([attn_out_pre, attn_out], dim=2)

    return attn_out, page_indices


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchical — Token-level sparse attention kernel
# ═══════════════════════════════════════════════════════════════════════════════

def hierarchical_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    page_size: int,
    top_k: int,
    macro_multiplier: int = 3,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 64,
    adaptive_budget: bool = True,
    token_budget: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
    num_kv_groups: int = 1,
) -> Tuple[torch.Tensor, dict]:
    """Three-stage hierarchical token-level sparse attention — standalone kernel.

    Pipeline
    ========

      Stage 1 — Macro-Selection (Page-Level)
        Quest page scoring → Top-M pages (M larger than top_k, default 3×).
        Pages containing sink and recent tokens are force-included.

      Stage 2 — Micro-Selection (Token-Level)
        Every token inside the M selected pages receives an exact pre-softmax
        attention score ``(Q · K_token) / √d``. Padding tokens are masked
        with -inf.

      Stage 3 — Consolidation
        Sink tokens (first num_sink positions) and recent tokens (last
        num_recent positions) are force-protected.  Remaining budget is
        filled with the highest-scoring competitive tokens, yielding exactly
        B = top_k × page_size tokens.

      Sparse attention over the final Top-B token set.

    When T_q > 1, positions 0..T-2 use full attention and only the last
    position uses hierarchical sparse attention.

    Args:
        Q:                (B, num_heads, T_q, head_dim)
        K:                (B, num_kv_heads, T_kv, head_dim)
        V:                (B, num_kv_heads, T_kv, head_dim)
        page_size:        Tokens per page.
        top_k:            Reference page count — final B = top_k × page_size.
        macro_multiplier: M = multiplier × top_k macro pages.
        num_sink_tokens:  Force-protected initial tokens (StreamingLLM sinks).
        num_recent_tokens: Force-protected trailing tokens (recency bias).
        adaptive_budget:  Dynamically size M from page-score concentration.
        token_budget:     Explicit override for B (None → top_k × page_size).
        mask:             Optional causal mask for prefill fallback.
        num_kv_groups:    num_heads // num_kv_heads (1 for MHA, >1 for GQA).

    Returns:
        output:  (B, num_heads, T_q, head_dim)
        metrics: dict with —
            num_macro_pages, num_candidates, token_budget,
            page_scores_mean, page_scores_std
    """
    B, H_q, T_q, d = Q.shape
    _, H_kv, T_kv, _ = K.shape
    device = Q.device

    budget = token_budget if token_budget is not None else top_k * page_size
    kv_len = K.size(2)  # original KV length before padding

    # ---- Build pages ----
    K_paged, V_paged, pad_mask, num_pages = _build_pages(K, V, page_size)

    # ---- Page metadata ----
    K_min = K_paged.min(dim=3).values  # (B, H_kv, num_pages, d)
    K_max = K_paged.max(dim=3).values

    # ---- Decode query ----
    if T_q == 1:
        Q_decode = Q
    else:
        Q_decode = Q[:, :, -1:, :]

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Macro-Selection (Page-Level)
    # ═══════════════════════════════════════════════════════════════════════

    # GQA broadcast for page scoring
    if num_kv_groups > 1:
        K_min_bc = K_min.repeat_interleave(num_kv_groups, dim=1)
        K_max_bc = K_max.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_min_bc = K_min
        K_max_bc = K_max

    Q_exp = Q_decode.expand(-1, -1, num_pages, -1)
    prod_min = Q_exp * K_min_bc
    prod_max = Q_exp * K_max_bc
    page_scores = torch.max(prod_min, prod_max).sum(dim=-1)  # (B, H_q, num_pages)

    # Nominal M
    base_M = min(macro_multiplier * top_k, num_pages)

    # Adaptive sizing from page-score concentration
    if adaptive_budget and num_pages > 2:
        avg_scores = page_scores.mean(dim=(0, 1))
        sorted_s, _ = avg_scores.sort(descending=True)
        top2_sum = sorted_s[:2].sum()
        total_sum = sorted_s.sum() + 1e-8
        concentration = (top2_sum / total_sum).item()
        if concentration > 0.5:
            M = max(base_M // 2, 2)
        elif concentration < 0.3:
            M = min(base_M * 2, num_pages)
        else:
            M = base_M
    else:
        M = base_M
    M = max(M, top_k)
    M = min(M, num_pages)

    # ---- Macro page selection with sink/recent protection ----
    # Identify protected pages
    protected_pages_set: set[int] = set()
    if num_sink_tokens > 0:
        protected_pages_set.add(0)
    if num_recent_tokens > 0:
        recent_start_page = max(0, (kv_len - num_recent_tokens) // page_size)
        for p in range(recent_start_page, num_pages):
            protected_pages_set.add(p)
    protected_pages = sorted(protected_pages_set)

    _, macro_page_indices = torch.topk(
        page_scores, k=min(M, num_pages), dim=-1
    )  # (B, H_q, M)

    # Force-include protected pages
    for fp in protected_pages:
        fp_tensor = torch.full(
            (B, H_q, 1), fp, device=device, dtype=torch.long
        )
        already_present = (macro_page_indices == fp_tensor).any(dim=-1)
        needs_replace = ~already_present

        if needs_replace.any():
            sel_scores = page_scores.gather(dim=-1, index=macro_page_indices)
            is_prot = torch.zeros_like(macro_page_indices, dtype=torch.bool)
            for pp in protected_pages:
                is_prot = is_prot | (macro_page_indices == pp)
            sel_scores_masked = sel_scores.masked_fill(is_prot, float("inf"))
            _, min_idx = sel_scores_masked.min(dim=-1, keepdim=True)
            existing_val = macro_page_indices.gather(dim=-1, index=min_idx)
            replacement = torch.where(
                needs_replace.unsqueeze(-1),
                fp_tensor.expand_as(min_idx),
                existing_val,
            )
            macro_page_indices.scatter_(dim=-1, index=min_idx, src=replacement)

    M_actual = macro_page_indices.size(-1)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Micro-Selection (Token-Level Refinement)
    # ═══════════════════════════════════════════════════════════════════════

    # GQA broadcast for K_paged
    if num_kv_groups > 1:
        K_pg = K_paged.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_pg = K_paged
    # pad_mask: (B, 1, ...) → (B, H_q, ...) — always broadcast to query heads
    pm = pad_mask.expand(-1, H_q, -1, -1, -1)

    # Gather selected pages → flatten tokens
    idx_k = macro_page_indices.view(B, H_q, M_actual, 1, 1).expand(
        -1, -1, -1, page_size, d
    )
    K_sel = K_pg.gather(dim=2, index=idx_k).reshape(
        B, H_q, M_actual * page_size, d
    )

    # Per-token exact attention scores
    scale = 1.0 / math.sqrt(d)
    token_scores = (
        torch.matmul(Q_decode, K_sel.transpose(-2, -1)).squeeze(-2) * scale
    )  # (B, H_q, M_actual * page_size)

    # Validity mask for flattened tokens
    idx_m = macro_page_indices.view(B, H_q, M_actual, 1, 1).expand(
        -1, -1, -1, page_size, 1
    )
    validity = (
        pm.gather(dim=2, index=idx_m)
        .reshape(B, H_q, M_actual * page_size)
        .bool()
    )

    # Mask padding tokens: set score to -inf
    token_scores = token_scores.masked_fill(~validity, float("-inf"))

    # Global position mapping
    base_positions = macro_page_indices * page_size  # (B, H_q, M_actual)
    offsets = torch.arange(page_size, device=device).view(1, 1, 1, page_size)
    global_pos = (base_positions.unsqueeze(-1) + offsets).reshape(
        B, H_q, M_actual * page_size
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Consolidation (Sink/Recent protection + Top-B)
    # ═══════════════════════════════════════════════════════════════════════

    eff_sink = min(num_sink_tokens, kv_len)
    eff_recent = min(num_recent_tokens, kv_len)

    is_sink = global_pos < eff_sink
    is_recent = global_pos >= (kv_len - eff_recent)
    is_protected = (is_sink | is_recent) & validity

    boosted = token_scores.clone()
    if is_protected.sum() > budget * B * H_q:
        # Edge case: more tokens need protection than budget allows.
        # Tier recent > sink > competitive with large finite values
        # (IEEE 754 inf / 2 == inf, which defeats the ordering).
        boosted[is_recent & validity] = 1e10
        boosted[is_sink & validity & ~is_recent] = 1e9
    else:
        boosted[is_protected] = float("inf")

    effective_B = min(budget, token_scores.size(-1))
    _, topk_indices = torch.topk(boosted, k=effective_B, dim=-1)

    selected_pos = global_pos.gather(dim=-1, index=topk_indices)
    selected_valid = validity.gather(dim=-1, index=topk_indices)

    # ═══════════════════════════════════════════════════════════════════════
    # Sparse attention on selected Top-B tokens
    # ═══════════════════════════════════════════════════════════════════════

    # Pad K/V to page boundary for position-based gathering
    if T_kv % page_size != 0:
        pad_len = page_size - (T_kv % page_size)
        K_padded = F.pad(K, (0, 0, 0, pad_len))
        V_padded = F.pad(V, (0, 0, 0, pad_len))
    else:
        K_padded = K
        V_padded = V

    # GQA broadcast for padded K/V
    if num_kv_groups > 1:
        K_bc = K_padded.repeat_interleave(num_kv_groups, dim=1)
        V_bc = V_padded.repeat_interleave(num_kv_groups, dim=1)
    else:
        K_bc = K_padded
        V_bc = V_padded

    # Gather tokens by global position
    idx_g = selected_pos.unsqueeze(-1).expand(-1, -1, -1, d).long()
    K_tk = K_bc.gather(dim=2, index=idx_g)
    V_tk = V_bc.gather(dim=2, index=idx_g)

    attn_scores = torch.matmul(Q_decode, K_tk.transpose(-2, -1)) * scale
    attn_scores = attn_scores.masked_fill(
        ~selected_valid.unsqueeze(-2), float("-inf")
    )
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_weights, V_tk)  # (B, H_q, 1, d)

    # ═══════════════════════════════════════════════════════════════════════
    # Multi-token prefill: full attention for positions 0..T-2
    # ═══════════════════════════════════════════════════════════════════════
    if T_q > 1:
        K_pre = K
        V_pre = V
        if num_kv_groups > 1:
            K_pre = K.repeat_interleave(num_kv_groups, dim=1)
            V_pre = V.repeat_interleave(num_kv_groups, dim=1)

        Q_prefill = Q[:, :, :-1, :]
        attn_scores_pre = torch.matmul(
            Q_prefill, K_pre.transpose(-2, -1)
        ) * scale

        if mask is not None:
            attn_scores_pre = attn_scores_pre + mask

        attn_weights_pre = F.softmax(attn_scores_pre, dim=-1)
        attn_out_pre = torch.matmul(attn_weights_pre, V_pre)
        attn_out = torch.cat([attn_out_pre, attn_out], dim=2)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = {
        "num_macro_pages": M,
        "num_candidates": token_scores.size(-1),
        "token_budget": budget,
        "page_scores_mean": page_scores.mean().item(),
        "page_scores_std": page_scores.std().item(),
    }

    return attn_out, metrics
