"""
Quest: Page-wise Sparse Attention for Long-Context LLM Serving.

Reference implementation of the Quest algorithm.  Divides the KV cache into
fixed-size pages, precomputes per-page min/max key metadata, and at each
decode step selects only the Top-K pages for attention computation.

Pipeline (per the Quest paper):
  1. **Page metadata init** – element-wise min & max of keys per page.
  2. **Stage 1 – Estimate critical pages**:
     a. Element-wise product of current query with reduced keys (min, max).
     b. Per-channel max across the two products.
     c. Sum to produce a single scalar score per page.
  3. **Stage 2 – Sparse attention** over the Top-K pages only.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Page-building helper
# ---------------------------------------------------------------------------

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
        pad_mask:  (1, 1, num_pages, page_size)  — True for valid tokens.
        num_pages: int
    """
    B, H, T, d = K.shape

    # Create validity mask BEFORE padding: shape (B, 1, T, 1)
    valid_mask = torch.ones(B, 1, T, 1, device=K.device, dtype=K.dtype)

    # Pad sequence length to be a multiple of page_size
    if T % page_size != 0:
        pad_len = page_size - (T % page_size)
        K = F.pad(K, (0, 0, 0, pad_len))
        V = F.pad(V, (0, 0, 0, pad_len))
        valid_mask = F.pad(valid_mask, (0, 0, 0, pad_len))  # pads with zeros
        T_padded = T + pad_len
    else:
        T_padded = T

    num_pages = T_padded // page_size

    K_paged = K.view(B, H, num_pages, page_size, d)
    V_paged = V.view(B, H, num_pages, page_size, d)
    pad_mask = valid_mask.view(B, 1, num_pages, page_size, 1).bool()

    return K_paged, V_paged, pad_mask, num_pages


# ---------------------------------------------------------------------------
# Quest attention — Module form
# ---------------------------------------------------------------------------

class QuestAttention(nn.Module):
    """Quest page-wise sparse attention module.

    Compatible with the same interface as MultiHeadFullAttention so the
    experiment harness can swap them transparently.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads:  Number of query heads.
        head_dim:   Dimension per head.
        num_kv_heads: Number of KV heads (for GQA support).
        page_size:  Tokens per page.
        top_k:      Number of pages selected for sparse attention.
        dropout_p:  Attention dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        num_kv_heads: int = 32,
        page_size: int = 64,
        top_k: int = 4,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.page_size = page_size
        self.top_k = top_k
        self.dropout_p = dropout_p

        # Linear projections
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_dim, bias=False)

    # ------------------------------------------------------------------
    # Step 1 — Page metadata (element-wise min & max of keys per page)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_page_metadata(
        K_paged: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute element-wise min and max for each page of keys.

        Args:
            K_paged: (B, num_kv_heads, num_pages, page_size, head_dim)

        Returns:
            K_min: (B, num_kv_heads, num_pages, head_dim)
            K_max: (B, num_kv_heads, num_pages, head_dim)
        """
        # Reduce over the page_size dimension (dim=3)
        K_min = K_paged.min(dim=3).values   # (B, H_kv, num_pages, d)
        K_max = K_paged.max(dim=3).values
        return K_min, K_max

    # ------------------------------------------------------------------
    # Step 2 — Stage 1: Estimate critical pages
    # ------------------------------------------------------------------
    @staticmethod
    def score_pages(
        Q: torch.Tensor,
        K_min: torch.Tensor,
        K_max: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a query-aware scalar score for each page.

        Exact Quest scoring pipeline:
          1. Element-wise product:  Q ⊙ K_min  and  Q ⊙ K_max
          2. Per-channel max:       max(prod_min, prod_max)
          3. Sum over channels:     sum over head_dim

        Args:
            Q:     Query tensor (B, num_heads, 1, head_dim) — single decode token.
            K_min: Per-page key minima (B, num_kv_heads, num_pages, head_dim).
            K_max: Per-page key maxima (B, num_kv_heads, num_pages, head_dim).

        Returns:
            scores: (B, num_heads, num_pages) — scalar score per head per page.
        """
        # Expand Q for element-wise product with every page
        # Q:     (B, H_q, 1, d)  → (B, H_q, num_pages, d)
        # K_min: (B, H_kv, num_pages, d) — may need broadcasting for GQA

        num_pages = K_min.size(2)

        # If GQA: broadcast KV heads to match query heads
        if K_min.size(1) != Q.size(1):
            # K_min: (B, H_kv, num_pages, d) → (B, H_q, num_pages, d)
            repeat = Q.size(1) // K_min.size(1)
            K_min = K_min.repeat_interleave(repeat, dim=1)
            K_max = K_max.repeat_interleave(repeat, dim=1)

        Q_expanded = Q.expand(-1, -1, num_pages, -1)

        # (a) Element-wise product
        prod_min = Q_expanded * K_min   # (B, H, num_pages, d)
        prod_max = Q_expanded * K_max   # (B, H, num_pages, d)

        # (b) Per-channel max  — max over the two products
        combined = torch.max(prod_min, prod_max)  # (B, H, num_pages, d)

        # (c) Sum over head_dim to produce a scalar per page
        scores = combined.sum(dim=-1)   # (B, H, num_pages)

        return scores

    # ------------------------------------------------------------------
    # Step 3 — Stage 2: Sparse attention on Top-K pages
    # ------------------------------------------------------------------
    def sparse_attention_on_pages(
        self,
        Q: torch.Tensor,
        K_paged: torch.Tensor,
        V_paged: torch.Tensor,
        pad_mask: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention restricted to selected pages.

        Args:
            Q:            (B, num_heads, 1, head_dim)
            K_paged:      (B, num_kv_heads, num_pages, page_size, head_dim)
            V_paged:      (B, num_kv_heads, num_pages, page_size, head_dim)
            pad_mask:     (B, 1, num_pages, page_size, 1) — True = valid token.
            page_indices: (B, num_heads, top_k) — selected page indices.

        Returns:
            output: (B, num_heads, 1, head_dim)
        """
        B, H_q, _, d = Q.shape
        top_k = page_indices.size(-1)
        page_size = K_paged.size(3)

        # Handle GQA broadcasting for K_paged and V_paged
        if K_paged.size(1) != H_q:
            repeat = H_q // K_paged.size(1)
            K_paged = K_paged.repeat_interleave(repeat, dim=1)
            V_paged = V_paged.repeat_interleave(repeat, dim=1)

        # Broadcast pad_mask to match query heads if needed
        if pad_mask.size(1) != H_q:
            repeat = H_q // pad_mask.size(1)
            pad_mask = pad_mask.repeat_interleave(repeat, dim=1)

        # --- Gather selected pages ---
        idx = page_indices.view(B, H_q, top_k, 1, 1).expand(
            -1, -1, -1, page_size, d
        )
        K_selected = K_paged.gather(dim=2, index=idx)   # (B, H, top_k, page_size, d)
        V_selected = V_paged.gather(dim=2, index=idx)

        # Gather corresponding mask entries
        # pad_mask: (B, H, num_pages, page_size, 1)
        idx_mask = page_indices.view(B, H_q, top_k, 1, 1).expand(
            -1, -1, -1, page_size, 1
        )
        mask_selected = pad_mask.gather(dim=2, index=idx_mask)  # (B, H, top_k, page_size, 1)

        # Flatten selected pages into one sequence
        K_sparse = K_selected.reshape(B, H_q, top_k * page_size, d)
        V_sparse = V_selected.reshape(B, H_q, top_k * page_size, d)
        mask_flat = mask_selected.reshape(B, H_q, 1, top_k * page_size)  # (B, H, 1, S)

        # Scaled dot-product attention over the sparse set
        scale = 1.0 / math.sqrt(d)
        attn_scores = torch.matmul(Q, K_sparse.transpose(-2, -1)) * scale

        # Apply padding mask: set padding positions to -inf before softmax
        attn_scores = attn_scores.masked_fill(~mask_flat, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=True)

        output = torch.matmul(attn_weights, V_sparse)  # (B, H_q, 1, d)
        return output

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        *,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quest sparse attention forward pass.

        Args:
            hidden_states: (B, T, hidden_dim).  For decode, T == 1.
            mask: Ignored (sparse attention operates on selected pages).
            return_kv: If True, also return K, V projections.

        Returns:
            output: (B, T, hidden_dim) or (output, K, V) if return_kv.
        """
        B, T, _ = hidden_states.shape

        # Project to Q, K, V
        Q_full = self.q_proj(hidden_states)
        K_full = self.k_proj(hidden_states)
        V_full = self.v_proj(hidden_states)

        # Reshape: (B, T, H*d) → (B, H, T, d)
        Q = Q_full.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K_full.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V_full.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # --- Build pages ---
        K_paged, V_paged, pad_mask, num_pages = _build_pages(K, V, self.page_size)

        # --- Step 1: Page metadata (min/max) ---
        K_min, K_max = self.compute_page_metadata(K_paged)

        # For decode, we take the last (or only) query token
        # Q_decode: (B, num_heads, 1, head_dim)
        if T == 1:
            Q_decode = Q
        else:
            Q_decode = Q[:, :, -1:, :]

        # --- Step 2 (Stage 1): Score pages ---
        scores = self.score_pages(Q_decode, K_min, K_max)  # (B, H, num_pages)

        # Select top-K pages
        effective_k = min(self.top_k, num_pages)
        _, page_indices = torch.topk(scores, k=effective_k, dim=-1)  # (B, H, top_k)

        # --- Step 3 (Stage 2): Sparse attention ---
        attn_out = self.sparse_attention_on_pages(
            Q_decode, K_paged, V_paged, pad_mask, page_indices
        )  # (B, H, 1, d)

        # Merge heads: (B, H, 1, d) → (B, 1, H*d) → handle T > 1 case
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, -1)

        # If input had multiple tokens, only the last position gets Quest output;
        # for a fair prefill comparison we compute full attention on earlier positions.
        if T > 1:
            # Broadcast KV for GQA if needed, so K/V match the query heads dim.
            K_bc = K
            V_bc = V
            if self.num_kv_groups > 1:
                K_bc = K.repeat_interleave(self.num_kv_groups, dim=1)
                V_bc = V.repeat_interleave(self.num_kv_groups, dim=1)

            # For the prefill phase (positions 0..T-2), run full attention
            # and only replace the last token with Quest's sparse output.
            # This matches how Quest operates in practice: full prefill, sparse decode.
            Q_prefill = Q[:, :, :-1, :]  # (B, H, T-1, d)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores_pre = torch.matmul(Q_prefill, K_bc.transpose(-2, -1)) * scale
            attn_weights_pre = F.softmax(attn_scores_pre, dim=-1)
            attn_out_pre = torch.matmul(attn_weights_pre, V_bc)
            attn_out_pre = (
                attn_out_pre.transpose(1, 2).contiguous().view(B, T - 1, -1)
            )
            attn_out = torch.cat([attn_out_pre, attn_out], dim=1)

        # Output projection
        output = self.o_proj(attn_out)

        if return_kv:
            return output, K, V
        return output
