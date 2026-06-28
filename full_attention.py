"""
Full (dense) scaled dot-product attention — the baseline.

Implements the standard formulation:

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Supports both single-head and multi-head variants with optional GQA
(Grouped-Query Attention) broadcasting.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single-head scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Compute full scaled dot-product attention.

    Args:
        Q:  Query tensor  of shape (B, H, T_q, d_k) or (B, T_q, d_k).
        K:  Key tensor    of shape (B, H, T_kv, d_k) or (B, T_kv, d_k).
        V:  Value tensor  of shape (B, H, T_kv, d_k) or (B, T_kv, d_k).
        mask: Optional additive mask (B, 1, T_q, T_kv) — -inf entries are masked.
        dropout_p: Dropout probability on attention weights (0 = no dropout).

    Returns:
        output: Attention output, same shape as Q.
    """
    d_k = Q.size(-1)
    scale = 1.0 / math.sqrt(d_k)

    # Q @ K^T — shape: (..., T_q, T_kv)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if mask is not None:
        attn_scores = attn_scores + mask

    attn_weights = F.softmax(attn_scores, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    output = torch.matmul(attn_weights, V)
    return output


# ---------------------------------------------------------------------------
# Multi-head full attention (the baseline)
# ---------------------------------------------------------------------------

class MultiHeadFullAttention(nn.Module):
    """Standard multi-head attention with optional GQA support.

    Shapes follow the Llama-2 convention:
      - Input:  (B, T, hidden_dim)
      - Output: (B, T, hidden_dim)
      - Projected Q: (B, num_heads,    T, head_dim)
      - Projected K: (B, num_kv_heads, T, head_dim)
      - Projected V: (B, num_kv_heads, T, head_dim)

    When num_kv_heads < num_heads, K and V are broadcast (GQA).
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        num_kv_heads: int = 32,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads  # >1 only for GQA

        # Combined QKV projection (common in efficient implementations)
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_dim, bias=False)

        self.dropout_p = dropout_p

    def _reshape_for_attention(
        self, x: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Reshape (B, T, num_heads * head_dim) → (B, num_heads, T, head_dim)."""
        B, T, _ = x.shape
        return x.view(B, T, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute full multi-head attention.

        Args:
            hidden_states: (B, T, hidden_dim)
            mask: Optional causal/additive mask.
            return_kv: If True, also return K and V projections (for caching).

        Returns:
            output: (B, T, hidden_dim), or (output, K, V) if return_kv.
        """
        B, T, _ = hidden_states.shape

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        Q = self._reshape_for_attention(Q, self.num_heads)
        K = self._reshape_for_attention(K, self.num_kv_heads)
        V = self._reshape_for_attention(V, self.num_kv_heads)

        # Broadcast KV for GQA if num_kv_heads < num_heads
        if self.num_kv_groups > 1:
            K = K.repeat_interleave(self.num_kv_groups, dim=1)
            V = V.repeat_interleave(self.num_kv_groups, dim=1)

        # Full attention: (B, H, T, head_dim)
        attn_out = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout_p
        )

        # Merge heads back: (B, H, T, d) → (B, T, H*d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

        output = self.o_proj(attn_out)

        if return_kv:
            return output, K, V
        return output
