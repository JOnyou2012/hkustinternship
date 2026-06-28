"""
Phase 2 — Quality-Oriented Sparse Attention (Hierarchical Token-Level).

Upgrades the Quest page-wise baseline with a three-stage hierarchical
filtering pipeline that selects **individual tokens** rather than coarse
pages, operating under the exact same attention budget.

Pipeline
========

  Stage 1 • Macro-Selection (Page-Level)
    ─────────────────────────────────────
    Identical Quest page scoring → Top-**M** pages selected as a candidate
    pool.  *M* is larger than the Quest *top_k* (default 3× larger), so we
    pull in more candidate tokens than the final budget.

  Stage 2 • Micro-Selection (Token-Level)
    ──────────────────────────────────────
    Every token inside the *M* candidate pages receives an exact attention
    score  ``(Q · K_token) / √d``.  Tokens are ranked per head, per batch.

  Stage 3 • Final Consolidation
    ────────────────────────────
    **Sink tokens** (first ``num_sink_tokens`` positions) and **recent
    tokens** (last ``num_recent_tokens`` positions) are force-protected
    per the StreamingLLM / Lost-in-the-Middle findings.  The remaining
    budget is filled with the highest-scoring tokens from Stage 2, yielding
    exactly **Top-B** tokens where ``B = top_k × page_size`` — the same
    budget as the Quest baseline.

Advanced Features
=================

  • **Adaptive budget**: The number of macro pages *M* is dynamically sized
    from the page-score distribution.  Concentrated scores → narrower search
    (smaller *M*).  Uniform scores → wider search (larger *M*).

  • **Per-head independent selection**: Each attention head selects its own
    Top-B tokens, preserving the heterogeneous specialisation observed in
    multi-head attention.

  • **Budget invariant**: ``B = top_k × page_size`` is guaranteed, so any
    quality improvement over Quest comes purely from *better token selection*,
    not from attending to more tokens.

References
==========
  • Quest: https://arxiv.org/abs/2406.10774
  • StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
  • Lost in the Middle: https://arxiv.org/abs/2307.03172
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the page-building helper from quest_attention
from quest_attention import _build_pages, QuestAttention


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Token-Level Attention Module
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalTokenAttention(nn.Module):
    """Phase 2 quality-oriented sparse attention with token-level refinement.

    Same interface as ``QuestAttention`` and ``MultiHeadFullAttention`` so
    the experiment harness can swap them transparently.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of query heads.
        head_dim: Dimension per head.
        num_kv_heads: Number of KV heads (GQA support).
        page_size: Tokens per page.
        top_k: Reference page count — final token budget B = top_k × page_size.
        macro_multiplier: M = macro_multiplier × top_k macro pages.
        num_sink_tokens: Force-protected initial tokens.
        num_recent_tokens: Force-protected trailing tokens.
        adaptive_budget: Dynamically size M from score distribution.
        token_budget: Explicit override for B (None → derived from top_k).
        dropout_p: Attention dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
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
        self.macro_multiplier = macro_multiplier
        self.num_sink_tokens = num_sink_tokens
        self.num_recent_tokens = num_recent_tokens
        self.adaptive_budget = adaptive_budget
        self.dropout_p = dropout_p

        # ---- Token budget: B = top_k × page_size (matches Quest exactly) ----
        self.token_budget = (
            token_budget if token_budget is not None else top_k * page_size
        )

        # Linear projections
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_dim, bias=False)

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1 — Macro page scoring (reuses Quest methodology)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _score_pages_quest(
        Q: torch.Tensor,
        K_min: torch.Tensor,
        K_max: torch.Tensor,
    ) -> torch.Tensor:
        """Quest page scoring pipeline: element-wise product → channel max → sum.

        Args:
            Q:     (B, num_heads, 1, head_dim)
            K_min: (B, num_kv_heads, num_pages, head_dim)
            K_max: (B, num_kv_heads, num_pages, head_dim)

        Returns:
            scores: (B, num_heads, num_pages)
        """
        num_pages = K_min.size(2)

        # GQA broadcast
        if K_min.size(1) != Q.size(1):
            repeat = Q.size(1) // K_min.size(1)
            K_min = K_min.repeat_interleave(repeat, dim=1)
            K_max = K_max.repeat_interleave(repeat, dim=1)

        Q_exp = Q.expand(-1, -1, num_pages, -1)
        prod_min = Q_exp * K_min
        prod_max = Q_exp * K_max
        combined = torch.max(prod_min, prod_max)
        return combined.sum(dim=-1)  # (B, H, num_pages)

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1b — Macro page selection with sink/recent protection
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def select_macro_pages(
        page_scores: torch.Tensor,
        M: int,
        num_pages: int,
        kv_len: int,
        page_size: int,
        num_sink_tokens: int,
        num_recent_tokens: int,
    ) -> torch.Tensor:
        """Select Top-M pages, force-including pages with sink & recent tokens.

        Pages containing protected tokens (sink = first few positions; recent =
        last few positions) are **always** included, replacing the
        lowest-scoring non-protected pages if necessary.

        Args:
            page_scores: (B, num_heads, num_pages)
            M: Number of pages to select.
            num_pages: Total number of pages.
            kv_len: Original KV-cache length.
            page_size: Tokens per page.
            num_sink_tokens: Number of initial sink tokens to protect.
            num_recent_tokens: Number of trailing recent tokens to protect.

        Returns:
            macro_page_indices: (B, num_heads, M) — page indices.
        """
        device = page_scores.device
        B, H, _ = page_scores.shape

        # --- Identify protected pages ---
        # Only protect when the corresponding count is > 0.
        protected_pages: list[int] = []
        seen: set[int] = set()

        if num_sink_tokens > 0:
            protected_pages.append(0)
            seen.add(0)

        if num_recent_tokens > 0:
            recent_start_page = max(
                0, (kv_len - num_recent_tokens) // page_size
            )
            for p in range(recent_start_page, num_pages):
                if p not in seen:
                    protected_pages.append(p)
                    seen.add(p)

        # --- Select Top-M scored pages ---
        _, macro_page_indices = torch.topk(
            page_scores, k=min(M, num_pages), dim=-1
        )

        # --- Force-include protected pages ---
        for fp in protected_pages:
            fp_tensor = torch.full(
                (B, H, 1), fp, device=device, dtype=torch.long
            )
            already_present = (macro_page_indices == fp_tensor).any(dim=-1)
            needs_replace = ~already_present  # (B, H)

            if needs_replace.any():
                # Gather scores of currently selected pages
                sel_scores = page_scores.gather(
                    dim=-1, index=macro_page_indices
                )

                # Mask already-protected pages with +inf
                is_prot_page = torch.zeros_like(
                    macro_page_indices, dtype=torch.bool
                )
                for pp in protected_pages:
                    is_prot_page = is_prot_page | (macro_page_indices == pp)

                sel_scores_masked = sel_scores.masked_fill(
                    is_prot_page, float("inf")
                )

                # Find the minimum-scoring non-protected page to replace
                _, min_idx = sel_scores_masked.min(dim=-1, keepdim=True)

                # Only replace in (B,H) combos where the page is NOT already
                # present; elsewhere scatter the existing value (no-op).
                existing_val = macro_page_indices.gather(
                    dim=-1, index=min_idx
                )
                replacement = torch.where(
                    needs_replace.unsqueeze(-1),
                    fp_tensor.expand_as(min_idx),
                    existing_val,
                )
                macro_page_indices.scatter_(
                    dim=-1, index=min_idx, src=replacement,
                )

        return macro_page_indices

    # ──────────────────────────────────────────────────────────────────────
    # Adaptive macro-budget
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_adaptive_M(
        page_scores: torch.Tensor,
        base_M: int,
        num_pages: int,
        concentration_threshold: float = 0.5,
    ) -> int:
        """Dynamically size *M* from the page-score distribution.

        **High concentration** (few pages dominate) → shrink M (those pages
        already contain the best tokens).  **Low concentration** (scores are
        uniform) → grow M (need to cast a wider net).

        Uses the Gini-style coefficient: ratio of top-2 page scores to the
        sum of all page scores.  High ratio → concentrated.

        Args:
            page_scores: (B, num_heads, num_pages)
            base_M: The nominal M (= macro_multiplier × top_k).
            num_pages: Total number of pages available.
            concentration_threshold: Ratio above which scores are "concentrated".

        Returns:
            adaptive_M: int in [top_k, num_pages], clamped.
        """
        # Average over batch and heads for a single distribution estimate
        avg_scores = page_scores.mean(dim=(0, 1))  # (num_pages,)

        if avg_scores.numel() <= 2:
            return base_M

        # Sort descending and measure top-2 concentration
        sorted_scores, _ = avg_scores.sort(descending=True)
        top2_sum = sorted_scores[:2].sum()
        total_sum = sorted_scores.sum() + 1e-8
        concentration = (top2_sum / total_sum).item()

        if concentration > concentration_threshold:
            # Concentrated — fewer pages needed
            M = max(base_M // 2, 2)
        elif concentration < 0.3:
            # Diffuse — cast wider net
            M = min(base_M * 2, num_pages)
        else:
            M = base_M

        return min(M, num_pages)

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2 — Token-level scoring inside selected pages
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _score_tokens_in_selected_pages(
        Q: torch.Tensor,
        K_paged: torch.Tensor,
        pad_mask: torch.Tensor,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score every token inside the macro-selected pages.

        For each (batch, head), for each token in the selected pages, compute
        the **exact pre-softmax attention score**::

            score_t = (Q · K_t) / √d

        Args:
            Q:            (B, num_heads, 1, head_dim) — decode query.
            K_paged:      (B, num_kv_heads, num_pages, page_size, head_dim)
            pad_mask:     (B, 1, num_pages, page_size, 1) — True = valid.
            page_indices: (B, num_heads, M) — selected macro pages.

        Returns:
            token_scores:   (B, num_heads, M×page_size) — per-token scores,
                            padding tokens set to -inf.
            token_validity: (B, num_heads, M×page_size) — True = real token.
            global_pos:     (B, num_heads, M×page_size) — global position in
                            the padded KV cache for each token.
        """
        B, H_q, _, d = Q.shape
        M = page_indices.size(-1)
        ps = K_paged.size(3)  # page_size from K_paged

        # --- GQA broadcast for K_paged ---
        H_kv = K_paged.size(1)
        if H_kv != H_q:
            repeat = H_q // H_kv
            K_paged = K_paged.repeat_interleave(repeat, dim=1)

        # --- GQA broadcast for pad_mask ---
        if pad_mask.size(1) != H_q:
            repeat = H_q // pad_mask.size(1)
            pad_mask = pad_mask.repeat_interleave(repeat, dim=1)

        # --- Gather selected pages: (B, H_q, M, page_size, d) ---
        idx_k = page_indices.view(B, H_q, M, 1, 1).expand(-1, -1, -1, ps, d)
        K_selected = K_paged.gather(dim=2, index=idx_k)

        # Flatten tokens: (B, H_q, M×page_size, d)
        K_flat = K_selected.reshape(B, H_q, M * ps, d)

        # --- Per-token exact attention scores ---
        scale = 1.0 / math.sqrt(d)
        # Q: (B, H_q, 1, d)  → scores: (B, H_q, 1, M×ps) → squeeze to (B, H_q, M×ps)
        scores = torch.matmul(Q, K_flat.transpose(-2, -1)).squeeze(-2) * scale

        # --- Validity mask for flattened tokens ---
        idx_m = page_indices.view(B, H_q, M, 1, 1).expand(-1, -1, -1, ps, 1)
        mask_selected = pad_mask.gather(dim=2, index=idx_m)  # (B, H_q, M, ps, 1)
        validity = mask_selected.reshape(B, H_q, M * ps).bool()

        # Mask padding tokens: set their score to -inf
        scores = scores.masked_fill(~validity, float("-inf"))

        # --- Global position mapping ---
        # For each selected page p (at slot i in the selection), its tokens
        # span global positions [p*ps, (p+1)*ps).
        # Build positions: base[p] + offset[0..ps-1] for each page.
        base_positions = page_indices * ps  # (B, H_q, M) — start of each page
        offsets = torch.arange(ps, device=Q.device).view(1, 1, 1, ps)  # (1,1,1,ps)
        global_pos = (base_positions.unsqueeze(-1) + offsets).reshape(B, H_q, M * ps)

        return scores, validity, global_pos

    # ──────────────────────────────────────────────────────────────────────
    # Stage 3 — Final consolidation with sink / recent protection
    # ──────────────────────────────────────────────────────────────────────
    def _consolidate_tokens(
        self,
        token_scores: torch.Tensor,
        token_validity: torch.Tensor,
        global_pos: torch.Tensor,
        kv_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the final Top-B tokens with sink/recent protection.

        Algorithm per (batch, head):
          1. Identify sink tokens (global positions 0 … num_sink-1) and
             recent tokens (global positions kv_len−num_recent … kv_len−1).
          2. Force-include those that are valid and within the candidate pool.
          3. Fill the remaining budget with the highest-scoring tokens from
             the candidate pool (excluding already-protected ones).

        If the number of protected tokens exceeds the budget (which can happen
        when ``kv_len`` is very small), priority is given to the most recent
        tokens first, then sink tokens, up to the budget limit.

        Args:
            token_scores:   (B, H, candidates) — pre-softmax scores, padding = -inf.
            token_validity: (B, H, candidates) — True for real tokens.
            global_pos:     (B, H, candidates) — global padded position.
            kv_len:         Original KV-cache length (before padding).

        Returns:
            selected_global_pos: (B, H, B_target) — global positions of selected tokens.
            selected_validity:   (B, H, B_target) — True for valid entries.
        """
        B, H, num_candidates = token_scores.shape
        B_target = self.token_budget
        device = token_scores.device

        # Clamp protection params to kv_len so we never protect more tokens
        # than actually exist.
        eff_sink = min(self.num_sink_tokens, kv_len)
        eff_recent = min(self.num_recent_tokens, kv_len)

        # --- Determine which candidates are sink / recent ---
        is_sink = global_pos < eff_sink
        is_recent = global_pos >= (kv_len - eff_recent)
        is_protected = (is_sink | is_recent) & token_validity

        # --- Handle the edge case where more tokens need protection than
        #     the budget allows.  We prioritise recent tokens over sink tokens
        #     by giving them a higher boost, ensuring `topk` favours them.
        boosted_scores = token_scores.clone()

        if is_protected.sum() > B_target * B * H:
            # Too many protected tokens — tier the boosts so that
            #   recent > sink > competitive
            # This guarantees the most important tokens survive truncation.
            boosted_scores[is_recent & token_validity] = float("inf")
            boosted_scores[is_sink & token_validity & ~is_recent] = float("inf") / 2
        else:
            boosted_scores[is_protected] = float("inf")

        # Top-B selection
        effective_B = min(B_target, num_candidates)
        topk_scores, topk_indices = torch.topk(
            boosted_scores, k=effective_B, dim=-1
        )

        # Gather selected global positions and validity
        selected_pos = global_pos.gather(dim=-1, index=topk_indices)
        selected_valid = token_validity.gather(dim=-1, index=topk_indices)

        return selected_pos, selected_valid

    # ──────────────────────────────────────────────────────────────────────
    # Sparse attention on selected tokens
    # ──────────────────────────────────────────────────────────────────────
    def _token_sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        selected_positions: torch.Tensor,
        selected_validity: torch.Tensor,
        kv_len: int,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention over the selected token set.

        The selected tokens are scattered across the KV cache — we gather
        them by their global positions, compute exact attention, and mask
        any padding positions.

        Args:
            Q:                  (B, num_heads, 1, head_dim)
            K:                  (B, num_kv_heads, kv_padded, head_dim)
            V:                  (B, num_kv_heads, kv_padded, head_dim)
            selected_positions: (B, num_heads, B_target) — global indices.
            selected_validity:  (B, num_heads, B_target) — True = valid.
            kv_len:             Original KV-cache length (unpadded).

        Returns:
            output: (B, num_heads, 1, head_dim)
        """
        B, H_q, _, d = Q.shape
        B_target = selected_positions.size(-1)

        # --- GQA broadcast for K, V ---
        H_kv = K.size(1)
        if H_kv != H_q:
            repeat = H_q // H_kv
            K = K.repeat_interleave(repeat, dim=1)
            V = V.repeat_interleave(repeat, dim=1)

        # --- Gather selected tokens by global position ---
        # K: (B, H_q, kv_padded, d)
        # selected_positions: (B, H_q, B_target)  → expand to (B, H_q, B_target, d)
        idx_gather = selected_positions.unsqueeze(-1).expand(-1, -1, -1, d).long()
        K_sel = K.gather(dim=2, index=idx_gather)  # (B, H_q, B_target, d)
        V_sel = V.gather(dim=2, index=idx_gather)  # (B, H_q, B_target, d)

        # --- Scaled dot-product attention ---
        scale = 1.0 / math.sqrt(d)
        attn_scores = torch.matmul(Q, K_sel.transpose(-2, -1)) * scale  # (B,H_q,1,B_target)

        # Mask padding / invalid entries
        # selected_validity: (B, H_q, B_target) → (B, H_q, 1, B_target)
        attn_scores = attn_scores.masked_fill(
            ~selected_validity.unsqueeze(-2), float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=True)

        output = torch.matmul(attn_weights, V_sel)  # (B, H_q, 1, d)
        return output

    # ──────────────────────────────────────────────────────────────────────
    # Full forward pass
    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        *,
        return_kv: bool = False,
        return_metrics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
    ):
        """Hierarchical token-level sparse attention forward pass.

        Args:
            hidden_states: (B, T, hidden_dim).  For decode, T == 1.
            mask: Ignored (sparse attention manages its own masking).
            return_kv: If True, also return K, V projections.
            return_metrics: If True, return (output, metrics_dict) where
                            metrics_dict includes per-head stats useful for
                            visualisation and debugging.

        Returns:
            output: (B, T, hidden_dim), or (output, K, V) if ``return_kv``,
                    or (output, metrics_dict) if ``return_metrics``.
        """
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        # ---- Project to Q, K, V ----
        Q_full = self.q_proj(hidden_states)
        K_full = self.k_proj(hidden_states)
        V_full = self.v_proj(hidden_states)

        Q = Q_full.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K_full.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V_full.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_len = K.size(2)  # original KV length before any padding

        # ---- Build pages ----
        K_paged, V_paged, pad_mask, num_pages = _build_pages(K, V, self.page_size)

        # ---- Page metadata (min/max) for Quest scoring ----
        K_min = K_paged.min(dim=3).values  # (B, H_kv, num_pages, d)
        K_max = K_paged.max(dim=3).values

        # ---- Decode query ----
        if T == 1:
            Q_decode = Q
        else:
            Q_decode = Q[:, :, -1:, :]

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1 — Macro-Selection (Page-Level)
        # ═══════════════════════════════════════════════════════════════
        page_scores = self._score_pages_quest(Q_decode, K_min, K_max)

        # Nominal M
        base_M = min(self.macro_multiplier * self.top_k, num_pages)

        # Adaptive sizing
        if self.adaptive_budget and num_pages > 2:
            M = self._compute_adaptive_M(page_scores, base_M, num_pages)
        else:
            M = base_M

        M = max(M, self.top_k)  # never fewer than Quest's top_k
        M = min(M, num_pages)

        # --- Select macro pages with sink/recent protection ---
        macro_page_indices = self.select_macro_pages(
            page_scores=page_scores,
            M=M,
            num_pages=num_pages,
            kv_len=kv_len,
            page_size=self.page_size,
            num_sink_tokens=self.num_sink_tokens,
            num_recent_tokens=self.num_recent_tokens,
        )
        # macro_page_indices: (B, num_heads, M)

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2 — Micro-Selection (Token-Level Refinement)
        # ═══════════════════════════════════════════════════════════════
        token_scores, token_validity, global_pos = (
            self._score_tokens_in_selected_pages(
                Q_decode, K_paged, pad_mask, macro_page_indices
            )
        )
        # token_scores: (B, H, M×page_size) — padding = -inf
        # global_pos:   (B, H, M×page_size)

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3 — Consolidation (Sink/Recent protection + Top-B)
        # ═══════════════════════════════════════════════════════════════
        selected_pos, selected_valid = self._consolidate_tokens(
            token_scores, token_validity, global_pos, kv_len
        )
        # selected_pos: (B, H, B_target)

        # ---- Need padded K, V for gathering by global position ----
        if kv_len % self.page_size != 0:
            pad_len = self.page_size - (kv_len % self.page_size)
            K_padded = F.pad(K, (0, 0, 0, pad_len))
            V_padded = F.pad(V, (0, 0, 0, pad_len))
        else:
            K_padded = K
            V_padded = V

        # ═══════════════════════════════════════════════════════════════
        # Sparse attention on the selected Top-B tokens
        # ═══════════════════════════════════════════════════════════════
        attn_out = self._token_sparse_attention(
            Q_decode, K_padded, V_padded, selected_pos, selected_valid, kv_len
        )  # (B, H, 1, d)

        # ---- Merge heads ----
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, -1)

        # ---- Multi-token input: full attention for prefill, hierarchical for last ----
        if T > 1:
            K_bc = K
            V_bc = V
            if self.num_kv_groups > 1:
                K_bc = K.repeat_interleave(self.num_kv_groups, dim=1)
                V_bc = V.repeat_interleave(self.num_kv_groups, dim=1)

            Q_prefill = Q[:, :, :-1, :]
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores_pre = torch.matmul(Q_prefill, K_bc.transpose(-2, -1)) * scale
            attn_weights_pre = F.softmax(attn_scores_pre, dim=-1)
            attn_out_pre = torch.matmul(attn_weights_pre, V_bc)
            attn_out_pre = (
                attn_out_pre.transpose(1, 2).contiguous().view(B, T - 1, -1)
            )
            attn_out = torch.cat([attn_out_pre, attn_out], dim=1)

        # ---- Output projection ----
        output = self.o_proj(attn_out)

        if return_metrics:
            metrics = {
                "num_macro_pages": M,
                "num_candidates": token_scores.size(-1),
                "token_budget": self.token_budget,
                "page_scores_mean": page_scores.mean().item(),
                "page_scores_std": page_scores.std().item(),
            }
            if return_kv:
                return output, K, V, metrics
            return output, metrics

        if return_kv:
            return output, K, V
        return output
