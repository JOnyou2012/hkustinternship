"""
Configuration dataclasses for the Quest Baseline Experiment.

Default dimensions follow the Llama-2 7B architecture.
All values can be overridden via command-line flags or by editing this file directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture parameters (Llama-2 7B defaults)."""

    hidden_dim: int = 4096       # d_model
    num_heads: int = 32          # total query heads
    head_dim: int = 128          # d_k = hidden_dim / num_heads
    num_kv_heads: int = 32       # KV heads (== num_heads for MHA; < num_heads for GQA)


@dataclass
class QuestConfig:
    """Quest algorithm hyperparameters.

    Attributes:
        page_size: Number of KV tokens per page / group.
        top_k: Number of pages selected in Stage 1 for sparse attention.
    """

    page_size: int = 64
    top_k: int = 4


@dataclass
class ExperimentConfig:
    """Experiment sweep and benchmark parameters."""

    seq_lengths: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    num_warmup: int = 10
    num_benchmark: int = 50
    device: str = "cuda"
    dtype: str = "float16"
    verify_correctness: bool = True
    correctness_threshold: float = 0.99


# ---------------------------------------------------------------------------
# Phase 2 — Hierarchical Token-Level Attention
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalConfig:
    """Phase 2 quality-oriented sparse attention hyperparameters.

    Implements a three-stage hierarchical filtering pipeline:
      1. **Macro-Selection**: Quest page scoring → Top-M pages.
      2. **Micro-Selection**: Per-token scoring within those M pages.
      3. **Final Consolidation**: Sink/recent protection + Top-B tokens.

    The key invariant: **token_budget == top_k * page_size** so Phase 2
    operates under the exact same attention budget as the Quest baseline.

    Attributes:
        page_size: KV tokens per page (same as Quest).
        top_k: Reference page count from Quest — the final token budget B
               equals top_k * page_size.
        macro_multiplier: M = macro_multiplier * top_k — how many pages to
                          pull into the micro-selection pool. A wider net
                          (higher multiplier) catches more high-value tokens
                          at the cost of more scoring overhead.
        num_sink_tokens: Number of initial tokens always force-protected
                         (StreamingLLM "attention sink" finding).
        num_recent_tokens: Number of most-recent tokens always force-protected.
        adaptive_budget: When True, dynamically size M based on the page-score
                         distribution (concentration → smaller M; uniformity →
                         larger M).
        token_budget: Explicit override for the final Top-B budget.  When None,
                      budget = top_k * page_size (matching Quest exactly).
    """

    page_size: int = 64
    top_k: int = 4
    macro_multiplier: int = 3
    num_sink_tokens: int = 4
    num_recent_tokens: int = 64
    adaptive_budget: bool = True
    token_budget: int | None = None


# ---------------------------------------------------------------------------
# Phase 2 — Evaluation / Metrics config
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    """Quality metrics computed alongside the Phase 2 benchmark.

    Attributes:
        compute_token_recall: Whether to compute token-recall@B — the fraction
                              of full-attention top-B tokens captured by the
                              sparse method.
        compute_attention_overlap: Whether to compute Jaccard overlap between
                                   the full and sparse attention token sets.
        num_top_b_for_recall: Number of top attention tokens to track (defaults
                              to the Quest budget top_k * page_size).
    """

    compute_token_recall: bool = True
    compute_attention_overlap: bool = True
    num_top_b_for_recall: int | None = None  # None → use token budget


@dataclass
class Config:
    """Top-level configuration bundling all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quest: QuestConfig = field(default_factory=QuestConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


# Convenience default instance
default_config = Config()
