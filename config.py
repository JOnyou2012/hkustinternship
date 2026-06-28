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


@dataclass
class Config:
    """Top-level configuration bundling all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quest: QuestConfig = field(default_factory=QuestConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Convenience default instance
default_config = Config()
