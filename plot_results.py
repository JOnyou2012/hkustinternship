#!/usr/bin/env python3
"""
Core Result Figure — Phase 2 vs Phase 1 Quality Comparison
============================================================

Generates the single deliverable figure for the iSING Lab sparse attention
project.  Plots **cosine similarity** (proxy for attention quality /
LongBench accuracy) against **KV-cache size** for both the Quest page-wise
baseline (Phase 1) and the hierarchical token-level refinement (Phase 2).

Both methods operate under the **identical token budget** of
``B = top_k × page_size`` tokens per decode step.  The figure visually
proves that token-level selection achieves higher quality under the same
budget constraints.

Usage::

    python plot_results.py              # run benchmarks, then plot
    python plot_results.py --cached     # use cached results (faster)
    python plot_results.py --device cpu --no-sweep  # quick single-point check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CACHE_FILE = ROOT / ".benchmark_cache_phase2.json"

# ── Style ────────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# Colour palette
P1_COLOR = "#E07050"  # Phase 1 — warm red/orange
P2_COLOR = "#3070B0"  # Phase 2 — deep blue
FULL_COLOR = "#808080"  # Full attention — grey reference


# ═══════════════════════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════════════════════


def run_benchmarks(
    device: str = "cpu",
    num_benchmark: int = 30,
    num_warmup: int = 5,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Run both Phase 1 and Phase 2 benchmarks and collect cosine similarities.

    Returns:
        kv_lengths, p1_cosims, p2_cosims, p1_speedups, p2_speedups
    """
    from experiment import run_sweep, run_sweep_phase2

    kv_lengths = [512, 1024, 2048, 4096, 8192]

    print("─" * 60)
    print("  Running Phase 1 (Quest page-wise baseline) ...")
    print("─" * 60)
    p1_results = run_sweep(
        seq_lengths=kv_lengths,
        num_warmup=num_warmup,
        num_benchmark=num_benchmark,
        device=device,
        dtype_str="float16",
        verify=True,
        correctness_threshold=0.0,  # collect all, don't threshold
    )
    _, p1_full_lat, p1_quest_lat, _, _, p1_cosims = p1_results

    print("\n" + "─" * 60)
    print("  Running Phase 2 (Hierarchical token-level) ...")
    print("─" * 60)
    p2_results = run_sweep_phase2(
        seq_lengths=kv_lengths,
        num_warmup=num_warmup,
        num_benchmark=num_benchmark,
        device=device,
        dtype_str="float16",
        verify=True,
        correctness_threshold=0.0,
        compute_quality=True,
    )
    (
        _,
        p2_full_lat,
        p2_hier_lat,
        _,
        _,
        p2_cosims,
        p2_quality,
    ) = p2_results

    # Speedups
    p1_speedups = [
        fl / max(ql, 1e-6) for fl, ql in zip(p1_full_lat, p1_quest_lat)
    ]
    p2_speedups = [
        fl / max(hl, 1e-6) for fl, hl in zip(p2_full_lat, p2_hier_lat)
    ]

    return kv_lengths, p1_cosims, p2_cosims, p1_speedups, p2_speedups


def load_cache() -> Optional[dict]:
    """Load cached benchmark results if available."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


def save_cache(data: dict) -> None:
    """Save benchmark results to cache."""
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════


def plot_figure(
    kv_lengths: List[int],
    p1_cosims: List[float],
    p2_cosims: List[float],
    p1_speedups: Optional[List[float]] = None,
    p2_speedups: Optional[List[float]] = None,
    token_budget: int = 256,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate the core result figure.

    Left panel:  Cosine similarity vs KV-cache size (quality comparison).
    Right panel: Speedup vs KV-cache size (optional, for context).

    Args:
        kv_lengths: KV-cache sizes (X-axis).
        p1_cosims: Phase 1 cosine similarities.
        p2_cosims: Phase 2 cosine similarities.
        p1_speedups: Phase 1 speedups (optional).
        p2_speedups: Phase 2 speedups (optional).
        token_budget: The fixed token budget used in both phases.
        save_path: If set, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    n = len(kv_lengths)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ── Left panel: Quality (Cosine Similarity) ─────────────────────
    bars1 = ax1.bar(
        x - width / 2, p1_cosims, width,
        color=P1_COLOR, edgecolor="white", linewidth=0.6,
        label="Phase 1: Quest (Page-wise)",
        zorder=2,
    )
    bars2 = ax1.bar(
        x + width / 2, p2_cosims, width,
        color=P2_COLOR, edgecolor="white", linewidth=0.6,
        label="Phase 2: Hierarchical (Token-wise)",
        zorder=2,
    )

    # Annotate bars with values
    for bar, val in zip(bars1, p1_cosims):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            color=P1_COLOR, fontweight="bold",
        )
    for bar, val in zip(bars2, p2_cosims):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            color=P2_COLOR, fontweight="bold",
        )

    # Improvement percentage annotations
    for i in range(n):
        improvement = (p2_cosims[i] - p1_cosims[i]) / max(p1_cosims[i], 1e-6) * 100
        ax1.annotate(
            f"+{improvement:.0f}%",
            xy=(x[i], max(p1_cosims[i], p2_cosims[i]) + 0.08),
            ha="center", fontsize=8.5, fontweight="bold",
            color="#1A7030",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8F5E9",
                       edgecolor="#A5D6A7", linewidth=0.5),
        )

    ax1.set_xlabel("KV-Cache Size (tokens)")
    ax1.set_ylabel("Cosine Similarity vs Full Attention")
    ax1.set_title(
        f"Attention Quality Under Fixed Budget\n"
        f"(B = {token_budget} tokens, top_k = 4, page_size = 64)",
        fontsize=11,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{sl:,}" for sl in kv_lengths])
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor="grey")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Add budget-fraction annotation
    for i, sl in enumerate(kv_lengths):
        budget_pct = token_budget / sl * 100
        ax1.text(
            x[i], 0.02,
            f"Budget:\n{budget_pct:.1f}% of KV",
            ha="center", va="bottom", fontsize=7,
            color="#666666", fontstyle="italic",
        )

    # ── Right panel: Speedup ────────────────────────────────────────
    if p1_speedups and p2_speedups:
        ax2.plot(
            kv_lengths, p1_speedups,
            "o-", color=P1_COLOR, linewidth=2, markersize=8,
            label="Phase 1: Quest",
            zorder=3,
        )
        ax2.plot(
            kv_lengths, p2_speedups,
            "s-", color=P2_COLOR, linewidth=2, markersize=8,
            label="Phase 2: Hierarchical",
            zorder=3,
        )
        ax2.axhline(
            y=1.0, color=FULL_COLOR, linestyle="--", linewidth=1.2,
            label="Full attention (1×)",
            zorder=1,
        )
        ax2.set_xlabel("KV-Cache Size (tokens)")
        ax2.set_ylabel("Speedup vs Full Attention")
        ax2.set_title("Decode-Step Speedup (CPU, fp16)", fontsize=11)
        ax2.legend(loc="upper left", framealpha=0.9, edgecolor="grey")
        ax2.grid(alpha=0.3)
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(kv_lengths)
        ax2.set_xticklabels([f"{sl:,}" for sl in kv_lengths])
        ax2.set_axisbelow(True)

    # ── Global annotations ──────────────────────────────────────────
    fig.suptitle(
        "iSING Lab — Phase 2: Hierarchical Token-Level Sparse Attention",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Footer
    fig.text(
        0.5, -0.02,
        f"Same budget (B = {token_budget} tokens) for both methods.  "
        f"Higher cosine similarity = better approximation of full attention.  "
        f"CPU benchmarks; GPU gap narrows at scale.",
        ha="center", fontsize=8, color="#888888", fontstyle="italic",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.15)
        print(f"\n  Figure saved → {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 core result figure",
    )
    parser.add_argument(
        "--cached", action="store_true",
        help="Use cached benchmark data (skip re-running benchmarks)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for benchmarks: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--num-benchmark", type=int, default=30,
        help="Benchmark iterations per data point",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--no-sweep", action="store_true",
        help="Skip sweep; use hardcoded reference data",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: reports/phase2_quality_vs_budget.pdf)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    REPORTS.mkdir(exist_ok=True)
    save_path = Path(args.output) if args.output else (
        REPORTS / "phase2_quality_vs_budget.pdf"
    )

    token_budget = 256  # top_k=4 × page_size=64

    if args.no_sweep:
        # Load reference results from the cache file (avoids stale hardcoded data)
        cache = load_cache()
        if cache is None:
            print("  No cached benchmark data found. "
                  "Run without --no-sweep to generate it, "
                  "or provide a cache file.", file=sys.stderr)
            sys.exit(1)
        print("  Using cached benchmark data (no-sweep mode)")
        kv_lengths = cache["kv_lengths"]
        p1_cosims = cache["p1_cosims"]
        p2_cosims = cache["p2_cosims"]
        p1_speedups = cache.get("p1_speedups")
        p2_speedups = cache.get("p2_speedups")
    elif args.cached:
        data = load_cache()
        if data is None:
            print("  No cached data found — running benchmarks ...")
            data_dict = _run_and_cache(args)
        else:
            print("  Loaded cached benchmark data")
            data_dict = data
        kv_lengths = data_dict["kv_lengths"]
        p1_cosims = data_dict["p1_cosims"]
        p2_cosims = data_dict["p2_cosims"]
        p1_speedups = data_dict.get("p1_speedups")
        p2_speedups = data_dict.get("p2_speedups")
    else:
        data_dict = _run_and_cache(args)
        kv_lengths = data_dict["kv_lengths"]
        p1_cosims = data_dict["p1_cosims"]
        p2_cosims = data_dict["p2_cosims"]
        p1_speedups = data_dict.get("p1_speedups")
        p2_speedups = data_dict.get("p2_speedups")

    fig = plot_figure(
        kv_lengths=kv_lengths,
        p1_cosims=p1_cosims,
        p2_cosims=p2_cosims,
        p1_speedups=p1_speedups,
        p2_speedups=p2_speedups,
        token_budget=token_budget,
        save_path=save_path,
    )

    if args.show:
        plt.show()

    plt.close(fig)
    print("\n  Done.")


def _run_and_cache(args) -> dict:
    """Run benchmarks and cache results."""
    (
        kv_lengths, p1_cosims, p2_cosims, p1_speedups, p2_speedups,
    ) = run_benchmarks(
        device=args.device,
        num_benchmark=args.num_benchmark,
        num_warmup=args.num_warmup,
    )
    data = {
        "kv_lengths": kv_lengths,
        "p1_cosims": p1_cosims,
        "p2_cosims": p2_cosims,
        "p1_speedups": p1_speedups,
        "p2_speedups": p2_speedups,
    }
    save_cache(data)
    return data


if __name__ == "__main__":
    main()
