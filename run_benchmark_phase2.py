#!/usr/bin/env python3
"""
Quest Baseline — Phase 2: Hierarchical Token-Level Benchmark
=============================================================

Evaluates the **quality-oriented** hierarchical sparse attention against
the Phase 1 (page-wise Quest) baseline and full (dense) attention.

The key advance: instead of selecting full pages, Phase 2 cherry-picks
individual tokens from a wider set of candidate pages, operating under
the **exact same attention budget** as the Quest baseline.  Quality gains
come from better token selection, not from attending to more tokens.

Pipeline per decode step:
  1. **Macro**: Quest page scoring → Top-M pages (M = multiplier × top_k)
  2. **Micro**: Exact Q·K scoring of every token in those M pages
  3. **Consolidation**: Sink/recent protection + Top-B tokens (B = top_k × page_size)

Usage::

    python run_benchmark_phase2.py
    python run_benchmark_phase2.py --device cpu
    python run_benchmark_phase2.py --kv-lens 512 1024 2048 4096 8192 --top-k 4
    python run_benchmark_phase2.py --page-size 32 --top-k 8 --macro-multiplier 2
    python run_benchmark_phase2.py --num-sink 8 --num-recent 128
    python run_benchmark_phase2.py --no-adaptive
    python run_benchmark_phase2.py --no-quality
    python run_benchmark_phase2.py --compare-baseline  # side-by-side vs Phase 1
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from experiment import run_sweep, run_sweep_phase2
from utils import format_results_table, format_phase2_results_table


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quest Baseline — Phase 2 Hierarchical Token-Level Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark_phase2.py
  python run_benchmark_phase2.py --device cpu --kv-lens 512 1024 2048
  python run_benchmark_phase2.py --page-size 32 --top-k 8 --macro-multiplier 2
  python run_benchmark_phase2.py --compare-baseline
        """,
    )

    # Model dimensions
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of query heads (default: 32)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Dimension per head (default: 128)")
    parser.add_argument("--num-kv-heads", type=int, default=32,
                        help="Number of KV heads — set < num-heads for GQA")

    # Quest / page parameters
    parser.add_argument("--page-size", type=int, default=64,
                        help="Tokens per page (default: 64)")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Reference page count — token budget = top_k × page_size")

    # Phase 2 — hierarchical parameters
    parser.add_argument("--macro-multiplier", type=int, default=3,
                        help="M = multiplier × top_k macro pages (default: 3)")
    parser.add_argument("--num-sink", type=int, default=4,
                        help="Force-protected initial sink tokens (default: 4)")
    parser.add_argument("--num-recent", type=int, default=64,
                        help="Force-protected trailing recent tokens (default: 64)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive macro-budget sizing")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Explicit Top-B token budget (default: top_k × page_size)")

    # Experiment
    parser.add_argument("--kv-lens", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192],
                        help="KV-cache sizes to sweep (default: 512 1024 2048 4096 8192)")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-benchmark", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip cosine-similarity verification")
    parser.add_argument("--correctness-threshold", type=float, default=0.99)
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip token-recall and attention-overlap metrics")

    # Comparison mode
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Run Phase 1 (Quest page-wise) baseline side-by-side")

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    # Sanity checks
    assert args.num_heads % args.num_kv_heads == 0, (
        f"num_heads ({args.num_heads}) must be divisible by "
        f"num_kv_heads ({args.num_kv_heads})"
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠  CUDA not available — falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    token_budget = args.token_budget or (args.top_k * args.page_size)

    print("=" * 70)
    print("  Phase 2 — Hierarchical Token-Level Sparse Attention")
    print("  iSING Lab, HKUST")
    print("=" * 70)
    print(f"  Token budget: {token_budget}  (= top_k × page_size = "
          f"{args.top_k} × {args.page_size})")
    print(f"  Macro multiplier: {args.macro_multiplier}×  "
          f"Sink: {args.num_sink}  Recent: {args.num_recent}  "
          f"Adaptive: {not args.no_adaptive}")

    # ── Run Phase 2 sweep ──
    results = run_sweep_phase2(
        seq_lengths=args.kv_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_kv_heads=args.num_kv_heads,
        page_size=args.page_size,
        top_k=args.top_k,
        macro_multiplier=args.macro_multiplier,
        num_sink_tokens=args.num_sink,
        num_recent_tokens=args.num_recent,
        adaptive_budget=not args.no_adaptive,
        num_warmup=args.num_warmup,
        num_benchmark=args.num_benchmark,
        device=args.device,
        dtype_str=args.dtype,
        verify=not args.no_verify,
        correctness_threshold=args.correctness_threshold,
        compute_quality=not args.no_quality,
    )

    (
        seq_lengths,
        full_lat, hier_lat,
        full_mem, hier_mem,
        cosine_sims,
        quality_metrics,
    ) = results

    # ── Phase 2 summary table ──
    print(f"\n{'='*70}")
    print("  Phase 2 Summary")
    print(f"{'='*70}\n")
    print(format_phase2_results_table(
        seq_lengths, full_lat, hier_lat,
        full_mem, hier_mem, cosine_sims,
        quality_metrics, token_budget,
    ))

    # ── Quality metrics summary ──
    if quality_metrics:
        print(f"\n{'='*70}")
        print("  Quality Metrics — Token Recall & Attention Overlap")
        print(f"{'='*70}\n")
        print(f"  {'KV Len':>7s} | {'Rec@B':>8s} | {'Jac Overlap':>12s} | "
              f"{'CosSim':>8s}")
        print(f"  {'-'*7}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}")
        for i, sl in enumerate(seq_lengths):
            qm = quality_metrics[i]
            cs = cosine_sims[i] if cosine_sims else float("nan")
            print(f"  {sl:>7d} | {qm['token_recall_mean']:>8.4f} | "
                  f"{qm['attention_jaccard_mean']:>12.4f} | {cs:>8.5f}")

        print(f"\n  Per-head token recall (final KV length "
              f"{seq_lengths[-1]}):")
        qm_last = quality_metrics[-1]
        recall_heads = qm_last.get("token_recall_per_head", [])
        if recall_heads:
            # Print heads in groups of 8
            for g in range(0, len(recall_heads), 8):
                group = recall_heads[g:g+8]
                items = "  ".join(
                    f"H{h:02d}: {r:.3f}" for h, r in enumerate(group, start=g)
                )
                print(f"    {items}")

    # ── Optional Phase 1 comparison ──
    if args.compare_baseline:
        print(f"\n{'='*70}")
        print("  Running Phase 1 (Quest page-wise) baseline for comparison...")
        print(f"{'='*70}\n")

        baseline_results = run_sweep(
            seq_lengths=args.kv_lens,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_kv_heads=args.num_kv_heads,
            page_size=args.page_size,
            top_k=args.top_k,
            num_warmup=args.num_warmup,
            num_benchmark=args.num_benchmark,
            device=args.device,
            dtype_str=args.dtype,
            verify=not args.no_verify,
            correctness_threshold=args.correctness_threshold,
        )

        (
            _, quest_full_lat, quest_lat,
            quest_full_mem, quest_mem,
            quest_cosine_sims,
        ) = baseline_results

        print(f"\n{'='*70}")
        print("  Side-by-Side Comparison: Phase 1 (Quest) vs Phase 2 (Hierarchical)")
        print(f"{'='*70}\n")

        header = (
            f"{'KV Len':>7s} | "
            f"{'P1 CosSim':>10s} | "
            f"{'P2 CosSim':>10s} | "
            f"{'Δ CosSim':>9s} | "
            f"{'P1 Speedup':>11s} | "
            f"{'P2 Speedup':>11s} | "
            f"{'Rec@B':>8s}"
        )
        print(header)
        print("-" * len(header))
        for i, sl in enumerate(seq_lengths):
            p1_cs = quest_cosine_sims[i] if quest_cosine_sims else float("nan")
            p2_cs = cosine_sims[i] if cosine_sims else float("nan")
            delta_cs = (p2_cs - p1_cs) if (p1_cs == p1_cs and p2_cs == p2_cs) else float("nan")
            p1_speedup = quest_full_lat[i] / max(quest_lat[i], 1e-6)
            p2_speedup = full_lat[i] / max(hier_lat[i], 1e-6)
            rec = quality_metrics[i]["token_recall_mean"] if quality_metrics else float("nan")

            print(
                f"{sl:>7d} | "
                f"{p1_cs:>10.5f} | "
                f"{p2_cs:>10.5f} | "
                f"{delta_cs:>+9.5f} | "
                f"{p1_speedup:>9.2f}x | "
                f"{p2_speedup:>9.2f}x | "
                f"{rec:>8.4f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
