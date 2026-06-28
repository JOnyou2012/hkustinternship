#!/usr/bin/env python3
"""
Quest Baseline — Phase 1: Decode-Step Benchmark
================================================

Simulates the **decode phase** of long-context LLM serving: a single new query
token attends to a large KV cache.  Full (dense) attention is compared against
Quest page-wise sparse attention across a sweep of KV-cache sizes.

Usage::

    python run_benchmark.py
    python run_benchmark.py --device cpu
    python run_benchmark.py --kv-lens 512 1024 2048 4096 8192 --top-k 2
    python run_benchmark.py --page-size 32 --top-k 8
    python run_benchmark.py --num-kv-heads 8 --num-heads 32
    python run_benchmark.py --no-verify
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from experiment import run_sweep
from utils import format_results_table


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Quest Baseline — Phase 1 Decode-Step Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py
  python run_benchmark.py --device cpu --kv-lens 256 512 1024
  python run_benchmark.py --page-size 32 --top-k 8
  python run_benchmark.py --num-kv-heads 8 --num-heads 32
        """,
    )

    # Model dimensions
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of query heads (default: 32)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Dimension per head (default: 128)")
    parser.add_argument("--num-kv-heads", type=int, default=32,
                        help="Number of KV heads — set < num-heads for GQA")

    # Quest hyperparameters
    parser.add_argument("--page-size", type=int, default=64,
                        help="Tokens per page (default: 64)")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Number of pages selected for sparse attention (default: 4)")

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

    print("=" * 70)
    print("  Quest Baseline — Phase 1: Page-wise Sparse Attention")
    print("  iSING Lab, HKUST")
    print("=" * 70)

    # Run sweep
    results = run_sweep(
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
        seq_lengths,
        full_lat, quest_lat,
        full_mem, quest_mem,
        cosine_sims,
    ) = results

    # Print summary table
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}\n")
    print(format_results_table(
        seq_lengths, full_lat, quest_lat,
        full_mem, quest_mem, cosine_sims,
    ))

    print("\nDone.")


if __name__ == "__main__":
    main()
