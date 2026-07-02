#!/usr/bin/env python3
"""
Core Result Figure — Real-Model LongBench Evaluation
=====================================================

Generates the lab's required deliverable figure: **Perplexity vs. Token Budget**
for Full attention, Quest (page-wise), and Hierarchical (token-level) sparse
attention on a real pretrained model (GPT-2) on real text (WikiText-2 or PG-19).

Two-panel figure:
  Left:  Perplexity vs. token budget (lower = better)
  Right: PPL ratio (sparse / full) vs. budget fraction

Usage::

    # Run full eval first, then plot from cached results
    python eval_perplexity.py --output results.json
    python plot_real_model_results.py --from-file results.json

    # Or run inline (slower — evaluates all three methods)
    python plot_real_model_results.py --run-eval --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CACHE_FILE = ROOT / ".perplexity_results.json"

# ── Style ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Colour palette — consistent with Phase 1/2 synthetic figure
P1_COLOR = "#E07050"   # Quest — warm red/orange
P2_COLOR = "#3070B0"   # Hierarchical — deep blue
FULL_COLOR = "#808080"  # Full attention — grey


def plot_longbench_figure(
    budgets: List[int],
    ppl_full: float,
    ppl_quest: List[float],
    ppl_hierarchical: List[float],
    kv_length: int = 1024,
    model_name: str = "GPT-2",
    dataset_name: str = "WikiText-2",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate the LongBench evaluation figure.

    Args:
        budgets:         Token budget values (X-axis).
        ppl_full:        Perplexity of unpatched full attention (horizontal line).
        ppl_quest:        Perplexity of Quest at each budget.
        ppl_hierarchical: Perplexity of Hierarchical at each budget.
        kv_length:        Context length used in evaluation.
        model_name:       Model name for figure title.
        dataset_name:     Dataset name for figure title.
        save_path:        If set, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    n = len(budgets)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ── Left panel: Perplexity vs. Token Budget ─────────────────────────
    bars_q = ax1.bar(
        x - width / 2, ppl_quest, width,
        color=P1_COLOR, edgecolor="white", linewidth=0.6,
        label="Quest (Page-wise)",
        zorder=2,
    )
    bars_h = ax1.bar(
        x + width / 2, ppl_hierarchical, width,
        color=P2_COLOR, edgecolor="white", linewidth=0.6,
        label="Hierarchical (Token-wise)",
        zorder=2,
    )

    # Full attention horizontal line
    ax1.axhline(
        y=ppl_full, color=FULL_COLOR, linestyle="--", linewidth=1.5,
        label=f"Full attention ({ppl_full:.2f})",
        zorder=1,
    )

    # Annotate bars
    for bar, val in zip(bars_q, ppl_quest):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            color=P1_COLOR, fontweight="bold",
        )
    for bar, val in zip(bars_h, ppl_hierarchical):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            color=P2_COLOR, fontweight="bold",
        )

    # Delta annotations — improvement of Hierarchical over Quest
    for i in range(n):
        delta = ppl_quest[i] - ppl_hierarchical[i]
        if delta > 0:
            ax1.annotate(
                f"−{delta:.2f}",
                xy=(x[i], max(ppl_quest[i], ppl_hierarchical[i]) + 1.0),
                ha="center", fontsize=8.5, fontweight="bold",
                color="#1A7030",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8F5E9",
                          edgecolor="#A5D6A7", linewidth=0.5),
            )

    ax1.set_xlabel("Token Budget (B)")
    ax1.set_ylabel("Perplexity (lower = better)")
    ax1.set_title(
        f"Language Modeling Quality vs. Token Budget\n"
        f"({model_name} on {dataset_name}, context = {kv_length} tokens)",
        fontsize=11,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in budgets])
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor="grey")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Budget % annotation
    for i, b in enumerate(budgets):
        pct = b / kv_length * 100
        ax1.text(
            x[i], max(ppl_quest[i], ppl_hierarchical[i]) - 0.5,
            f"{pct:.0f}% of KV",
            ha="center", va="top", fontsize=7,
            color="#666666", fontstyle="italic",
        )

    # ── Right panel: PPL Ratio (sparse / full) vs. Budget Fraction ──────
    budget_pcts = [b / kv_length * 100 for b in budgets]
    quest_ratio = [q / ppl_full for q in ppl_quest]
    hier_ratio = [h / ppl_full for h in ppl_hierarchical]

    ax2.plot(
        budget_pcts, quest_ratio,
        "o-", color=P1_COLOR, linewidth=2, markersize=8,
        label="Quest (Page-wise)",
        zorder=3,
    )
    ax2.plot(
        budget_pcts, hier_ratio,
        "s-", color=P2_COLOR, linewidth=2, markersize=8,
        label="Hierarchical (Token-wise)",
        zorder=3,
    )
    ax2.axhline(
        y=1.0, color=FULL_COLOR, linestyle="--", linewidth=1.2,
        label="Full attention (1×)",
        zorder=1,
    )

    ax2.set_xlabel("Budget as % of Context Length")
    ax2.set_ylabel("Perplexity Ratio (sparse / full)")
    ax2.set_title("Quality Degradation vs. Budget Fraction", fontsize=11)
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor="grey")
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)
    # Invert X axis (larger budget → right)
    ax2.invert_xaxis()

    # ── Global annotations ──────────────────────────────────────────────
    fig.suptitle(
        f"iSING Lab — Real-Model Sparse Attention Evaluation\n"
        f"{model_name} on {dataset_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    fig.text(
        0.5, -0.02,
        f"Same token budget for both sparse methods.  "
        f"Lower perplexity = better language modeling quality.  "
        f"Evaluated on CPU (Apple M4).",
        ha="center", fontsize=8, color="#888888", fontstyle="italic",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.15)
        print(f"\n  Figure saved → {save_path}")

    return fig


def sweep_budgets(
    model_name: str = "gpt2",
    dataset_name: str = "wikitext2",
    budgets: Optional[List[int]] = None,
    chunk_size: int = 512,
    max_samples: int = 50,
    device: str = "cpu",
    quiet: bool = False,
) -> Tuple[List[int], float, List[float], List[float]]:
    """Sweep token budgets, returning perplexity for each method at each budget.

    Runs Quest and Hierarchical at each budget point.  Full attention is
    evaluated once (it doesn't depend on budget).

    Returns:
        budgets, ppl_full, ppl_quest_list, ppl_hierarchical_list
    """
    from eval_perplexity import run_evaluation

    if budgets is None:
        budgets = [64, 128, 256, 512]

    # Full attention — once only (independent of budget)
    if not quiet:
        print("Evaluating Full attention (unpatched baseline)...")
    results_full = run_evaluation(
        model_name=model_name,
        dataset_name=dataset_name,
        chunk_size=chunk_size,
        max_samples=max_samples,
        device_str=device,
        page_size=64,
        top_k=1,  # dummy — not used for full
        macro_multiplier=1,
        quiet=True,
    )
    ppl_full = results_full["full"]["perplexity"]

    # Quest and Hierarchical at each budget
    ppl_quest_list = []
    ppl_hier_list = []

    for budget in budgets:
        # Derive top_k from budget (page_size fixed at 64)
        page_size = 64
        top_k = max(1, budget // page_size)

        if not quiet:
            print(f"\n--- Budget = {budget} (top_k={top_k}) ---")

        results = run_evaluation(
            model_name=model_name,
            dataset_name=dataset_name,
            chunk_size=chunk_size,
            max_samples=max_samples,
            device_str=device,
            page_size=page_size,
            top_k=top_k,
            macro_multiplier=3,
            num_sink_tokens=4,
            num_recent_tokens=min(64, budget // 4),  # scale recent with budget
            adaptive_budget=True,
            quiet=quiet,
        )

        ppl_quest_list.append(results["quest"]["perplexity"])
        ppl_hier_list.append(results["hierarchical"]["perplexity"])

    return budgets, ppl_full, ppl_quest_list, ppl_hier_list


def load_cached_results(path: Path) -> dict:
    """Load previously saved perplexity results from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate the LongBench-style evaluation figure",
    )
    parser.add_argument("--run-eval", action="store_true",
                        help="Run the full perplexity evaluation (slow)")
    parser.add_argument("--from-file", type=str, default=None,
                        help="Load results from a JSON file")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name (default: gpt2)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        help="Dataset (default: wikitext2)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Chunk size for evaluation")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max chunks (default: 50)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[64, 128, 256, 512],
                        help="Budgets to sweep (default: 64 128 256 512)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: reports/longbench_evaluation.pdf)")
    parser.add_argument("--show", action="store_true",
                        help="Display figure interactively")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    save_path = Path(args.output) if args.output else (
        REPORTS / "longbench_evaluation.pdf"
    )

    if args.from_file:
        # Load pre-computed results
        data = load_cached_results(Path(args.from_file))
        budgets = data["budgets"]
        ppl_full = data["ppl_full"]
        ppl_quest = data["ppl_quest"]
        ppl_hier = data["ppl_hierarchical"]

    elif args.run_eval:
        # Run the full evaluation
        budgets, ppl_full, ppl_quest, ppl_hier = sweep_budgets(
            model_name=args.model,
            dataset_name=args.dataset,
            budgets=args.budgets,
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
            device=args.device,
            quiet=args.quiet,
        )
        # Cache results
        cache_data = {
            "model": args.model,
            "dataset": args.dataset,
            "chunk_size": args.chunk_size,
            "budgets": budgets,
            "ppl_full": ppl_full,
            "ppl_quest": ppl_quest,
            "ppl_hierarchical": ppl_hier,
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
        print(f"  Results cached → {CACHE_FILE}")

    else:
        # Use cached data if available, otherwise show a message
        if CACHE_FILE.exists():
            data = load_cached_results(CACHE_FILE)
            budgets = data["budgets"]
            ppl_full = data["ppl_full"]
            ppl_quest = data["ppl_quest"]
            ppl_hier = data["ppl_hierarchical"]
            print(f"  Loaded cached results from {CACHE_FILE}")
        else:
            print(
                "  No cached results found. Run with --run-eval to evaluate, "
                "or --from-file to load a results JSON.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Determine context length (GPT-2 max)
    if "gpt2" in args.model:
        kv_len = 1024
    else:
        kv_len = args.chunk_size

    fig = plot_longbench_figure(
        budgets=budgets,
        ppl_full=ppl_full,
        ppl_quest=ppl_quest,
        ppl_hierarchical=ppl_hier,
        kv_length=kv_len,
        model_name=args.model.upper(),
        dataset_name=args.dataset,
        save_path=save_path,
    )

    if args.show:
        plt.show()

    plt.close(fig)
    print("\n  Done.")


if __name__ == "__main__":
    main()
