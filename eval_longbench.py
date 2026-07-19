#!/usr/bin/env python3
"""
LongBench Evaluation — Full vs Quest vs Hierarchical Sparse Attention
=====================================================================

Evaluates three attention configurations on the LongBench benchmark suite
at identical token budgets, measuring downstream task performance directly
(rather than synthetic cosine-similarity or perplexity proxies).

Methods:
  1. **Full** (unpatched dense attention) — the quality ceiling.
  2. **Quest** (page-wise sparse) — Phase 1 baseline.
  3. **Hierarchical** (token-level sparse) — Phase 2 refinement.

All sparse methods operate at the **same token budget** B = top_k × page_size,
so any quality difference comes purely from better token selection.

Generation uses full-sequence forwarding (no KV cache) so sparse attention
operates on the complete context at every decode step — correct but slower
than cached generation.

Usage::

    # Quick smoke test on a single task
    python eval_longbench.py --tasks samsum --max-samples 5 --max-new-tokens 32

    # Full evaluation sweep
    python eval_longbench.py --tasks narrativeqa hotpotqa trec samsum \\
        --max-samples 50 --budgets 128 256 --device cuda

    # With a specific model
    python eval_longbench.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --tasks narrativeqa --max-samples 20

Requirements:
    pip install transformers datasets
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ── Local imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hf_model_patcher import patch_model, _detect_model_type, _build_config_from_hf  # noqa: E402
from longbench_utils import (  # noqa: E402
    TASK_CONFIGS,
    compute_metric,
    format_prompt,
    get_task_list,
    load_task_dataset,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """Load a pretrained causal LM and its tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device:     torch device.
        dtype:      Model dtype (float16 recommended for GPU).

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_name}  (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_sparse(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Autoregressive generation with full-sequence forwarding on every step.

    Because our sparse-attention modules do not use HuggingFace's
    ``past_key_values`` mechanism, each decode step must reprocess the
    **entire** growing sequence (prompt + all generated tokens so far).
    This is O(n²) in total sequence length but produces correct attention
    over the full KV context at every step.

    Args:
        model:           HF causal LM (optionally patched with sparse attn).
        input_ids:       Prompt tokens, shape (1, prompt_len).
        max_new_tokens:  Stop after generating this many tokens.
        eos_token_id:    Stop early when this token is produced.
        pad_token_id:    Used for attention masking (ignored by sparse modules).

    Returns:
        (full_ids, generated_token_ids)
        full_ids:        (1, prompt_len + num_generated) — the full sequence.
        generated_token_ids: list of generated token ints.
    """
    generated: List[int] = []
    current_ids = input_ids  # (1, T) — grows each step
    device = input_ids.device

    for _step in range(max_new_tokens):
        # Forward pass over the full sequence
        # Sparse attention: builds pages from all K/V, selects Top-B for
        # the last query position.
        outputs = model(current_ids)

        # Greedy decode: pick the most likely next token
        logits = outputs.logits[0, -1, :]  # last position over vocab
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1,)
        token_id = next_token.item()

        generated.append(token_id)

        # Stop conditions
        if eos_token_id is not None and token_id == eos_token_id:
            break

        # Append the new token for the next step
        current_ids = torch.cat(
            [current_ids, next_token.unsqueeze(0).to(device)], dim=1
        )

    return current_ids, generated


# ═══════════════════════════════════════════════════════════════════════════════
# Patching helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _update_sparse_budget(model, top_k: int, page_size: int = 64) -> None:
    """Update ``top_k`` and ``page_size`` on all patched layers in-place.

    This avoids reloading and re-patching the model when sweeping budgets.
    Only valid on an already-patched model.
    """
    from hf_model_patcher import _get_transformer_layers, _get_attention_module
    from hf_model_patcher import _SparseAttentionWrapper

    layers = _get_transformer_layers(model)
    for layer in layers:
        wrapper = _get_attention_module(layer)
        if isinstance(wrapper, _SparseAttentionWrapper):
            mod = wrapper.sparse_module
            if hasattr(mod, "top_k"):
                mod.top_k = top_k
            if hasattr(mod, "page_size"):
                mod.page_size = page_size
            # Update derived budget on hierarchical module
            if hasattr(mod, "token_budget") and mod.token_budget is not None:
                mod.token_budget = top_k * page_size


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    samples: List[Dict[str, Any]],
    device: torch.device,
    method_label: str,
    *,
    max_new_tokens: int = 64,
    max_prompt_tokens: Optional[int] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Evaluate one LongBench task with the given (possibly patched) model.

    Args:
        model:           HF causal LM (already patched if needed).
        tokenizer:       Tokenizer matching the model.
        task_name:       Key in TASK_CONFIGS.
        samples:         List of sample dicts from ``load_task_dataset``.
        device:          torch device.
        method_label:    Human-readable label for progress printing.
        max_new_tokens:  Generation budget per sample.
        max_prompt_tokens: Truncate prompt to this many tokens (None = no cap).
        quiet:           Suppress per-sample progress output.

    Returns:
        Dict with keys:
            task, method, num_samples, scores, mean, std, tokens_per_second,
            errors
    """
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    scores: List[float] = []
    total_time = 0.0
    total_tokens = 0
    errors = 0

    for idx, sample in enumerate(samples):
        # Format and tokenise the prompt
        prompt = format_prompt(task_name, sample)
        encoded = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Truncate if needed
        if max_prompt_tokens and encoded.size(1) > max_prompt_tokens:
            encoded = encoded[:, :max_prompt_tokens]

        prompt_len = encoded.size(1)
        total_tokens += prompt_len

        # Generate
        t0 = time.perf_counter()
        try:
            full_ids, gen_tokens = generate_sparse(
                model, encoded,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos,
                pad_token_id=pad,
            )
        except Exception as exc:
            if not quiet:
                print(f"    [{idx}] ERROR: {exc}", file=sys.stderr)
            errors += 1
            continue
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Decode only the generated tokens (skip the prompt)
        gen_len = len(gen_tokens)
        total_tokens += gen_len

        if gen_len > 0:
            prediction = tokenizer.decode(
                full_ids[0, prompt_len:], skip_special_tokens=True
            )
        else:
            prediction = ""

        # Score
        score = compute_metric(task_name, prediction, sample)
        scores.append(score)

        if not quiet and (idx < 3 or (idx + 1) % 10 == 0 or idx == len(samples) - 1):
            avg = sum(scores) / len(scores) if scores else 0.0
            tok_per_sec = total_tokens / max(total_time, 0.01)
            print(
                f"    [{method_label}] {task_name} {idx+1}/{len(samples)}  "
                f"avg={avg:.3f}  {tok_per_sec:.0f} tok/s  "
                f"(gen_tokens={gen_len}, time={elapsed:.1f}s)"
                + (" " * 10),
                end="\r",
            )

    if not quiet:
        print(" " * 100, end="\r")  # clear line

    mean_score = sum(scores) / len(scores) if scores else 0.0
    std_score = (
        (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
        if len(scores) > 1
        else 0.0
    )
    tok_per_sec = total_tokens / max(total_time, 0.01)

    return {
        "task": task_name,
        "method": method_label,
        "num_samples": len(scores),
        "scores": scores,
        "mean": mean_score,
        "std": std_score,
        "tokens_per_second": tok_per_sec,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_longbench_sweep(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    tasks: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    page_size: int = 64,
    macro_multiplier: int = 3,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 64,
    adaptive_budget: bool = True,
    max_samples: int = 50,
    max_new_tokens: int = 64,
    max_prompt_tokens: Optional[int] = None,
    device_str: str = "cpu",
    dtype_str: str = "float16",
    output_file: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run the full LongBench comparison: Full vs Quest vs Hierarchical.

    Args:
        model_name:         HF model identifier.
        tasks:              LongBench task names to evaluate.
        budgets:            Token budgets to sweep (B = top_k × page_size).
        page_size:          Tokens per page.
        macro_multiplier:   Hierarchical macro-stage multiplier.
        num_sink_tokens:    Sink tokens for hierarchical attention.
        num_recent_tokens:  Recent tokens for hierarchical attention.
        adaptive_budget:    Enable adaptive macro budget sizing.
        max_samples:        Max samples per task.
        max_new_tokens:     Max tokens to generate per sample.
        max_prompt_tokens:  Truncate prompts longer than this.
        device_str:         "cpu" or "cuda".
        dtype_str:          "float16", "float32", or "bfloat16".
        output_file:        Save results to this JSON file.
        quiet:              Suppress progress output.

    Returns:
        Nested dict: results[method][budget][task] → evaluation dict.
    """
    if tasks is None:
        tasks = get_task_list()
    if budgets is None:
        budgets = [128, 256, 512]

    dtype = getattr(torch, dtype_str)
    device = torch.device(device_str)

    # ── Load tokenizer once (shared across all methods) ──────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load datasets ────────────────────────────────────────────────────
    all_samples: Dict[str, list] = {}
    for task_name in tasks:
        if not quiet:
            print(f"Loading dataset: {task_name} ({max_samples} samples)")
        all_samples[task_name] = load_task_dataset(
            task_name,
            max_samples=max_samples,
            max_context_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )
        if not quiet:
            print(f"  → {len(all_samples[task_name])} samples")

    results: Dict[str, dict] = {}

    methods = ["full", "quest", "hierarchical"]

    for method in methods:
        method_budgets = [None] if method == "full" else budgets
        results[method] = {}

        # Load a fresh model for each method
        if not quiet:
            print(f"\n{'='*70}")
            print(f"  METHOD: {method}")
            print(f"{'='*70}")

        model, _ = load_model_and_tokenizer(model_name, device, dtype)

        # Patch (skip for full)
        if method != "full":
            top_k_first = budgets[0] // page_size
            patch_model(
                model,
                method=method,
                page_size=page_size,
                top_k=max(1, top_k_first),
                macro_multiplier=macro_multiplier,
                num_sink_tokens=num_sink_tokens,
                num_recent_tokens=num_recent_tokens,
                adaptive_budget=adaptive_budget,
            )

        for budget in method_budgets:
            budget_label = "full" if budget is None else str(budget)
            results[method][budget_label] = {}

            if budget is not None and method != "full":
                top_k = max(1, budget // page_size)
                actual_budget = top_k * page_size
                _update_sparse_budget(model, top_k=top_k, page_size=page_size)
                if not quiet:
                    print(f"\n  --- Budget: {actual_budget} "
                          f"(top_k={top_k} × page_size={page_size}) ---")
            else:
                if not quiet:
                    print(f"\n  --- Full (unpatched) attention ---")

            for task_name in tasks:
                samples = all_samples[task_name]

                if not quiet:
                    print(f"  Evaluating: {task_name}  "
                          f"({len(samples)} samples)")

                method_label = (
                    f"{method}" if budget is None
                    else f"{method}-b{budget}"
                )

                eval_result = evaluate_task(
                    model, tokenizer, task_name, samples,
                    device,
                    method_label=method_label,
                    max_new_tokens=max_new_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    quiet=quiet,
                )
                results[method][budget_label][task_name] = eval_result

                if not quiet:
                    cfg = TASK_CONFIGS[task_name]
                    print(
                        f"    → {eval_result['mean']:.4f} "
                        f"({cfg['metric']}) "
                        f"±{eval_result['std']:.4f}  "
                        f"[{eval_result['tokens_per_second']:.0f} tok/s]"
                    )

        # Free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────
    if output_file:
        # Convert to serializable format
        serializable = {}
        for method_name, method_data in results.items():
            serializable[method_name] = {}
            for budget_label, budget_data in method_data.items():
                serializable[method_name][budget_label] = {}
                for task_name, eval_data in budget_data.items():
                    serializable[method_name][budget_label][task_name] = {
                        k: v for k, v in eval_data.items()
                        if k != "scores"  # omit per-sample scores for compactness
                    }

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        if not quiet:
            print(f"\nResults saved → {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Results formatting
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: Dict[str, Any]) -> None:
    """Print a formatted comparison table of LongBench results."""
    methods = [m for m in ("full", "quest", "hierarchical") if m in results]

    # Find the first budget level (skip "full" key which is budget=None)
    budget_labels = []
    for m in methods:
        for b in results[m]:
            if b != "full":
                budget_labels.append(b)
    budget_labels = sorted(set(budget_labels), key=int)

    tasks = set()
    for m in methods:
        for b_data in results[m].values():
            tasks.update(b_data.keys())
    tasks = sorted(tasks)

    print(f"\n{'='*100}")
    print(f"  LongBench Evaluation Results")
    print(f"{'='*100}")

    for task in tasks:
        cfg = TASK_CONFIGS.get(task, {})
        metric = cfg.get("metric", "?")
        category = cfg.get("category", "?")
        print(f"\n  ── {task}  ({category}, metric={metric}) ──")

        # Header row
        header = f"  {'Method':<16s}"
        for b in budget_labels:
            header += f" {'B=' + b:>10s}"
        header += "  Δ vs Full"
        print(header)
        print(f"  {'-'*16}{'  ' + '-'*10 * len(budget_labels)}{'  ' + '-'*10}")

        # Full reference
        full_mean = None
        if "full" in results and "full" in results["full"]:
            full_data = results["full"]["full"].get(task)
            if full_data:
                full_mean = full_data["mean"]

        for method in methods:
            row = f"  {method:<16s}"
            for b in budget_labels:
                method_data = results[method].get(b, {}).get(task)
                if method_data:
                    row += f" {method_data['mean']:>10.4f}"
                else:
                    row += " " * 11

            # Delta vs full
            if full_mean is not None:
                method_data = results[method].get(
                    "full" if method == "full" else budget_labels[0],
                    {}
                ).get(task)
                if method_data:
                    delta = method_data["mean"] - full_mean
                    row += f"  {delta:>+10.4f}"
                else:
                    row += " " * 12
            elif method == "full":
                row += " " * 12  # no delta for full vs itself

            print(row)

    # Overall averages
    print(f"\n  ── Average over all tasks ──")
    header = f"  {'Method':<16s}"
    for b in budget_labels:
        header += f" {'B=' + b:>10s}"
    print(header)
    print(f"  {'-'*16}{'  ' + '-'*10 * len(budget_labels)}")

    for method in methods:
        row = f"  {method:<16s}"
        for b in budget_labels:
            method_data = results[method].get(b, {})
            task_means = [
                d["mean"] for d in method_data.values()
                if isinstance(d, dict) and "mean" in d
            ]
            if task_means:
                avg = sum(task_means) / len(task_means)
                row += f" {avg:>10.4f}"
            else:
                row += " " * 11
        print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LongBench evaluation — Full vs Quest vs Hierarchical"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help="LongBench tasks to evaluate (default: all available)",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[128, 256],
        help="Token budgets to sweep (default: 128 256)",
    )
    parser.add_argument(
        "--page-size", type=int, default=64,
        help="Tokens per page (default: 64)",
    )
    parser.add_argument(
        "--macro-multiplier", type=int, default=3,
        help="Hierarchical macro multiplier (default: 3)",
    )
    parser.add_argument(
        "--num-sink", type=int, default=4,
        help="Sink tokens for hierarchical (default: 4)",
    )
    parser.add_argument(
        "--num-recent", type=int, default=64,
        help="Recent tokens for hierarchical (default: 64)",
    )
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="Disable adaptive macro budget",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50,
        help="Max samples per task (default: 50)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens to generate per sample (default: 64)",
    )
    parser.add_argument(
        "--max-prompt-tokens", type=int, default=None,
        help="Truncate prompts longer than this",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sample progress",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="Print available tasks and exit",
    )

    args = parser.parse_args()

    if args.list_tasks:
        from longbench_utils import describe_tasks
        print(describe_tasks())
        return

    # Validate requested tasks
    if args.tasks is not None:
        available = set(TASK_CONFIGS.keys())
        unknown = set(args.tasks) - available
        if unknown:
            print(f"Unknown tasks: {sorted(unknown)}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            sys.exit(1)

    # Default output path
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(
            ROOT / "reports" / f"longbench_results_{timestamp}.json"
        )

    print(f"LongBench Evaluation")
    print(f"  Model:    {args.model}")
    print(f"  Tasks:    {args.tasks or 'ALL'}")
    print(f"  Budgets:  {args.budgets}")
    print(f"  Samples:  {args.max_samples}")
    print(f"  Gen:      {args.max_new_tokens} max new tokens")
    print(f"  Device:   {args.device}  ({args.dtype})")
    print(f"  Output:   {output_file}")

    results = run_longbench_sweep(
        model_name=args.model,
        tasks=args.tasks,
        budgets=args.budgets,
        page_size=args.page_size,
        macro_multiplier=args.macro_multiplier,
        num_sink_tokens=args.num_sink,
        num_recent_tokens=args.num_recent,
        adaptive_budget=not args.no_adaptive,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        device_str=args.device,
        dtype_str=args.dtype,
        output_file=output_file,
        quiet=args.quiet,
    )

    print_results_table(results)

    print(f"\nDone — results saved to {output_file}")


if __name__ == "__main__":
    main()
