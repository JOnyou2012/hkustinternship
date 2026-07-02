"""
Perplexity evaluation for sparse attention on real text.

Evaluates three configurations on WikiText-2:
  1. Full (dense) attention — unpatched baseline
  2. Quest sparse attention — page-wise selection
  3. Hierarchical sparse attention — token-level refinement

All sparse methods operate at the same token budget (B = top_k × page_size),
so any quality difference comes purely from better token selection.

Usage::

    python eval_perplexity.py
    python eval_perplexity.py --model gpt2 --dataset wikitext --budget 256
    python eval_perplexity.py --chunk-size 512 --max-samples 50
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

from hf_model_patcher import patch_model


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_wikitext2(tokenizer, max_samples: int = 100, chunk_size: int = 1024):
    """Load WikiText-2 test set, tokenize into fixed-length chunks.

    Args:
        tokenizer:   HF tokenizer (must match the model).
        max_samples: Maximum number of chunks to evaluate (limit for speed).
        chunk_size:  Tokens per chunk (should not exceed model max context).

    Returns:
        List of tokenised chunks as LongTensors, each of shape (chunk_size,).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate all text and tokenize
    texts: List[str] = []
    for example in dataset:
        text = example["text"]
        if text.strip():  # skip empty lines / section headers
            texts.append(text)

    all_text = tokenizer.eos_token.join(texts)

    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]  # (total_len,)

    # Chunk into fixed-length segments
    chunks: List[torch.Tensor] = []
    for i in range(0, len(tokens) - chunk_size, chunk_size):
        chunks.append(tokens[i : i + chunk_size])
        if max_samples and len(chunks) >= max_samples:
            break

    if not chunks:
        # At least one chunk
        chunks.append(tokens[:chunk_size])

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Perplexity computation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(
    model,
    chunks: List[torch.Tensor],
    device: torch.device,
    label: str = "",
) -> Tuple[float, float]:
    """Compute perplexity by summing cross-entropy loss over chunks.

    Uses the standard formulation::

        PPL = exp( total_loss / total_tokens )

    The model's internal causal mask prevents each position from attending
    to future positions.  Loss is computed only on non-padding tokens.

    Args:
        model:  HF causal LM in eval mode.
        chunks: List of token Tensors, each (chunk_size,).
        device: torch device.
        label:  Optional label for progress printing.

    Returns:
        (perplexity, tokens_per_second)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.perf_counter()

    for i, chunk in enumerate(chunks):
        input_ids = chunk.unsqueeze(0).to(device)  # (1, chunk_size)
        B, T = input_ids.shape

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        if loss is not None and not torch.isnan(loss):
            # HF loss is already averaged per-token by default
            total_loss += loss.item() * T
        else:
            # Manual cross-entropy if model doesn't return loss
            logits = outputs.logits  # (B, T, vocab_size)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += ce.item()

        total_tokens += T

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - start_time
            rate = total_tokens / max(elapsed, 0.01)
            current_ppl = math.exp(total_loss / max(total_tokens, 1))
            print(f"  [{label}] chunk {i+1}/{len(chunks)} | "
                  f"PPL={current_ppl:.2f} | {rate:.0f} tok/s", end="\r")

    elapsed = time.perf_counter() - start_time
    perplexity = math.exp(total_loss / max(total_tokens, 1))
    tok_per_sec = total_tokens / max(elapsed, 0.01)

    print(f"  [{label}] {len(chunks)} chunks | "
          f"PPL={perplexity:.3f} | {tok_per_sec:.1f} tok/s" + " " * 20)

    return perplexity, tok_per_sec


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep — Compare Full vs Quest vs Hierarchical
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_name: str = "gpt2",
    dataset_name: str = "wikitext2",
    chunk_size: int = 512,
    max_samples: int = 100,
    device_str: str = "cpu",
    page_size: int = 64,
    top_k: int = 4,
    macro_multiplier: int = 3,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 64,
    adaptive_budget: bool = True,
    quiet: bool = False,
) -> Dict[str, dict]:
    """Run the full comparison: Full vs Quest vs Hierarchical on real text.

    Returns:
        Dict mapping method name → {"perplexity": float, "tokens_per_sec": float}
    """
    device = torch.device(device_str)
    token_budget = top_k * page_size

    if not quiet:
        print(f"{'='*65}")
        print(f"  Perplexity Evaluation — Real-Model Sparse Attention")
        print(f"  Model: {model_name}  Dataset: {dataset_name}")
        print(f"  Chunk size: {chunk_size}  Max samples: {max_samples}")
        print(f"  Token budget: {token_budget} (top_k={top_k} × page_size={page_size})")
        print(f"{'='*65}")

    # ---- Load tokenizer (shared across runs) ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load data ----
    if dataset_name in ("wikitext2", "wikitext"):
        chunks = load_wikitext2(tokenizer, max_samples=max_samples,
                                 chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not quiet:
        print(f"  Loaded {len(chunks)} chunks of {chunk_size} tokens each\n")

    # ---- Determine model dims ----
    config = GPT2Config.from_pretrained(model_name)
    hidden_dim = config.hidden_size
    num_heads = config.n_head
    head_dim = hidden_dim // num_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    results: Dict[str, dict] = {}

    # ── 1. Full (unpatched) baseline ─────────────────────────────────────
    if not quiet:
        print("  [1/3] Full attention (unpatched baseline)")

    model_full = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_full.eval()

    ppl_full, tps_full = compute_perplexity(
        model_full, chunks, device, label="full"
    )
    results["full"] = {"perplexity": ppl_full, "tokens_per_sec": tps_full}

    del model_full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 2. Quest (page-wise sparse) ──────────────────────────────────────
    if not quiet:
        print("\n  [2/3] Quest page-wise sparse attention")

    model_quest = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_quest.eval()

    patch_model(
        model_quest,
        method="quest",
        page_size=page_size,
        top_k=top_k,
        bias=True,
    )

    ppl_quest, tps_quest = compute_perplexity(
        model_quest, chunks, device, label="quest"
    )
    results["quest"] = {"perplexity": ppl_quest, "tokens_per_sec": tps_quest}

    del model_quest
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 3. Hierarchical (token-level sparse) ─────────────────────────────
    if not quiet:
        print("\n  [3/3] Hierarchical token-level sparse attention")

    model_hier = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model_hier.eval()

    patch_model(
        model_hier,
        method="hierarchical",
        page_size=page_size,
        top_k=top_k,
        macro_multiplier=macro_multiplier,
        num_sink_tokens=num_sink_tokens,
        num_recent_tokens=num_recent_tokens,
        adaptive_budget=adaptive_budget,
        bias=True,
    )

    ppl_hier, tps_hier = compute_perplexity(
        model_hier, chunks, device, label="hier"
    )
    results["hierarchical"] = {
        "perplexity": ppl_hier,
        "tokens_per_sec": tps_hier,
    }

    del model_hier
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────
    if not quiet:
        print(f"\n{'='*65}")
        print(f"  Results (token budget = {token_budget})")
        print(f"{'='*65}")
        print(f"  {'Method':<20s} {'PPL':>8s}  {'Δ vs Full':>10s}  "
              f"{'tok/s':>8s}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*8}")

        ppl_ref = results["full"]["perplexity"]
        for method in ["full", "quest", "hierarchical"]:
            r = results[method]
            ppl = r["perplexity"]
            delta = ppl - ppl_ref
            print(f"  {method:<20s} {ppl:>8.3f}  {delta:>+10.3f}  "
                  f"{r['tokens_per_sec']:>8.1f}")

        improvement = (results["quest"]["perplexity"] -
                       results["hierarchical"]["perplexity"])
        print(f"\n  Hierarchical vs Quest improvement: "
              f"{improvement:+.3f} PPL (lower is better)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation for sparse attention on real text",
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HF model name (default: gpt2)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "wikitext"],
                        help="Dataset to evaluate on")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Tokens per chunk (default: 512)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max chunks to evaluate (default: 100)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--page-size", type=int, default=64,
                        help="Tokens per page (default: 64)")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Pages to select (default: 4, budget = 256)")
    parser.add_argument("--macro-multiplier", type=int, default=3,
                        help="Hierarchical macro multiplier (default: 3)")
    parser.add_argument("--num-sink", type=int, default=4,
                        help="Sink tokens (default: 4)")
    parser.add_argument("--num-recent", type=int, default=64,
                        help="Recent tokens (default: 64)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive macro budget")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    results = run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        device_str=args.device,
        page_size=args.page_size,
        top_k=args.top_k,
        macro_multiplier=args.macro_multiplier,
        num_sink_tokens=args.num_sink,
        num_recent_tokens=args.num_recent,
        adaptive_budget=not args.no_adaptive,
        quiet=args.quiet,
    )

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
