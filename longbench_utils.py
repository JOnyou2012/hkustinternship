"""
LongBench evaluation utilities — dataset loading, prompt templates, metrics.

Supports a representative subset of LongBench tasks spanning single-doc QA,
multi-doc QA, summarisation, few-shot classification, and synthetic retrieval.

All tasks are loaded from ``THUDM/LongBench`` on HuggingFace.  Each task
config specifies the dataset subset name, the metric, and the prompt template
used to format inputs for the model.

Usage::

    from longbench_utils import load_task_dataset, format_prompt, compute_metric

    samples = load_task_dataset("narrativeqa", max_samples=50)
    prompt = format_prompt("narrativeqa", samples[0])
    score = compute_metric("narrativeqa", prediction, samples[0])
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing helpers (must be above TASK_CONFIGS which references them)
# ═══════════════════════════════════════════════════════════════════════════════

def _postprocess_trec(raw: str) -> str:
    """Extract the TREC class label from raw output."""
    match = re.search(r"\b(ABBR|DESC|ENTY|HUM|LOC|NUM)\b", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return raw.strip().upper().split()[0] if raw.strip() else ""


def _postprocess_number(raw: str) -> str:
    """Extract the first integer from raw output."""
    match = re.search(r"\d+", raw)
    return match.group(0) if match else raw.strip()


def _postprocess_qa(raw: str) -> str:
    """Clean QA output — take text up to first newline or period-sentence."""
    raw = raw.strip()
    first_line = raw.split("\n")[0].strip()
    if len(first_line) > 10:
        return first_line
    return raw[:200]


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry maps a short task name to:
#   subset       — configuration name in the THUDM/LongBench dataset
#   metric       — "f1" | "rouge_l" | "accuracy" | "exact_match"
#   max_context  — truncate context longer than this (None = keep all)
#   prompt       — format-string template; available keys: {context}, {input}
#   postprocess  — optional function to clean raw model output before scoring
#   category     — for grouping in result tables

TASK_CONFIGS: Dict[str, dict] = {
    # ── Single-Document QA ──────────────────────────────────────────────
    "narrativeqa": {
        "subset": "narrativeqa",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "You are given a story and a question. Read the story carefully "
            "and answer the question in a few words.\n\n"
            "Story:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Single-Doc QA",
    },
    "qasper": {
        "subset": "qasper",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "You are given a scientific article and a question. Answer the "
            "question based on the article in a few words.\n\n"
            "Article:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Single-Doc QA",
    },
    "multifieldqa_en": {
        "subset": "multifieldqa_en",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "Read the following text and answer the question in a few words.\n\n"
            "{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Single-Doc QA",
    },

    # ── Multi-Document QA ───────────────────────────────────────────────
    "hotpotqa": {
        "subset": "hotpotqa",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "Answer the question based on the given passages.\n\n"
            "{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Multi-Doc QA",
    },
    "2wikimqa": {
        "subset": "2wikimqa",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "Answer the question based on the given passages.\n\n"
            "{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Multi-Doc QA",
    },
    "musique": {
        "subset": "musique",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "Answer the question based on the given passages.\n\n"
            "{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Multi-Doc QA",
    },

    # ── Summarisation ───────────────────────────────────────────────────
    "gov_report": {
        "subset": "gov_report",
        "metric": "rouge_l",
        "max_context": 8192,
        "prompt": (
            "Summarise the following government report in a few sentences.\n\n"
            "{input}\n\n"
            "Summary:"
        ),
        "postprocess": None,
        "category": "Summarisation",
    },
    "qmsum": {
        "subset": "qmsum",
        "metric": "rouge_l",
        "max_context": 8192,
        "prompt": (
            "Summarise the following meeting transcript.\n\n"
            "{input}\n\n"
            "Summary:"
        ),
        "postprocess": None,
        "category": "Summarisation",
    },
    "multi_news": {
        "subset": "multi_news",
        "metric": "rouge_l",
        "max_context": 8192,
        "prompt": (
            "Summarise the following news articles in a few sentences.\n\n"
            "{input}\n\n"
            "Summary:"
        ),
        "postprocess": None,
        "category": "Summarisation",
    },

    # ── Few-Shot ────────────────────────────────────────────────────────
    "trec": {
        "subset": "trec",
        "metric": "accuracy",
        "max_context": None,
        "prompt": "{input}",  # few-shot examples are already in the input field
        "postprocess": _postprocess_trec,
        "category": "Few-Shot",
    },
    "triviaqa": {
        "subset": "triviaqa",
        "metric": "f1",
        "max_context": 8192,
        "prompt": (
            "Answer the following trivia question in a few words.\n\n"
            "{input}\n\n"
            "Answer:"
        ),
        "postprocess": None,
        "category": "Few-Shot",
    },
    "samsum": {
        "subset": "samsum",
        "metric": "rouge_l",
        "max_context": 4096,
        "prompt": (
            "Summarise the following dialogue in a few sentences.\n\n"
            "{input}\n\n"
            "Summary:"
        ),
        "postprocess": None,
        "category": "Few-Shot",
    },

    # ── Synthetic ───────────────────────────────────────────────────────
    "passage_count": {
        "subset": "passage_count",
        "metric": "accuracy",
        "max_context": 8192,
        "prompt": (
            "There are many passages below. Count the total number of "
            "passages and respond with just the number.\n\n"
            "{context}\n\n"
            "Number of passages:"
        ),
        "postprocess": _postprocess_number,
        "category": "Synthetic",
    },
    "passage_retrieval_en": {
        "subset": "passage_retrieval_en",
        "metric": "accuracy",
        "max_context": 8192,
        "prompt": (
            "Here are several passages. Find the one that is most relevant "
            "to the summary below and give its number.\n\n"
            "{context}\n\n"
            "Summary: {input}\n\n"
            "Most relevant passage number:"
        ),
        "postprocess": _postprocess_number,
        "category": "Synthetic",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_task_dataset(
    task_name: str,
    max_samples: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    tokenizer=None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load a LongBench task dataset from HuggingFace.

    Args:
        task_name:   One of the keys in ``TASK_CONFIGS``.
        max_samples: Cap the number of samples (None = all).
        max_context_tokens: Truncate context to this many tokens (None = use
                     the task config default).
        tokenizer:   Tokenizer for token-count-based truncation (None = no
                     truncation, keep full text).
        seed:        Random seed for shuffling before truncation.

    Returns:
        List of dicts, each with keys:
            - input:    str — the question / instruction
            - context:  str — the long document(s)
            - answers:  List[str] — reference answers
            - all_classes: Optional[List[str]] — for classification tasks
            - length:   str — "short", "medium", or "long"
            - _id:      str — sample identifier

    Raises:
        ImportError: If ``datasets`` is not installed.
        ValueError: If the task is unknown.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for LongBench evaluation.  "
            "Install it with:  pip install datasets"
        )

    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'.  "
            f"Available: {sorted(TASK_CONFIGS.keys())}"
        )

    cfg = TASK_CONFIGS[task_name]

    dataset = load_dataset("THUDM/LongBench", cfg["subset"], split="test")

    # Shuffle with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=seed)

    samples: List[Dict[str, Any]] = []
    for example in dataset:
        sample = {
            "input": example.get("input", ""),
            "context": example.get("context", ""),
            "answers": example.get("answers", []),
            "all_classes": example.get("all_classes", None),
            "length": example.get("length", "medium"),
            "_id": example.get("_id", ""),
        }

        # Truncate long context if requested
        context_limit = max_context_tokens or cfg.get("max_context")
        if context_limit and tokenizer is not None and sample["context"]:
            context_tokens = tokenizer.encode(sample["context"])
            if len(context_tokens) > context_limit:
                context_tokens = context_tokens[:context_limit]
                sample["context"] = tokenizer.decode(
                    context_tokens, skip_special_tokens=True
                )
        elif context_limit and sample["context"]:
            # Character-based rough truncation
            if len(sample["context"]) > context_limit * 4:
                sample["context"] = sample["context"][: context_limit * 4]

        samples.append(sample)

        if max_samples and len(samples) >= max_samples:
            break

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt formatting
# ═══════════════════════════════════════════════════════════════════════════════

def format_prompt(task_name: str, sample: Dict[str, Any]) -> str:
    """Format a LongBench sample into a model-ready prompt string.

    Args:
        task_name: Task key from TASK_CONFIGS.
        sample:    A dict from ``load_task_dataset``.

    Returns:
        Prompt string ready for tokenisation.
    """
    cfg = TASK_CONFIGS[task_name]
    template = cfg["prompt"]

    # Some tasks embed the context in the input field (e.g. summarisation)
    context = sample.get("context", "")
    inp = sample.get("input", "")

    return template.format(context=context, input=inp)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace."""
    text = text.lower().strip()
    # Remove articles and punctuation for F1 matching
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenisation after normalisation."""
    return _normalize(text).split()


def compute_f1(prediction: str, references: List[str]) -> float:
    """Word-level F1 score, max over reference answers.

    Args:
        prediction: Model's predicted answer string.
        references: List of ground-truth answer strings.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return best_f1


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    """ROUGE-L F-score via longest common subsequence, max over references.

    Uses a simple LCS implementation (no external ``rouge_score`` dependency).
    For evaluation-scale use, install ``rouge_score`` and replace with the
    library version for exact parity with LongBench official scores.

    Args:
        prediction: Model's predicted summary string.
        references: List of ground-truth summary strings.

    Returns:
        ROUGE-L F-score in [0, 1].
    """
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0

    best_f = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue

        lcs_len = _lcs_length(pred_tokens, ref_tokens)

        if lcs_len == 0:
            continue

        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)

        beta = 1.0  # standard ROUGE-L beta
        if precision + recall > 0:
            f = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
            best_f = max(best_f, f)

    return best_f


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of longest common subsequence (space-optimised DP)."""
    if len(a) < len(b):
        a, b = b, a
    # dp[j] = LCS length for a_prefix vs b[:j]
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[len(b)]


def compute_accuracy(
    prediction: str,
    references: List[str],
    all_classes: Optional[List[str]] = None,
) -> float:
    """Classification accuracy — prediction matches any reference after normalisation.

    Args:
        prediction:  Model's predicted class / answer.
        references:  List of acceptable ground-truth strings.
        all_classes: Optional list of all class labels (for constrained matching).

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    pred_norm = _normalize(prediction)

    for ref in references:
        if _normalize(ref) == pred_norm:
            return 1.0

    # Check if prediction is a substring of any reference or vice versa
    for ref in references:
        ref_norm = _normalize(ref)
        if pred_norm in ref_norm or ref_norm in pred_norm:
            return 1.0

    return 0.0


def compute_metric(
    task_name: str,
    prediction: str,
    sample: Dict[str, Any],
) -> float:
    """Dispatch to the correct metric for this task.

    Applies task-specific post-processing to the raw prediction before scoring.

    Args:
        task_name:  Key in TASK_CONFIGS.
        prediction: Raw model output string.
        sample:     The LongBench sample dict (contains "answers" and
                    optionally "all_classes").

    Returns:
        Score in [0, 1] (higher is better).
    """
    cfg = TASK_CONFIGS[task_name]
    metric_name = cfg["metric"]

    # Apply task-specific post-processing
    postprocess = cfg.get("postprocess")
    if postprocess is not None:
        prediction = postprocess(prediction)
    else:
        prediction = _postprocess_qa(prediction)

    references = sample.get("answers", [])

    if not references:
        return 0.0

    if metric_name == "f1":
        return compute_f1(prediction, references)
    elif metric_name == "rouge_l":
        return compute_rouge_l(prediction, references)
    elif metric_name == "accuracy":
        all_classes = sample.get("all_classes", None)
        return compute_accuracy(prediction, references, all_classes)
    elif metric_name == "exact_match":
        pred_norm = _normalize(prediction)
        return 1.0 if any(_normalize(r) == pred_norm for r in references) else 0.0
    else:
        raise ValueError(f"Unknown metric '{metric_name}' for task '{task_name}'")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for the evaluation harness
# ═══════════════════════════════════════════════════════════════════════════════

def get_task_list(category: Optional[str] = None) -> List[str]:
    """Return all available task names, optionally filtered by category.

    Args:
        category: One of "Single-Doc QA", "Multi-Doc QA", "Summarisation",
                  "Few-Shot", "Synthetic", or None for all tasks.

    Returns:
        Sorted list of task name strings.
    """
    if category is None:
        return sorted(TASK_CONFIGS.keys())
    return sorted(
        k for k, v in TASK_CONFIGS.items() if v.get("category") == category
    )


def get_task_categories() -> List[str]:
    """Return the distinct task categories available."""
    seen = {}
    for v in TASK_CONFIGS.values():
        seen[v.get("category", "Other")] = True
    return list(seen.keys())


def describe_tasks() -> str:
    """Return a human-readable summary of available tasks."""
    lines = ["LongBench tasks available:"]
    for cat in get_task_categories():
        tasks = get_task_list(cat)
        lines.append(f"\n  {cat} ({len(tasks)} tasks):")
        for t in tasks:
            cfg = TASK_CONFIGS[t]
            lines.append(f"    {t:<22s}  metric={cfg['metric']:<10s}  "
                         f"max_ctx={cfg.get('max_context', 'full')}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(describe_tasks())

    # Smoke-test metrics
    print("\n--- Metric smoke tests ---")
    print(f"  F1: {compute_f1('the cat sat on mat', ['The cat sat on the mat.']):.3f}")
    print(f"  R-L: {compute_rouge_l('the cat sat', ['The cat sat on the mat.']):.3f}")
    print(f"  Acc: {compute_accuracy('desC', ['DESC'])}")

    # Smoke-test data loading (first 2 samples from a small task)
    print("\n--- Data loading smoke test (samsum, 2 samples) ---")
    try:
        samples = load_task_dataset("samsum", max_samples=2)
        for i, s in enumerate(samples):
            print(f"  [{i}] input={s['input'][:80]}...  answers={s['answers']}")
            print(f"      prompt starts: {format_prompt('samsum', s)[:120]}...")
    except ImportError:
        print("  (skipped — 'datasets' not installed)")
    except Exception as e:
        print(f"  (skipped — {e})")
