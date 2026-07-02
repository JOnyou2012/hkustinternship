# Quest Baseline — Hierarchical Token-Level Sparse Attention

Reference implementation reproducing the **Quest** page-wise sparse attention
algorithm and extending it with **quality-oriented hierarchical token-level
refinement**. iSING Lab, HKUST.

---

## Overview

Standard full attention scales quadratically $O(T^2)$ with sequence length,
creating two critical bottlenecks at decode time:

1. **KV-cache capacity** — storing the full key/value history for long
   contexts exhausts GPU memory.
2. **Full-attention latency** — re-reading the entire KV cache at every
   decode step becomes prohibitively slow.

This project implements and benchmarks two sparse attention approaches that
address both bottlenecks by attending to only the most relevant tokens at
each decode step. A HuggingFace patcher allows these methods to replace
standard attention in real pretrained models.

---

### Phase 1 — Quest Page-Wise Baseline

Divides the KV cache into fixed-size *pages*, precomputes per-page key
metadata (element-wise min/max), and selects only the **Top-K** pages for
attention. Sparse attention runs on exactly $K \times \text{page\_size}$
tokens regardless of the full KV-cache size.

### Phase 2 — Hierarchical Token-Level Refinement

Upgrades Quest with a **three-stage hierarchical filtering pipeline** that
selects individual tokens rather than coarse pages, operating under the
**exact same attention budget** ($B = K \times \text{page\_size}$):

1. **Macro-Selection** — Quest page scoring → Top-M pages (wider net).
2. **Micro-Selection** — Exact Q·K scoring of every token inside those pages.
3. **Consolidation** — Sink/recent token protection + Top-B token selection.

Phase 2 achieves **34–130% better cosine similarity** than Phase 1 at the
same budget, with token recall@B reaching 87% at 512 tokens.

### HuggingFace Integration

Standalone kernel functions extract the Q/K/V routing so sparse attention
can replace standard attention layers in any HuggingFace model. A model
patcher handles weight transfer and layer replacement for 9 model families
(GPT-2, Llama, Mistral, Qwen2, Gemma, Falcon, Phi, Phi-3, OPT).

---

## Mathematical Foundation

$$Attention(Q,K,V)=\text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

**Phase 1 (Quest)** restricts the softmax to the Top-K pages:

$$Attention_{Quest}(Q,K,V) \approx
\text{softmax}\!\left(\frac{Q K_{topk}^T}{\sqrt{d}}\right) V_{topk}$$

**Phase 2 (Hierarchical)** refines this to individual tokens from a wider
candidate pool:

$$Attention_{Hier}(Q,K,V) =
\text{softmax}\!\left(\frac{Q K_{topB}^T}{\sqrt{d}}\right) V_{topB}$$

where $\text{topB}$ are the $B = K \times P_{size}$ best tokens selected
from $M > K$ pages, with sink and recent tokens always protected.

---

## Project Structure

```
├── config.py                    # Configuration dataclasses (model, Quest, Phase 2, metrics)
├── utils.py                     # GPU timer, memory tracking, cosine similarity, formatting
├── full_attention.py            # Standard multi-head attention — the baseline
│
├── quest_attention.py           # Phase 1: Quest page-wise sparse attention module
├── hierarchical_attention.py    # Phase 2: Hierarchical token-level attention module
├── sparse_attention_kernels.py  # Standalone kernel functions (no nn.Module) for both phases
│                                #   _build_pages, quest_sparse_attention,
│                                #   hierarchical_sparse_attention
│
├── experiment.py                # Benchmark harness (Phase 1 + Phase 2) + correctness
├── run_benchmark.py             # Phase 1 CLI
├── run_benchmark_phase2.py      # Phase 2 CLI with side-by-side comparison
│
├── test_quest.py                # Phase 1 unit tests (19 tests)
├── test_phase2.py               # Phase 2 unit tests (27 tests)
│
├── hf_model_patcher.py          # HuggingFace attention layer replacement — 9 model families
├── test_hf_patcher.py           # HF patcher unit tests (18 tests)
├── eval_perplexity.py            # Perplexity evaluation on WikiText-2
├── plot_real_model_results.py   # Perplexity figure generation
│
├── plot_results.py              # Generate core Phase-1-vs-Phase-2 quality figure
├── talk_script.md               # 5-minute presentation script
│
├── reports/
│   ├── phase1_summary.pdf       # Phase 1 technical report
│   ├── phase1_summary.tex       # Phase 1 report source
│   ├── quest_explained_simply.pdf        # Phase 1 simple explanation
│   ├── quest_explained_simply.tex
│   ├── phase2_summary.pdf       # Phase 2 technical report
│   ├── phase2_summary.tex       # Phase 2 report source
│   ├── hierarchical_explained_simply.pdf # Phase 2 simple explanation
│   ├── hierarchical_explained_simply.tex
│   ├── phase2_quality_vs_budget.pdf      # Core result figure
│   ├── longbench_evaluation_demo.pdf     # LongBench evaluation demo
│   └── (more generated plots)
│
└── README.md
```

---

## Quick Start

### Install dependencies

```bash
pip install torch matplotlib
```

For real-model evaluation:
```bash
pip install transformers datasets
```

### Run Phase 1 benchmark (Quest baseline)

```bash
# CPU
python run_benchmark.py --device cpu --kv-lens 512 1024 2048 4096

# GPU
python run_benchmark.py --device cuda --kv-lens 512 1024 2048 4096 8192
```

### Run Phase 2 benchmark (hierarchical token-level)

```bash
# CPU
python run_benchmark_phase2.py --device cpu

# With Phase 1 vs Phase 2 side-by-side comparison
python run_benchmark_phase2.py --device cpu --compare-baseline
```

### Run all unit tests

```bash
python test_quest.py          # 19 tests — Phase 1 (Quest)
python test_phase2.py         # 27 tests — Phase 2 (Hierarchical)
python test_hf_patcher.py     # 18 tests — HuggingFace patcher
```

### Generate the core result figure

```bash
# Auto-runs benchmarks and generates the quality-vs-budget plot
python plot_results.py

# Use cached data from a previous run (faster)
python plot_results.py --cached
```

### Evaluate on a real model (HuggingFace)

```bash
# Perplexity comparison on WikiText-2 (GPT-2 by default)
python eval_perplexity.py --model gpt2 --max-samples 50
```

---

## CLI Flags (Phase 2)

| Flag | Default | Description |
|------|---------|-------------|
| `--num-heads` | 32 | Number of query heads |
| `--head-dim` | 128 | Dimension per head |
| `--num-kv-heads` | 32 | KV heads (set fewer for GQA) |
| `--page-size` | 64 | Tokens per page |
| `--top-k` | 4 | Reference page count; token budget B = K × page_size |
| `--macro-multiplier` | 3 | M = multiplier × top_k macro pages |
| `--num-sink` | 4 | Force-protected initial sink tokens |
| `--num-recent` | 64 | Force-protected trailing recent tokens |
| `--no-adaptive` | (off) | Disable adaptive macro-budget sizing |
| `--token-budget` | None | Explicit Top-B budget (default: top_k × page_size) |
| `--kv-lens` | 512 1024 2048 4096 8192 | KV-cache sizes to sweep |
| `--num-warmup` | 10 | Warmup iterations |
| `--num-benchmark` | 50 | Timed iterations per data point |
| `--device` | cuda | `cuda` or `cpu` |
| `--dtype` | float16 | `float16`, `float32`, or `bfloat16` |
| `--no-verify` | (off) | Skip cosine-similarity verification |
| `--no-quality` | (off) | Skip token-recall and overlap metrics |
| `--compare-baseline` | (off) | Run Phase 1 side-by-side with Phase 2 |

---

## Key Results

Both methods operate under the **exact same token budget** ($B = 256$ tokens
per decode step, i.e. $K=4$ pages $\times$ 64 tokens/page).

| KV Length | P1 CosSim (Quest) | P2 CosSim (Hierarchical) | Improvement |
|-----------|-------------------|--------------------------|-------------|
| 512       | 0.735             | **0.983**                | **+34%**    |
| 1,024     | 0.501             | **0.937**                | **+87%**    |
| 2,048     | 0.379             | **0.794**                | **+110%**   |
| 4,096     | 0.249             | **0.554**                | **+123%**   |
| 8,192     | 0.178             | **0.410**                | **+130%**   |

Phase 2 token recall@256: 87% at 512 tokens → 75% at 2,048 → 20% at 8,192.

---

## Reproduction Instructions

### Environment

```bash
# Create conda environment
conda create -n quest python=3.11 -y
conda activate quest

# Install PyTorch (CPU or CUDA)
pip install torch          # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Install plotting dependency
pip install matplotlib

# For HF model evaluation
pip install transformers datasets
```

### Step 1 — Verify Phase 1 baseline

```bash
python test_quest.py
# Expected: 19 tests pass, OK

python run_benchmark.py --device cpu --kv-lens 512 1024 2048 --num-benchmark 20
```

### Step 2 — Verify Phase 2 optimization

```bash
python test_phase2.py
# Expected: 27 tests pass, OK

python run_benchmark_phase2.py --device cpu --kv-lens 512 1024 2048 \
    --num-benchmark 20 --num-warmup 3
```

### Step 3 — Reproduce the core comparison

```bash
# Full side-by-side Phase 1 vs Phase 2
python run_benchmark_phase2.py --device cpu --compare-baseline \
    --num-benchmark 30 --num-warmup 5

# Generate the figure from benchmark data
python plot_results.py
```

### Step 4 — Verify HuggingFace integration

```bash
python test_hf_patcher.py
# Expected: 18 tests pass, OK

# Real-model perplexity evaluation
python eval_perplexity.py --model gpt2 --max-samples 10
```

---

## Implementation Details

### Phase 1 — Quest Page-Wise Sparse Attention

```
Input:  Q (1 decode query), KV cache of T tokens
Output: Attention output over Top-K pages (K × page_size tokens)

1. Build pages: partition K/V into fixed-size page_size pages
2. Page metadata: compute element-wise min/max per page
3. Score pages: (Q · K_min) and (Q · K_max) → channel-max → sum → scalar/page
4. Select Top-K pages by score
5. Gather selected K/V pages → scaled dot-product attention
```

### Phase 2 — Hierarchical Token-Level Attention

```
Input:  Q (1 decode query), KV cache of T tokens
Output: Attention output (B = top_k × page_size tokens)

STAGE 1 — MACRO-SELECTION (Page-Level)
  1. Quest page scoring as in Phase 1
  2. Adaptive budget: size M from page-score concentration
  3. Select Top-M pages; force-include sink & recent pages

STAGE 2 — MICRO-SELECTION (Token-Level)
  4. Gather all tokens from M selected pages
  5. Compute exact pre-softmax score: score(t) = (Q · K_t) / √d
  6. Mask padding tokens → score = -inf

STAGE 3 — CONSOLIDATION
  7. Mark sink tokens (positions 0..num_sink-1) as +inf
  8. Mark recent tokens (positions T-num_recent..T-1) as +inf
  9. Top-B select → exactly B = top_k × page_size tokens

SPARSE ATTENTION
 10. Gather selected tokens from padded KV cache by global position
 11. Scaled dot-product attention with padding mask
 12. Merge heads → output projection
```

### HuggingFace Integration

```
1. Detect model type from config (~model_type)
2. Navigate to the transformer layers list via a model-specific path
3. For each layer:
   a. Read existing attention module's Q/K/V/O projection weights
   b. Transfer weights: GPT-2 Conv1D → nn.Linear or standard → standard
   c. Replace with _SparseAttentionWrapper(quest/hierarchical module)
4. Return patched model ready for forward pass
```

The standalone kernels in `sparse_attention_kernels.py` operate on
**already-projected** Q/K/V tensors with no `nn.Module` dependency, enabling
clean interception: extract Q/K/V from any HF model → route through the
sparse kernel → plug the output back.

### Key Design Decisions

- **Budget invariant**: $B = \text{top\_k} \times \text{page\_size}$ is
  guaranteed. Quality gains come purely from better token selection.
- **Per-head independent selection**: Each attention head selects its own
  Top-B tokens, preserving heterogeneous specialisation.
- **Force-include with no duplication**: `select_macro_pages` uses
  per-element `torch.where`-based conditional replacement.
- **Clamped protection**: Sink/recent counts are clamped to the actual KV
  length to prevent over-protection on short sequences.
- **Module interface compatibility**: `HierarchicalTokenAttention`,
  `QuestAttention`, and `MultiHeadFullAttention` share the same
  `(hidden_states, mask, *, return_kv)` interface.
- **Kernel separation**: Standalone kernels (`quest_sparse_attention`,
  `hierarchical_sparse_attention`) operate on pre-projected Q/K/V, enabling
  integration with any model that provides its own projections.

---

## References

- **Quest**: Tang et al. *Quest: Query-Aware Sparsity for Efficient
  Long-Context LLM Inference.* arXiv:2406.10774, 2024.
- **StreamingLLM**: Xiao et al. *Efficient Streaming Language Models with
  Attention Sinks.* arXiv:2309.17453, 2023.
- **Lost in the Middle**: Liu et al. *Lost in the Middle: How Language
  Models Use Long Contexts.* arXiv:2307.03172, 2023.
