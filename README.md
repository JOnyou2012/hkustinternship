# Quest Baseline — Phase 1: Page-wise Sparse Attention

Reference implementation reproducing the **Quest** page-wise sparse attention
algorithm for long-context LLM serving.  iSING Lab, HKUST.

## Overview

Standard full attention scales quadratically $O(T^2)$ with sequence length,
creating two critical bottlenecks at decode time:

1. **KV-cache capacity** — storing the full key/value history for long contexts
   exhausts GPU memory.
2. **Full-attention latency** — re-reading the entire KV cache at every decode
   step becomes prohibitively slow.

**Quest** addresses both by dividing the KV cache into fixed-size *pages*,
precomputing per-page key metadata (element-wise min/max), and selecting only
the **Top-K** most relevant pages at each decode step.  Sparse attention is
then computed exclusively over those selected pages.

## Mathematical Foundation

$$Attention(Q,K,V)=\text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

Quest approximates this by restricting the softmax to the Top-K pages:

$$Attention_{Quest}(Q,K,V) \approx \text{softmax}\left(\frac{Q K_{topk}^T}{\sqrt{d}}\right) V_{topk}$$

## Quest Algorithm

```
Input:  Q (1 query token), KV cache of T tokens
Output: Attention output

1. PAGE CONSTRUCTION
   - Partition K, V into ⌈T / page_size⌉ pages
   - Pad final page with zeros if needed; create validity mask

2. PAGE METADATA (precomputed once per page, cached)
   - K_min[p] = element-wise min of keys in page p
   - K_max[p] = element-wise max of keys in page p

3. STAGE 1 — ESTIMATE CRITICAL PAGES
   For each page p:
     a. prod_min = Q ⊙ K_min[p]      (element-wise product)
     b. prod_max = Q ⊙ K_max[p]      (element-wise product)
     c. combined = max(prod_min, prod_max)   (per-channel max)
     d. score[p] = sum(combined)             (scalar score)

4. STAGE 2 — SPARSE ATTENTION
   - Select Top-K pages by score
   - Compute scaled dot-product attention ONLY over selected pages
   - Apply padding mask to exclude padded positions from softmax
```

## Project Structure

```
├── config.py            # Configuration dataclasses (model, Quest, experiment)
├── utils.py             # GPU timer, memory tracking, cosine similarity, formatting
├── full_attention.py    # Standard multi-head scaled dot-product attention
├── quest_attention.py   # Quest page-wise sparse attention (module + helpers)
├── experiment.py        # Decode-step benchmark harness and correctness verification
├── run_benchmark.py     # CLI entry point
├── test_quest.py        # Unit tests (19 tests covering all components)
└── README.md
```

## Quick Start

### Install dependencies

```bash
pip install torch
```

### Run the benchmark (CPU)

```bash
python run_benchmark.py --device cpu --kv-lens 512 1024 2048 4096
```

### Run the benchmark (CUDA GPU — requires CUDA-capable PyTorch)

```bash
python run_benchmark.py --device cuda --kv-lens 512 1024 2048 4096 8192
```

### Run with GQA (Grouped-Query Attention, e.g. Llama-2 70B style)

```bash
python run_benchmark.py --num-heads 64 --num-kv-heads 8 --head-dim 128
```

### Custom Quest hyperparameters

```bash
python run_benchmark.py --page-size 32 --top-k 8
```

### Run unit tests

```bash
python -m unittest test_quest -v
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num-heads` | 32 | Number of query heads |
| `--head-dim` | 128 | Dimension per head |
| `--num-kv-heads` | 32 | KV heads (set < num-heads for GQA) |
| `--page-size` | 64 | Tokens per page |
| `--top-k` | 4 | Pages selected for sparse attention |
| `--kv-lens` | 512 1024 2048 4096 8192 | KV-cache sizes to sweep |
| `--num-warmup` | 10 | GPU warmup iterations |
| `--num-benchmark` | 50 | Timed iterations per data point |
| `--device` | cuda | `cuda` or `cpu` |
| `--dtype` | float16 | `float16`, `float32`, or `bfloat16` |
| `--no-verify` | (off) | Skip cosine-similarity verification |
| `--correctness-threshold` | 0.99 | Threshold for similarity check |

## Expected Results

On GPU with large KV caches (8K+ tokens):

- **Latency**: Quest achieves sub-linear scaling vs. the linear (in practice
  quadratic due to memory bandwidth) scaling of full attention.
- **Memory**: Quest's peak memory during the attention computation is
  proportional to `top_k × page_size` rather than the full KV cache size.
- **Quality**: When `top_k` covers most of the available pages, cosine
  similarity with full attention is >0.99.  The approximation trades
  controlled quality loss for significant speed/memory gains.

On CPU with small caches, Quest's page-scoring overhead may dominate —
the algorithm is designed for GPU execution with long contexts.

## Key Implementation Details

- **Padding handling**: Pages that are not fully populated (last page when
  `T % page_size ≠ 0`) are padded with zeros and excluded from softmax via
  an additive `-inf` mask. This guarantees correctness even with odd-sized
  caches.
- **GQA support**: Both full attention and Quest support Grouped-Query
  Attention by broadcasting KV heads to match query heads.
- **GPU optimization**: The scoring pipeline (`element-wise product → max
  → sum`) is implemented as a single fused sequence of PyTorch operations
  that maps cleanly to CUDA kernels.
- **Decode-mode benchmarking**: The harness simulates the decode phase
  (1 query token vs large KV cache) — Quest's target operating scenario.

## References

- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
  (iSING Lab, HKUST)
- Llama-2: Touvron et al., 2023
