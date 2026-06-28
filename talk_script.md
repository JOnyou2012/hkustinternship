# 5-Minute Talk: Hierarchical Token-Level Sparse Attention

**Speaker:** Jonathan  
**Lab:** iSING Lab, HKUST  
**Duration:** 5 minutes (±15 seconds at normal pace)  
**Visual aid:** `reports/phase2_quality_vs_budget.pdf` — two-panel figure

---

## [0:00–1:00] INTRODUCTION — The Long-Context Bottleneck

> *[Show slide with attention complexity equation]*

Good morning. Every time a large language model generates a single word, it
re-reads its entire conversation history. This is full attention — and it
scales quadratically. For a 32,000-token context, that's about a billion
operations per word. Two bottlenecks emerge: the KV cache consumes GPU memory
proportional to sequence length, and attention latency grows without bound.

Last year, researchers at this lab proposed **Quest** — a sparse attention
algorithm that partitions the KV cache into fixed-size pages, precomputes a
summary for each page, and at each decode step selects only the top-K most
relevant pages. Instead of reading 32,000 tokens, Quest reads just 256 —
four pages of 64 tokens each. In Phase 1 of this project, we reproduced Quest
from scratch, validated it with 19 unit tests, and demonstrated a 1.3×
speedup at 4,000 tokens on CPU.

But Quest has a fundamental quality problem, and that's what Phase 2 solves.

---

## [1:00–2:30] METHODOLOGY — From Pages to Tokens

> *[Transition to architecture diagram or pipeline illustration]*

Here's the problem with page-wise selection. When Quest selects a page
because one word in it is relevant, it drags along 63 other words — most of
which are useless filler. And a critical word in an unselected page? It's
lost forever. The coarse granularity is the bottleneck.

Our solution is a **three-stage hierarchical filtering pipeline** that
operates under the exact same budget. Same 256 tokens. But instead of picking
pages, we pick individual tokens.

Stage one — **Macro-Selection**. We use Quest's proven page-scoring
methodology, but we cast a wider net. Instead of selecting 4 pages, we select
12 — three times as many. We also force-include the page containing the first
few tokens and the pages containing the most recent tokens. These are
structural guarantees backed by the StreamingLLM and Lost-in-the-Middle
findings.

Stage two — **Micro-Selection**. Inside those 12 pages, we score every single
token individually using exact query-key dot products. This is the critical
innovation. We're no longer limited by page boundaries — every token competes
on its own merit. From 768 candidate tokens, we get exact per-token relevance
scores.

Stage three — **Consolidation**. We protect four attention-sink tokens and
sixty-four recent tokens unconditionally. Those 68 tokens are 27% of our
budget, spent on structural stability. The remaining 188 slots go to the
highest-scoring tokens from stage two. The final selection is exactly 256
tokens — the same as Quest's budget.

We also implemented an adaptive budget mechanism that dynamically sizes the
macro net. When page scores are concentrated — a few pages dominate — we
shrink the net. When scores are diffuse, we widen it. This adapts to the
conversation structure automatically.

---

## [2:30–4:00] RESULTS — The Core Figure

> *[Point to the left panel — bar chart]*

Let me walk you through the core result. The left panel shows cosine
similarity against full attention — our quality metric — across five KV-cache
sizes from 512 to 8,192 tokens.

The orange bars are Quest, Phase 1. The blue bars are our hierarchical
method, Phase 2. Same budget everywhere — 256 tokens.

At 512 tokens, Quest achieves 0.73 cosine similarity. Phase 2 achieves 0.98.
That's a 34% improvement, and we're already near perfect.

At 1,024 tokens, Quest drops to 0.50. Phase 2 holds at 0.94 — an 87%
improvement. We're still above 0.9 while Quest has already lost half its
fidelity.

At 2,048 tokens, the gap widens to 110%. At 4,096, it's 123%. At 8,192
tokens — the longest context we tested — Quest is at 0.18 while Phase 2 holds
at 0.41. That's a 130% improvement.

Notice the trend. The improvement percentage grows with context length. This
is exactly what you want for long-context deployment — the method gets
*relatively better* as the context grows longer.

> *[Point to the right panel — speedup chart]*

The right panel shows speedup versus full attention. Phase 1 is faster on
CPU — it does less work per token. Phase 2's three-stage pipeline adds
overhead. But this is on CPU. On GPU at 32,000-plus token contexts, the
attention matmul dominates everything else, and both methods attend to
exactly 256 tokens. The speed gap narrows to near zero, while the quality gap
remains.

We also track token recall — what fraction of the true top-256 attention
tokens did we capture? At 512 tokens, it's 87%. At 2,048 tokens — 75%. That
means we're catching three out of every four critical tokens while reading
only 12.5% of the total.

---

## [4:00–5:00] CONCLUSION — Impact and Next Steps

> *[Return to title slide or summary]*

Let me summarize what this means for real-world LLM deployment.

Phase 2 proves that **token-level selection dominates page-level selection**
under the same budget constraint. You get substantially better attention
quality — 34 to 130 percent better — for the same memory cost. No extra GPU
memory. No larger KV cache. Just smarter selection.

The architectural contributions are threefold. First, the hierarchical
pipeline itself — macro, micro, consolidation — is a reusable pattern that
can be plugged into any sparse attention system. Second, the sink and recent
token protection provides structural stability that prevents catastrophic
quality degradation. Third, the adaptive budget makes the method robust
across different conversation structures without manual tuning.

What's next? The natural extension is to integrate this into a full Llama-2
or Llama-3 inference pipeline and evaluate on the LongBench benchmark suite.
Kernel fusion of the three-stage pipeline for GPU efficiency. Learned scoring
functions that replace the hand-designed Quest heuristic. And dynamic per-head
budget allocation — some attention heads clearly need more tokens than others.

The code is open-source, fully tested with 46 unit tests, and ready for
integration. Thank you.

---

## Timing Notes

| Section | Words | Est. Time |
|---------|-------|-----------|
| Introduction | 160 | ~1:00 |
| Methodology | 250 | ~1:35 |
| Results | 240 | ~1:30 |
| Conclusion | 150 | ~0:55 |
| **Total** | **~800** | **~5:00** |

*Pacing: ~150 words per minute. Natural pauses between sections account for
slide transitions. Practice the figure walkthrough with a pointer — the
numbers need to land clearly.*
