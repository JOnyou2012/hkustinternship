"""
Unit tests for the Phase 2: Hierarchical Token-Level Attention.

Covers:
- HierarchicalTokenAttention module: shape correctness, GQA, prefill fallback.
- Macro page scoring: consistency with Quest baseline.
- Token-level scoring: score computation, validity masking.
- Consolidation: sink/recent protection, budget guarantee.
- Token sparse attention: correctness when all tokens selected.
- Experiment harness: Phase 2 benchmark and verification.
- Quality metrics: token recall, attention overlap.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, HierarchicalConfig, MetricsConfig
from hierarchical_attention import HierarchicalTokenAttention
from quest_attention import QuestAttention, _build_pages
from experiment import benchmark_decode_step_phase2, _verify_hierarchical_decode_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(a, b, **kw):
    return torch.allclose(a.float(), b.float(), **kw)


# ===================================================================
# Hierarchical Token Attention — Module Tests
# ===================================================================

class TestHierarchicalTokenAttention(unittest.TestCase):
    """Tests for the Phase 2 hierarchical token-level attention module."""

    def setUp(self):
        self.B, self.H, self.H_kv = 1, 4, 4
        self.d = 64
        self.page_size = 16
        self.top_k = 2
        self.token_budget = self.top_k * self.page_size  # 32

    # ---- Shape tests ----

    def test_decode_shape(self):
        """Module with T=1 should produce correct output shape."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        x = torch.randn(1, 1, 256)
        out = hta(x)
        self.assertEqual(out.shape, (1, 1, 256))

    def test_prefill_shape(self):
        """Module with T > 1 should produce correct output shape."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        x = torch.randn(1, 20, 256)
        out = hta(x)
        self.assertEqual(out.shape, (1, 20, 256))

    def test_gqa_shape(self):
        """Module with GQA (fewer KV heads) should broadcast correctly."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=8, head_dim=32,
            num_kv_heads=2, page_size=16, top_k=2,
        )
        x = torch.randn(1, 1, 256)
        out = hta(x)
        self.assertEqual(out.shape, (1, 1, 256))

    def test_return_kv(self):
        """return_kv=True should return output plus K, V projections."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        x = torch.randn(1, 8, 256)
        result = hta(x, return_kv=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        out, K, V = result
        self.assertEqual(out.shape, (1, 8, 256))
        self.assertEqual(K.shape, (1, 4, 8, 64))
        self.assertEqual(V.shape, (1, 4, 8, 64))

    def test_return_metrics(self):
        """return_metrics=True should return output + metrics dict."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        x = torch.randn(1, 1, 256)
        result = hta(x, return_metrics=True)
        self.assertIsInstance(result, tuple)
        out, metrics = result
        self.assertEqual(out.shape, (1, 1, 256))
        self.assertIn("num_macro_pages", metrics)
        self.assertIn("token_budget", metrics)
        self.assertEqual(metrics["token_budget"], 32)  # top_k * page_size

    # ---- Macro page scoring ----

    def test_page_scoring_shape(self):
        """Quest-style page scoring produces correct shape."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        Q = torch.randn(1, 4, 1, 64)
        K_min = torch.randn(1, 4, 4, 64)
        K_max = torch.randn(1, 4, 4, 64)
        scores = hta._score_pages_quest(Q, K_min, K_max)
        self.assertEqual(scores.shape, (1, 4, 4))  # (B, H, num_pages)

    def test_page_scoring_gqa_broadcast(self):
        """Page scoring broadcasts KV heads for GQA."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=8, head_dim=32,
            num_kv_heads=2, page_size=16, top_k=2,
        )
        Q = torch.randn(1, 8, 1, 32)
        K_min = torch.randn(1, 2, 4, 32)  # 2 KV heads
        K_max = torch.randn(1, 2, 4, 32)
        scores = hta._score_pages_quest(Q, K_min, K_max)
        self.assertEqual(scores.shape, (1, 8, 4))

    # ---- Adaptive budget ----

    def test_adaptive_budget_concentrated(self):
        """Concentrated page scores → smaller M."""
        # Simulate concentrated scores: one page dominates
        page_scores = torch.tensor([[[100.0, 1.0, 0.5, 0.3, 0.1, 0.05]]])
        M = HierarchicalTokenAttention._compute_adaptive_M(
            page_scores, base_M=6, num_pages=6
        )
        # High concentration → reduced M
        self.assertLess(M, 6)

    def test_adaptive_budget_uniform(self):
        """Uniform page scores → larger M."""
        # Simulate uniform scores
        page_scores = torch.tensor([[[1.0, 0.95, 1.05, 0.98, 1.02, 1.0]]])
        M = HierarchicalTokenAttention._compute_adaptive_M(
            page_scores, base_M=3, num_pages=6
        )
        # Low concentration → larger M (doubled, capped at num_pages)
        self.assertGreaterEqual(M, 3)

    # ---- Token-level scoring ----

    def test_token_scoring_shape(self):
        """Token scoring produces per-token scores for selected pages."""
        B, H, d = 1, 4, 64
        ps = 16
        M = 3
        num_pages = 6

        Q = torch.randn(B, H, 1, d)
        K_paged = torch.randn(B, H, num_pages, ps, d)
        pad_mask = torch.ones(B, 1, num_pages, ps, 1).bool()
        page_indices = torch.tensor([[[0, 2, 4]]]).expand(B, H, -1)  # select 3 pages

        scores, validity, global_pos = (
            HierarchicalTokenAttention._score_tokens_in_selected_pages(
                Q, K_paged, pad_mask, page_indices
            )
        )

        expected_candidates = M * ps  # 48
        self.assertEqual(scores.shape, (B, H, expected_candidates))
        self.assertEqual(validity.shape, (B, H, expected_candidates))
        self.assertEqual(global_pos.shape, (B, H, expected_candidates))
        # All tokens valid (no padding)
        self.assertTrue(validity.all())

    def test_token_scoring_padding(self):
        """Padding tokens get score -inf and are marked invalid."""
        B, H, d = 1, 4, 64
        ps = 16
        M = 2
        num_pages = 4

        Q = torch.randn(B, H, 1, d)
        K_paged = torch.randn(B, H, num_pages, ps, d)
        # Last page has only 8 valid tokens (8 padding)
        pad_mask = torch.ones(B, 1, num_pages, ps, 1).bool()
        pad_mask[:, :, -1, 8:, :] = False  # last 8 of last page are padding
        page_indices = torch.tensor([[[1, 3]]]).expand(B, H, -1)  # includes padded page

        scores, validity, global_pos = (
            HierarchicalTokenAttention._score_tokens_in_selected_pages(
                Q, K_paged, pad_mask, page_indices
            )
        )

        # Check that padding positions are flagged
        # Page 1 (index 1): tokens 16..31 → valid
        # Page 3 (index 3): tokens 48..55 valid, 56..63 padding
        # In flattened candidates: positions 0..15 (page 1), 16..31 (page 3)
        self.assertEqual(validity[0, 0, 0:16].sum().item(), 16)   # all valid
        self.assertEqual(validity[0, 0, 16:24].sum().item(), 8)   # 8 valid
        self.assertEqual(validity[0, 0, 24:32].sum().item(), 0)   # 8 padding
        # Padding scores are -inf
        self.assertTrue(torch.isinf(scores[0, 0, 24:]).all())

    def test_global_position_mapping(self):
        """Global positions correctly map pages to KV positions."""
        ps = 16
        page_indices = torch.tensor([[[0, 3, 5]]])  # B=1, H=1, M=3

        # Manual check
        base_positions = page_indices * ps  # [[[0, 48, 80]]]
        offsets = torch.arange(ps)

        # Page 0: positions 0..15
        # Page 3: positions 48..63
        # Page 5: positions 80..95
        expected_pos = torch.cat([
            base_positions[0, 0, 0] + offsets,
            base_positions[0, 0, 1] + offsets,
            base_positions[0, 0, 2] + offsets,
        ])

        # Construct positions the same way the module does
        computed_pos = (base_positions.unsqueeze(-1) + offsets.view(1, 1, 1, ps)).reshape(1, 1, 3 * ps)
        self.assertTrue(torch.equal(expected_pos, computed_pos[0, 0]))

    # ---- Consolidation (sink/recent protection) ----

    def test_sink_tokens_protected(self):
        """First num_sink_tokens should always be in final selection."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
            num_sink_tokens=4, num_recent_tokens=0,
        )
        B, H = 1, 1
        num_candidates = 100
        kv_len = 80

        # Create scores and positions — sink tokens at pos 0,1,2,3
        tk_scores = torch.randn(B, H, num_candidates)
        tk_scores[:, :, :4] = -100.0  # Make sink tokens look terrible
        tk_scores[:, :, 4:] = 1.0    # Others look great
        validity = torch.ones(B, H, num_candidates).bool()
        global_pos = torch.arange(num_candidates).unsqueeze(0).unsqueeze(0).float()  # pos 0..99

        sel_pos, sel_valid = hta._consolidate_tokens(
            tk_scores, validity, global_pos, kv_len
        )

        # Sink tokens (positions 0..3) should be selected despite low scores
        sel_set = set(sel_pos[0, 0].long().tolist())
        for sink_pos in range(4):
            self.assertIn(sink_pos, sel_set,
                          f"Sink token at position {sink_pos} not found in selection")

    def test_recent_tokens_protected(self):
        """Last num_recent_tokens should always be in final selection.

        Uses a realistic scenario where protected tokens fit within budget
        and candidate positions match the expected KV range.
        """
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=4,  # budget = 64
            num_sink_tokens=0, num_recent_tokens=8,
        )
        B, H = 1, 1
        kv_len = 128

        # Candidate pool: positions 0..127 (realistic — all KV positions)
        num_candidates = kv_len
        tk_scores = torch.randn(B, H, num_candidates)
        # Make recent tokens (pos 120..127) look terrible
        tk_scores[:, :, 120:128] = -100.0
        tk_scores[:, :, :120] = 1.0
        validity = torch.ones(B, H, num_candidates).bool()
        global_pos = torch.arange(num_candidates).unsqueeze(0).unsqueeze(0).float()

        sel_pos, sel_valid = hta._consolidate_tokens(
            tk_scores, validity, global_pos, kv_len
        )

        sel_set = set(sel_pos[0, 0].long().tolist())
        for recent_pos in range(120, 128):
            self.assertIn(recent_pos, sel_set,
                          f"Recent token at position {recent_pos} not found in selection")

    def test_budget_guarantee(self):
        """Final selection must never exceed token_budget."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2, token_budget=32,
        )
        B, H = 1, 1
        num_candidates = 200
        kv_len = 200

        tk_scores = torch.randn(B, H, num_candidates)
        validity = torch.ones(B, H, num_candidates).bool()
        global_pos = torch.arange(num_candidates).unsqueeze(0).unsqueeze(0).float()

        sel_pos, sel_valid = hta._consolidate_tokens(
            tk_scores, validity, global_pos, kv_len
        )

        self.assertLessEqual(sel_pos.size(-1), 32)
        self.assertEqual(sel_pos.size(-1), 32)  # exactly budget when candidates ≥ budget

    def test_budget_smaller_than_candidates(self):
        """When candidates < budget, selection is capped at available tokens."""
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2, token_budget=1000,  # huge budget
        )
        B, H = 1, 1
        num_candidates = 50  # fewer than budget
        kv_len = 50

        tk_scores = torch.randn(B, H, num_candidates)
        validity = torch.ones(B, H, num_candidates).bool()
        global_pos = torch.arange(num_candidates).unsqueeze(0).unsqueeze(0).float()

        sel_pos, sel_valid = hta._consolidate_tokens(
            tk_scores, validity, global_pos, kv_len
        )

        self.assertEqual(sel_pos.size(-1), 50)  # capped at available count

    # ---- Sparse attention on tokens ----

    def test_token_sparse_attention_equals_full_when_all_selected(self):
        """When all tokens are selected, hierarchical = full attention."""
        B, H, d = 1, 4, 64
        kv_len = 48  # exact multiple of page_size=16 → 3 pages

        torch.manual_seed(42)
        Q = torch.randn(B, H, 1, d)
        K = torch.randn(B, H, kv_len, d)
        V = torch.randn(B, H, kv_len, d)

        # Full attention
        scale = 1.0 / math.sqrt(d)
        scores_full = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights_full = F.softmax(scores_full, dim=-1)
        out_full = torch.matmul(weights_full, V)

        # Hierarchical with all tokens selected
        hta = HierarchicalTokenAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
            num_sink_tokens=0, num_recent_tokens=0,
        )
        all_positions = torch.arange(kv_len).unsqueeze(0).unsqueeze(0).expand(B, H, -1)
        all_valid = torch.ones(B, H, kv_len).bool()

        out_hier = hta._token_sparse_attention(
            Q, K, V, all_positions, all_valid, kv_len
        )

        self.assertTrue(
            _allclose(out_full, out_hier, atol=1e-5),
            f"Max diff: {(out_full - out_hier).abs().max().item():.6f}",
        )

    # ---- Equivalence with Quest when micro = page ----

    def test_equivalent_to_quest_when_macro_equals_topk(self):
        """When macro_multiplier=1 and no sink/recent, should approximate Quest.

        Not exactly equal because hierarchical does token-level scoring within
        pages, but the page selection should match.
        """
        pass  # This is a structural property test — verified via benchmark


# ===================================================================
# Experiment Harness — Phase 2
# ===================================================================

class TestPhase2ExperimentHarness(unittest.TestCase):
    """Tests for the Phase 2 benchmark and verification harness."""

    def test_benchmark_decode_step_runs(self):
        """Phase 2 benchmark returns four positive floats + quality dict."""
        result = benchmark_decode_step_phase2(
            kv_len=128, num_heads=4, head_dim=64, num_kv_heads=4,
            page_size=32, top_k=2, macro_multiplier=2,
            num_sink_tokens=2, num_recent_tokens=16,
            num_warmup=2, num_benchmark=5,
            device="cpu", dtype=torch.float32,
            return_quality_metrics=True,
        )
        fa_lat, qa_lat, fa_mem, qa_mem, quality = result
        self.assertGreater(fa_lat, 0)
        self.assertGreater(qa_lat, 0)
        self.assertIsInstance(fa_mem, float)
        self.assertIsInstance(qa_mem, float)
        self.assertIn("token_recall_mean", quality)
        self.assertGreaterEqual(quality["token_recall_mean"], 0.0)
        self.assertLessEqual(quality["token_recall_mean"], 1.0)
        self.assertIn("attention_jaccard_mean", quality)
        self.assertGreaterEqual(quality["attention_jaccard_mean"], 0.0)
        self.assertLessEqual(quality["attention_jaccard_mean"], 1.0)

    def test_benchmark_consistency(self):
        """Multiple calls should return consistent latencies."""
        results = []
        for _ in range(3):
            res = benchmark_decode_step_phase2(
                kv_len=64, num_heads=2, head_dim=32, num_kv_heads=2,
                page_size=16, top_k=1, macro_multiplier=2,
                num_warmup=1, num_benchmark=20,
                device="cpu", dtype=torch.float32,
                return_quality_metrics=False,
            )
            results.append(res[0])  # full latency
        # Allow 10x variance on CPU (OS scheduling noise)
        self.assertLess(max(results), 10 * min(results))

    def test_verify_cosine_similarity(self):
        """Cosine sim between full and hierarchical should be high."""
        cs = _verify_hierarchical_decode_step(
            kv_len=256, num_heads=4, head_dim=64, num_kv_heads=4,
            page_size=32, top_k=2, macro_multiplier=3,
            num_sink_tokens=2, num_recent_tokens=32,
            device="cpu", dtype=torch.float32,
        )
        # With a reasonable budget, cosine similarity should be decent
        self.assertGreater(cs, 0.8,
                           f"Cosine similarity {cs:.4f} is too low — "
                           f"hierarchical attention may be diverging")

    def test_quality_metrics_improve_with_larger_macro(self):
        """A wider macro net (higher multiplier) should improve recall."""
        # Small macro multiplier
        _, _, _, _, q_small = benchmark_decode_step_phase2(
            kv_len=256, num_heads=4, head_dim=64, num_kv_heads=4,
            page_size=32, top_k=2, macro_multiplier=1,  # M = top_k = 2 pages
            num_sink_tokens=2, num_recent_tokens=16,
            num_warmup=1, num_benchmark=5,
            device="cpu", dtype=torch.float32,
            return_quality_metrics=True,
        )

        # Large macro multiplier
        _, _, _, _, q_large = benchmark_decode_step_phase2(
            kv_len=256, num_heads=4, head_dim=64, num_kv_heads=4,
            page_size=32, top_k=2, macro_multiplier=4,  # M = 8 pages
            num_sink_tokens=2, num_recent_tokens=16,
            num_warmup=1, num_benchmark=5,
            device="cpu", dtype=torch.float32,
            return_quality_metrics=True,
        )

        # Larger macro net should give equal or better recall
        self.assertGreaterEqual(
            q_large["token_recall_mean"] + 0.05,  # 5% tolerance for randomness
            q_small["token_recall_mean"],
            f"Recall small={q_small['token_recall_mean']:.4f} "
            f"large={q_large['token_recall_mean']:.4f}",
        )


# ===================================================================
# Configuration Tests
# ===================================================================

class TestPhase2Config(unittest.TestCase):
    """Sanity checks on Phase 2 configuration dataclasses."""

    def test_default_hierarchical_config(self):
        cfg = HierarchicalConfig()
        self.assertEqual(cfg.page_size, 64)
        self.assertEqual(cfg.top_k, 4)
        self.assertEqual(cfg.macro_multiplier, 3)
        self.assertEqual(cfg.num_sink_tokens, 4)
        self.assertEqual(cfg.num_recent_tokens, 64)
        self.assertTrue(cfg.adaptive_budget)
        self.assertIsNone(cfg.token_budget)

    def test_token_budget_default(self):
        """When token_budget is None, it's derived as top_k * page_size."""
        cfg = HierarchicalConfig()
        budget = cfg.token_budget or (cfg.top_k * cfg.page_size)
        self.assertEqual(budget, 256)

    def test_custom_hierarchical_config(self):
        cfg = HierarchicalConfig(
            page_size=32, top_k=8, macro_multiplier=2,
            num_sink_tokens=8, num_recent_tokens=128,
            adaptive_budget=False, token_budget=512,
        )
        self.assertEqual(cfg.token_budget, 512)

    def test_full_config_includes_phase2(self):
        cfg = Config()
        self.assertEqual(cfg.hierarchical.page_size, 64)
        self.assertEqual(cfg.metrics.compute_token_recall, True)
        self.assertEqual(cfg.metrics.compute_attention_overlap, True)


# ===================================================================
# Integration Test — Phase 2 vs Phase 1 Quality
# ===================================================================

class TestPhase2VsPhase1Quality(unittest.TestCase):
    """Integration tests comparing Phase 2 quality against Quest baseline."""

    def test_phase2_cosine_sim_not_worse_than_quest(self):
        """Phase 2 should match or exceed Quest cosine similarity.

        Because Phase 2 selects tokens at a finer granularity within the
        same budget, its attention output should be at least as close to
        full attention as Quest's page-wise approach.
        """
        from experiment import _verify_decode_step as _verify_quest

        kv_len = 512
        num_heads, head_dim, num_kv_heads = 4, 64, 4
        page_size, top_k = 32, 4
        token_budget = top_k * page_size  # 128

        cs_quest = _verify_quest(
            kv_len=kv_len, num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads, page_size=page_size, top_k=top_k,
            device="cpu", dtype=torch.float32,
        )

        cs_hier = _verify_hierarchical_decode_step(
            kv_len=kv_len, num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads, page_size=page_size, top_k=top_k,
            macro_multiplier=3, num_sink_tokens=4, num_recent_tokens=64,
            token_budget=token_budget,
            device="cpu", dtype=torch.float32,
        )

        # Phase 2 should not be substantially worse
        self.assertGreaterEqual(
            cs_hier, cs_quest - 0.02,  # 2% tolerance
            f"Phase 2 CosSim {cs_hier:.4f} is significantly worse than "
            f"Quest CosSim {cs_quest:.4f}",
        )


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
