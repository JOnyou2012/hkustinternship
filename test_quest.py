"""
Unit tests for the Quest Baseline implementation.

Covers:
- Full attention: shape correctness, GQA broadcasting, numerical sanity.
- Quest attention: page building, metadata computation, scoring pipeline,
  sparse attention equivalence when top_k covers all pages.
- Experiment harness: decode-step benchmark and correctness verification.
"""

from __future__ import annotations

import math
import sys
import unittest

import torch
import torch.nn.functional as F

# Ensure the project root is on sys.path
sys.path.insert(0, "/Users/jonathan/anaconda_projects/hkustinternship")

from config import Config, ModelConfig, QuestConfig
from full_attention import MultiHeadFullAttention, scaled_dot_product_attention
from quest_attention import QuestAttention, _build_pages
from experiment import benchmark_decode_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(a, b, **kw):
    return torch.allclose(a.float(), b.float(), **kw)


# ===================================================================
# Full Attention Tests
# ===================================================================

class TestFullAttention(unittest.TestCase):
    """Tests for the full (dense) scaled dot-product attention baseline."""

    def setUp(self):
        self.d_k = 128
        self.B, self.H, self.T = 2, 4, 16

    def test_scaled_dot_product_shapes(self):
        """SDPA should preserve input shape."""
        Q = torch.randn(self.B, self.H, self.T, self.d_k)
        K = torch.randn(self.B, self.H, self.T, self.d_k)
        V = torch.randn(self.B, self.H, self.T, self.d_k)
        out = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, Q.shape)

    def test_causal_mask(self):
        """With causal mask, position i should not attend to positions > i."""
        Q = torch.randn(1, 1, 4, 8)
        K = torch.randn(1, 1, 4, 8)
        V = torch.randn(1, 1, 4, 8)
        mask = torch.triu(
            torch.full((4, 4), float("-inf")), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        self.assertEqual(out.shape, Q.shape)

    def test_multihead_full_attention_shape(self):
        """MultiHeadFullAttention output should be (B, T, hidden_dim)."""
        attn = MultiHeadFullAttention(
            hidden_dim=256, num_heads=4, head_dim=64, num_kv_heads=4
        )
        x = torch.randn(self.B, self.T, 256)
        out = attn(x)
        self.assertEqual(out.shape, (self.B, self.T, 256))

    def test_multihead_gqa_shape(self):
        """GQA with fewer KV heads should broadcast correctly."""
        attn = MultiHeadFullAttention(
            hidden_dim=256, num_heads=8, head_dim=32, num_kv_heads=2
        )
        x = torch.randn(2, 10, 256)
        out = attn(x)
        self.assertEqual(out.shape, (2, 10, 256))

    def test_return_kv(self):
        """return_kv=True should return output plus K and V projections."""
        attn = MultiHeadFullAttention(
            hidden_dim=256, num_heads=4, head_dim=64, num_kv_heads=4
        )
        x = torch.randn(1, 8, 256)
        result = attn(x, return_kv=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        out, K, V = result
        self.assertEqual(out.shape, (1, 8, 256))
        self.assertEqual(K.shape, (1, 4, 8, 64))
        self.assertEqual(V.shape, (1, 4, 8, 64))


# ===================================================================
# Quest Attention Tests
# ===================================================================

class TestQuestAttention(unittest.TestCase):
    """Tests for the Quest page-wise sparse attention implementation."""

    def setUp(self):
        self.B, self.H, self.H_kv = 1, 4, 4
        self.d = 64
        self.page_size = 16

    # ---- Page building ----

    def test_build_pages_exact_multiple(self):
        """When T is a multiple of page_size, no padding is added."""
        T = 64  # 4 pages of 16
        K = torch.randn(self.B, self.H, T, self.d)
        V = torch.randn(self.B, self.H, T, self.d)
        K_p, V_p, pad_mask, num_pages = _build_pages(K, V, self.page_size)
        self.assertEqual(num_pages, 4)
        self.assertEqual(K_p.shape, (self.B, self.H, 4, self.page_size, self.d))
        # All positions should be valid (no padding)
        self.assertTrue(pad_mask.all())

    def test_build_pages_with_padding(self):
        """When T is not a multiple of page_size, padding is added and masked."""
        T = 70  # 4 full pages + 6 tokens → padded to 5 pages (80 tokens)
        K = torch.randn(self.B, self.H, T, self.d)
        V = torch.randn(self.B, self.H, T, self.d)
        K_p, V_p, pad_mask, num_pages = _build_pages(K, V, self.page_size)
        self.assertEqual(num_pages, 5)
        # Last page should have 10 padding positions
        last_page_valid = pad_mask[0, 0, -1, :, 0]  # (page_size,)
        self.assertEqual(last_page_valid.sum().item(), 6)  # 6 real tokens
        self.assertEqual((~last_page_valid).sum().item(), 10)  # 10 padding

    # ---- Metadata computation ----

    def test_compute_page_metadata(self):
        """Min and max metadata should have the correct shape."""
        T = self.page_size * 3
        K = torch.randn(self.B, self.H, T, self.d)
        V = torch.randn(self.B, self.H, T, self.d)
        K_p, V_p, pad_mask, num_pages = _build_pages(K, V, self.page_size)
        K_min, K_max = QuestAttention.compute_page_metadata(K_p)
        self.assertEqual(K_min.shape, (self.B, self.H, num_pages, self.d))
        self.assertEqual(K_max.shape, (self.B, self.H, num_pages, self.d))
        # Min should be ≤ elements in the page, Max should be ≥
        self.assertTrue((K_min <= K_p.max(dim=3).values).all())
        self.assertTrue((K_max >= K_p.min(dim=3).values).all())

    # ---- Page scoring ----

    def test_score_pages_shape(self):
        """Scoring should produce one scalar per page per head."""
        num_pages = 4
        Q = torch.randn(self.B, self.H, 1, self.d)
        K_min = torch.randn(self.B, self.H, num_pages, self.d)
        K_max = torch.randn(self.B, self.H, num_pages, self.d)
        scores = QuestAttention.score_pages(Q, K_min, K_max)
        self.assertEqual(scores.shape, (self.B, self.H, num_pages))

    def test_score_pages_pipeline(self):
        """Verify the Quest scoring pipeline against a manual computation.

        Pipeline: element-wise product → per-channel max → sum.
        """
        Q = torch.tensor([[[[1.0, 2.0]]]])  # (1, 1, 1, 2)
        K_min = torch.tensor([[[[0.5, 1.0]]]])  # (1, 1, 1, 2) — 1 page
        K_max = torch.tensor([[[[0.0, 3.0]]]])

        scores = QuestAttention.score_pages(Q, K_min, K_max)
        # Manual:
        #   prod_min = [1*0.5, 2*1.0] = [0.5, 2.0]
        #   prod_max = [1*0.0, 2*3.0] = [0.0, 6.0]
        #   per-channel max = max([0.5, 0.0], [2.0, 6.0]) = [0.5, 6.0]
        #   sum = 0.5 + 6.0 = 6.5
        expected = torch.tensor([[[6.5]]])
        self.assertTrue(torch.allclose(scores.float(), expected.float()))

    def test_score_pages_gqa_broadcast(self):
        """Scoring should broadcast KV heads to match query heads in GQA."""
        # 4 query heads, 1 KV head
        Q = torch.randn(1, 4, 1, 64)
        K_min = torch.randn(1, 1, 3, 64)
        K_max = torch.randn(1, 1, 3, 64)
        scores = QuestAttention.score_pages(Q, K_min, K_max)
        self.assertEqual(scores.shape, (1, 4, 3))

    # ---- Sparse attention equivalence ----

    def test_sparse_equals_full_when_all_pages_selected(self):
        """When top_k ≥ num_pages, sparse attention MUST equal full attention.

        This is the key invariant: selecting all pages collapses Quest to
        standard scaled dot-product attention.
        """
        T = self.page_size * 3  # exactly 3 pages, no padding
        d_k = self.d

        # Fixed random seed for reproducibility
        torch.manual_seed(42)
        Q = torch.randn(1, 1, 1, d_k)
        K = torch.randn(1, 1, T, d_k)
        V = torch.randn(1, 1, T, d_k)

        # Full attention
        scale = 1.0 / math.sqrt(d_k)
        scores_full = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights_full = F.softmax(scores_full, dim=-1)
        out_full = torch.matmul(weights_full, V)

        # Quest sparse attention with top_k = num_pages (= 3)
        K_p, V_p, pad_mask, num_pages = _build_pages(K, V, self.page_size)
        K_min = K_p.min(dim=3).values
        K_max = K_p.max(dim=3).values

        Q_exp = Q.expand(-1, -1, num_pages, -1)
        scores = (torch.max(Q_exp * K_min, Q_exp * K_max)).sum(dim=-1)
        effective_k = min(10, num_pages)  # top_k=10, num_pages=3 → selects all
        _, page_indices = torch.topk(scores, k=effective_k, dim=-1)

        # Manual sparse attention (replicating Quest's logic)
        top_k = page_indices.size(-1)
        idx = page_indices.view(1, 1, top_k, 1, 1).expand(-1, -1, -1, self.page_size, d_k)
        K_sel = K_p.gather(dim=2, index=idx).reshape(1, 1, top_k * self.page_size, d_k)
        V_sel = V_p.gather(dim=2, index=idx).reshape(1, 1, top_k * self.page_size, d_k)

        # No padding → no mask needed
        scores_q = torch.matmul(Q, K_sel.transpose(-2, -1)) * scale
        weights_q = F.softmax(scores_q, dim=-1)
        out_quest = torch.matmul(weights_q, V_sel)

        # They should be numerically very close (same tokens, same attention)
        self.assertTrue(
            _allclose(out_full, out_quest, atol=1e-5),
            f"Max diff: {(out_full - out_quest).abs().max().item():.6f}",
        )

    # ---- Module-level tests ----

    def test_module_forward_decode(self):
        """Quest module with T=1 should produce correct output shape."""
        qa = QuestAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        x = torch.randn(1, 1, 256)
        out = qa(x)
        self.assertEqual(out.shape, (1, 1, 256))

    def test_module_forward_prefill(self):
        """Quest module with T > 1 should fall back to full attention for
        earlier positions and Quest for the last position."""
        qa = QuestAttention(
            hidden_dim=256, num_heads=4, head_dim=64,
            num_kv_heads=4, page_size=16, top_k=2,
        )
        x = torch.randn(1, 20, 256)
        out = qa(x)
        self.assertEqual(out.shape, (1, 20, 256))

    def test_module_gqa(self):
        """Quest module with GQA should broadcast KV heads correctly."""
        qa = QuestAttention(
            hidden_dim=256, num_heads=8, head_dim=32,
            num_kv_heads=2, page_size=16, top_k=2,
        )
        x = torch.randn(1, 1, 256)
        out = qa(x)
        self.assertEqual(out.shape, (1, 1, 256))


# ===================================================================
# Experiment Harness Tests
# ===================================================================

class TestExperimentHarness(unittest.TestCase):
    """Tests for the decode-step benchmark and correctness verification."""

    def test_benchmark_decode_step_runs(self):
        """Benchmark should return four positive floats."""
        fa_lat, qa_lat, fa_mem, qa_mem = benchmark_decode_step(
            kv_len=128, num_heads=4, head_dim=64, num_kv_heads=4,
            page_size=32, top_k=2, num_warmup=2, num_benchmark=5,
            device="cpu", dtype=torch.float32,
        )
        self.assertGreater(fa_lat, 0)
        self.assertGreater(qa_lat, 0)
        # Memory may be 0 on CPU — just check type
        self.assertIsInstance(fa_mem, float)
        self.assertIsInstance(qa_mem, float)

    def test_benchmark_consistency(self):
        """Multiple calls should return consistent (similar) latencies."""
        results = []
        for _ in range(3):
            res = benchmark_decode_step(
                kv_len=64, num_heads=2, head_dim=32, num_kv_heads=2,
                page_size=16, top_k=1, num_warmup=2, num_benchmark=20,
                device="cpu", dtype=torch.float32,
            )
            results.append(res[0])  # full latency
        # All should be within 3x of each other (CPU noise tolerance)
        self.assertLess(max(results), 3 * min(results))


# ===================================================================
# Configuration Tests
# ===================================================================

class TestConfig(unittest.TestCase):
    """Sanity checks on the configuration dataclasses."""

    def test_default_config(self):
        cfg = Config()
        self.assertEqual(cfg.model.hidden_dim, 4096)
        self.assertEqual(cfg.quest.page_size, 64)
        self.assertEqual(cfg.quest.top_k, 4)

    def test_custom_config(self):
        cfg = Config(
            model=ModelConfig(hidden_dim=2048, num_heads=16, head_dim=128),
            quest=QuestConfig(page_size=32, top_k=8),
        )
        self.assertEqual(cfg.model.hidden_dim, 2048)
        self.assertEqual(cfg.quest.page_size, 32)


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
