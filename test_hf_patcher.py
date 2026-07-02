"""
Unit tests for HuggingFace model patcher.

Covers:
- Model type detection
- Layer access for supported model types
- Weight transfer: GPT-2 Conv1D → nn.Linear, standard nn.Linear → nn.Linear
- _SparseAttentionWrapper return-value convention
- patch_model integration (GPT-2 smoke test)
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hf_model_patcher import (
    _detect_model_type,
    _get_transformer_layers,
    _get_attention_module,
    _set_attention_module,
    _SparseAttentionWrapper,
    _MODEL_PATHS,
    transfer_gpt2_weights,
    transfer_standard_weights,
    transfer_weights,
    patch_model,
    verify_weight_transfer,
)
from quest_attention import QuestAttention
from hierarchical_attention import HierarchicalTokenAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyConfig:
    model_type = "gpt2"
    hidden_size = 256
    num_attention_heads = 4
    # GPT-2 has no num_key_value_heads attr → defaults to num_heads


class _DummyGPT2Attention(nn.Module):
    """Minimal mock of GPT2Attention with Conv1D-like layers."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        # Conv1D stores weight as (in_features, out_features) = (hidden, 3*hidden)
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)  # approximate for test
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        # Transpose weights to mimic Conv1D convention
        self.c_attn.weight.data = self.c_attn.weight.data.t()
        self.c_proj.weight.data = self.c_proj.weight.data.t()

    @property
    def num_heads(self):
        return 4


class _DummyLlamaAttention(nn.Module):
    """Minimal mock of LlamaAttention with separate q/k/v/o Linear layers."""

    def __init__(self, hidden_dim=256, num_heads=4, num_kv_heads=2):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)


class _DummyModel:
    """Minimal model-like object for testing layer access."""

    def __init__(self, model_type="gpt2"):
        self.config = _DummyConfig()
        self.config.model_type = model_type


class _DummyGPT2Model(_DummyModel):
    def __init__(self):
        super().__init__("gpt2")

        class _GPT2Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _DummyGPT2Attention()

        self.transformer = type("transformer", (), {})()
        self.transformer.h = nn.ModuleList([_GPT2Layer() for _ in range(2)])


# ===================================================================
# Model Type Detection & Layer Access
# ===================================================================

class TestModelDetection(unittest.TestCase):

    def test_detect_gpt2(self):
        model = _DummyModel("gpt2")
        self.assertEqual(_detect_model_type(model), "gpt2")

    def test_detect_llama(self):
        model = _DummyModel("llama")
        self.assertEqual(_detect_model_type(model), "llama")

    def test_all_model_paths_have_valid_keys(self):
        """Every model type in _MODEL_PATHS should be a valid dotted path."""
        for model_type, path in _MODEL_PATHS.items():
            self.assertIsInstance(path, str)
            self.assertTrue(len(path.split(".")) >= 1,
                           f"Invalid path for {model_type}: {path}")


class TestLayerAccess(unittest.TestCase):

    def test_get_layers_gpt2(self):
        model = _DummyGPT2Model()
        layers = _get_transformer_layers(model)
        self.assertEqual(len(layers), 2)

    def test_get_attention_module(self):
        model = _DummyGPT2Model()
        layer = _get_transformer_layers(model)[0]
        attn = _get_attention_module(layer)
        self.assertIsInstance(attn, _DummyGPT2Attention)

    def test_unsupported_model_raises(self):
        model = _DummyModel("unknown_arch")
        with self.assertRaises(ValueError):
            _get_transformer_layers(model)


# ===================================================================
# Weight Transfer
# ===================================================================

class TestWeightTransfer(unittest.TestCase):

    def test_transfer_gpt2_weights(self):
        hf_attn = _DummyGPT2Attention(hidden_dim=256)
        target = QuestAttention(
            hidden_dim=256, num_heads=4, head_dim=64, num_kv_heads=4,
            bias=True,
        )
        transfer_gpt2_weights(hf_attn, target)

        # Weights should be non-zero after transfer
        self.assertFalse(torch.allclose(target.q_proj.weight,
                                        torch.zeros_like(target.q_proj.weight)))
        self.assertFalse(torch.allclose(target.o_proj.weight,
                                        torch.zeros_like(target.o_proj.weight)))

    def test_transfer_standard_weights(self):
        hf_attn = _DummyLlamaAttention(hidden_dim=256, num_heads=4, num_kv_heads=2)
        target = QuestAttention(
            hidden_dim=256, num_heads=4, head_dim=64, num_kv_heads=2,
        )
        transfer_standard_weights(hf_attn, target)

        self.assertTrue(torch.equal(hf_attn.q_proj.weight, target.q_proj.weight))
        self.assertTrue(torch.equal(hf_attn.k_proj.weight, target.k_proj.weight))
        self.assertTrue(torch.equal(hf_attn.o_proj.weight, target.o_proj.weight))

    def test_transfer_weights_dispatch_gpt2(self):
        hf_attn = _DummyGPT2Attention(hidden_dim=256)
        target = QuestAttention(
            hidden_dim=256, num_heads=4, head_dim=64, num_kv_heads=4,
            bias=True,
        )
        transfer_weights(hf_attn, target, "gpt2")
        self.assertFalse(torch.allclose(target.q_proj.weight,
                                        torch.zeros_like(target.q_proj.weight)))

    def test_transfer_weights_dispatch_llama(self):
        hf_attn = _DummyLlamaAttention(hidden_dim=256, num_heads=4, num_kv_heads=2)
        target = QuestAttention(
            hidden_dim=256, num_heads=4, head_dim=64, num_kv_heads=2,
        )
        transfer_weights(hf_attn, target, "llama")
        self.assertTrue(torch.equal(hf_attn.q_proj.weight, target.q_proj.weight))


# ===================================================================
# SparseAttentionWrapper
# ===================================================================

class TestSparseAttentionWrapper(unittest.TestCase):

    def test_single_tensor_output(self):
        """Wrapper should return (tensor, None) for single-tensor output."""
        mod = nn.Linear(4, 4)
        wrapper = _SparseAttentionWrapper(mod)
        x = torch.randn(2, 4)
        out = wrapper(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsNone(out[1])

    def test_tuple_output_with_metrics(self):
        """Wrapper should extract main output and drop metrics dict."""
        class _MetricsModule(nn.Module):
            def forward(self, x):
                return x, {"num_macro_pages": 3, "token_budget": 256}

        mod = _MetricsModule()
        wrapper = _SparseAttentionWrapper(mod, verbose_metrics=True)
        x = torch.randn(2, 4)
        out = wrapper(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0], x))
        self.assertIsNone(out[1])

    def test_tuple_output_with_kv(self):
        """Wrapper should handle (output, K, V) tuples."""
        class _KVModule(nn.Module):
            def forward(self, x):
                return x, torch.ones(1), torch.ones(1)

        mod = _KVModule()
        wrapper = _SparseAttentionWrapper(mod)
        x = torch.randn(2, 4)
        out = wrapper(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0], x))
        self.assertIsNone(out[1])


# ===================================================================
# Integration — patch_model with real modules
# ===================================================================

class TestPatchModelIntegration(unittest.TestCase):

    def test_patch_quest_creates_wrappers(self):
        model = _DummyGPT2Model()
        patch_model(model, method="quest", page_size=16, top_k=2)

        for layer in model.transformer.h:
            wrapper = _get_attention_module(layer)
            self.assertIsInstance(wrapper, _SparseAttentionWrapper)
            self.assertIsInstance(wrapper.sparse_module, QuestAttention)

    def test_patch_hierarchical_creates_wrappers(self):
        model = _DummyGPT2Model()
        patch_model(model, method="hierarchical", page_size=16, top_k=2,
                    macro_multiplier=2)

        for layer in model.transformer.h:
            wrapper = _get_attention_module(layer)
            self.assertIsInstance(wrapper, _SparseAttentionWrapper)
            self.assertIsInstance(wrapper.sparse_module, HierarchicalTokenAttention)

    def test_patch_specific_layers(self):
        model = _DummyGPT2Model()
        patch_model(model, method="quest", page_size=16, top_k=2, layers=[0])

        # Layer 0 patched
        wrapper0 = _get_attention_module(model.transformer.h[0])
        self.assertIsInstance(wrapper0, _SparseAttentionWrapper)

        # Layer 1 untouched
        attn1 = _get_attention_module(model.transformer.h[1])
        self.assertIsInstance(attn1, _DummyGPT2Attention)

    def test_patch_forward_runs(self):
        """Patched model should complete a forward pass without error."""
        model = _DummyGPT2Model()
        patch_model(model, method="quest", page_size=16, top_k=2)

        # Simulate a simple forward through the first layer
        x = torch.randn(1, 8, 256)  # (B, T, hidden_dim)
        wrapper = _get_attention_module(model.transformer.h[0])
        out = wrapper(x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0].shape, (1, 8, 256))


# ===================================================================
# Verify weight transfer
# ===================================================================

class TestVerifyWeightTransfer(unittest.TestCase):

    def test_verify_returns_valid_flags(self):
        model = _DummyGPT2Model()
        patch_model(model, method="quest", page_size=16, top_k=2)
        result = verify_weight_transfer(model, layer_idx=0)
        self.assertIn("q_proj_valid", result)
        self.assertIn("k_proj_valid", result)
        self.assertIn("v_proj_valid", result)
        self.assertIn("o_proj_valid", result)
        # All should be valid (non-zero weights)
        self.assertTrue(all(result.values()))


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
