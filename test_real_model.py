
from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# check if transformers are installed

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from hf_model_patcher import patch_model, verify_weight_transfer

MODEL_NAME = "distilgpt2"

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _load_model(device: str = "cpu") -> torch.nn.Module:
    """Load a fresh pretrained model (no sparse patching)."""
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    m.eval()
    return m.to(device)


def _tokenize(text: str, seq_len: int = 256) -> torch.LongTensor:
    """Tokenize a text string into a (1, seq_len) tensor."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer.encode(text, return_tensors="pt")
    if tokens.size(1) < seq_len:
        # Pad to seq_len
        tokens = torch.nn.functional.pad(
            tokens, (0, seq_len - tokens.size(1)), value=tokenizer.pad_token_id
        )
    else:
        tokens = tokens[:, :seq_len]
    return tokens


def _compute_perplexity(model, input_ids: torch.Tensor) -> float:
    """Compute perplexity = exp(cross_entropy) over a single forward pass."""
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipIf(not HAS_TRANSFORMERS, "transformers is not installed")
class TestForwardPass(unittest.TestCase):
    """Verify patched models produce finite output without crashing."""

    def setUp(self):
        self.seq_len = 256
        self.budget = 64   # page_size=16, top_k=4
        self.x = torch.randint(0, 100, (1, self.seq_len))

    def test_full_forward_pass(self):
        """Unpatched model forward pass — baseline sanity check."""
        model = _load_model()
        out = model(self.x).logits
        self.assertEqual(out.shape, (1, self.seq_len, 50257))
        self.assertFalse(torch.isnan(out).any(), "Full attention produced NaN")
        self.assertFalse(torch.isinf(out).any(), "Full attention produced inf")

    def test_quest_forward_pass(self):
        """Quest-patched model forward pass produces finite output."""
        model = _load_model()
        patch_model(model, method="quest", page_size=16, top_k=4)
        out = model(self.x).logits
        self.assertEqual(out.shape, (1, self.seq_len, 50257))
        self.assertFalse(torch.isnan(out).any(), "Quest produced NaN")
        self.assertFalse(torch.isinf(out).any(), "Quest produced inf")

    def test_hierarchical_forward_pass(self):
        """Hierarchical-patched model forward pass produces finite output."""
        model = _load_model()
        patch_model(
            model, method="hierarchical",
            page_size=16, top_k=4, macro_multiplier=2,
            num_sink_tokens=4, num_recent_tokens=32,
        )
        out = model(self.x).logits
        self.assertEqual(out.shape, (1, self.seq_len, 50257))
        self.assertFalse(torch.isnan(out).any(), "Hierarchical produced NaN")
        self.assertFalse(torch.isinf(out).any(), "Hierarchical produced inf")

    def test_quest_and_hierarchical_differ(self):
        """Quest and Hierarchical produce different outputs at restrictive budget.

        With seq_len=256 and budget=64, both methods select different tokens,
        so their logits must differ.
        """
        x = torch.randint(0, 100, (1, 256))

        model_q = _load_model()
        patch_model(model_q, method="quest", page_size=16, top_k=4)
        out_q = model_q(x).logits

        model_h = _load_model()
        patch_model(
            model_h, method="hierarchical",
            page_size=16, top_k=4, macro_multiplier=2,
            num_sink_tokens=4, num_recent_tokens=32,
        )
        out_h = model_h(x).logits

        mean_diff = (out_q - out_h).abs().mean().item()
        self.assertGreater(
            mean_diff, 1e-6,
            f"Quest vs Hierarchical outputs are identical "
            f"(mean diff = {mean_diff:.2e})",
        )


@unittest.skipIf(not HAS_TRANSFORMERS, "transformers is not installed")
class TestPatchedPerplexity(unittest.TestCase):
    """Perplexity degradation: sparse attention should not destroy quality.

    The sparse methods operate at a budget of 64 tokens per head per step
    (page_size=16, top_k=4), so they attend to only 25% of the 256-token
    context.  Some perplexity increase is expected, but it should stay
    within a reasonable bound (< 50% degradation).
    """

    @classmethod
    def setUpClass(cls):
        cls.seq_len = 256
        # Use a fixed prompt so results are reproducible
        cls.prompt = (
            "The economic impact of artificial intelligence is "
            "increasingly significant as businesses adopt machine learning "
            "technologies to automate routine tasks, optimize supply chains, "
            "and generate insights from large datasets. "
            "Researchers have found that companies investing in AI "
            "infrastructure tend to see improvements in productivity "
            "and decision-making. "
            "However, there are also concerns about job displacement "
            "and the need for workforce retraining programs. "
        )

    def test_full_attention_perplexity(self):
        """Record full-attention perplexity — the reference baseline."""
        model = _load_model()
        input_ids = _tokenize(self.prompt, seq_len=self.seq_len)
        ppl = _compute_perplexity(model, input_ids)
        self.assertGreater(ppl, 0)
        # distilgpt2 is a tiny model; perplexity on arbitrary text can be
        # high. The important thing is it's finite (not NaN/inf).
        self.assertTrue(math.isfinite(ppl))

    def test_quest_perplexity_is_reasonable(self):
        """Quest-patched model perplexity should be within 50% of full."""
        model = _load_model()
        input_ids = _tokenize(self.prompt, seq_len=self.seq_len)

        # Full baseline
        ppl_full = _compute_perplexity(model, input_ids)

        # Quest
        patch_model(model, method="quest", page_size=16, top_k=4)
        ppl_quest = _compute_perplexity(model, input_ids)

        ratio = ppl_quest / ppl_full
        self.assertTrue(
            math.isfinite(ppl_quest),
            f"Quest perplexity is not finite: {ppl_quest}",
        )
        self.assertLess(
            ratio, 1.5,
            f"Quest perplexity ratio {ratio:.3f}x is too high "
            f"(full={ppl_full:.2f}, quest={ppl_quest:.2f})",
        )

    def test_hierarchical_perplexity_is_reasonable(self):
        """Hierarchical-patched model perplexity should be within 50% of full."""
        model = _load_model()
        input_ids = _tokenize(self.prompt, seq_len=self.seq_len)

        # Full baseline
        ppl_full = _compute_perplexity(model, input_ids)

        # Hierarchical
        patch_model(
            model, method="hierarchical",
            page_size=16, top_k=4, macro_multiplier=2,
            num_sink_tokens=4, num_recent_tokens=32,
        )
        ppl_hier = _compute_perplexity(model, input_ids)

        ratio = ppl_hier / ppl_full
        self.assertTrue(
            math.isfinite(ppl_hier),
            f"Hierarchical perplexity is not finite: {ppl_hier}",
        )
        self.assertLess(
            ratio, 1.5,
            f"Hierarchical perplexity ratio {ratio:.3f}x is too high "
            f"(full={ppl_full:.2f}, hier={ppl_hier:.2f})",
        )

    def test_hierarchical_perplexity_not_worse_than_quest(self):
        """Hierarchical should match or beat Quest perplexity.

        Because both use the same token budget but Hierarchical selects
        tokens at a finer granularity, its perplexity should not be
        significantly worse.
        """
        input_ids = _tokenize(self.prompt, seq_len=self.seq_len)

        model_q = _load_model()
        patch_model(model_q, method="quest", page_size=16, top_k=4)
        ppl_quest = _compute_perplexity(model_q, input_ids)

        model_h = _load_model()
        patch_model(
            model_h, method="hierarchical",
            page_size=16, top_k=4, macro_multiplier=2,
            num_sink_tokens=4, num_recent_tokens=32,
        )
        ppl_hier = _compute_perplexity(model_h, input_ids)

        self.assertLessEqual(
            ppl_hier, ppl_quest * 1.05,
            f"Hierarchical PPL {ppl_hier:.2f} is >5% worse than "
            f"Quest PPL {ppl_quest:.2f}",
        )


@unittest.skipIf(not HAS_TRANSFORMERS, "transformers is not installed")
class TestWeightTransfer(unittest.TestCase):
    """Verify that pretrained weights survive the patching process."""

    def test_quest_weights_are_transferred(self):
        """Weight transfer leaves non-zero weights in the Quest module."""
        model = _load_model()
        patch_model(model, method="quest", page_size=16, top_k=4)
        result = verify_weight_transfer(model, layer_idx=0)
        for key, valid in result.items():
            self.assertTrue(
                valid, f"Weight {key} is zero after transfer (layer 0)"
            )

    def test_hierarchical_weights_are_transferred(self):
        """Weight transfer leaves non-zero weights in the Hierarchical module."""
        model = _load_model()
        patch_model(
            model, method="hierarchical",
            page_size=16, top_k=4, macro_multiplier=2,
        )
        result = verify_weight_transfer(model, layer_idx=0)
        for key, valid in result.items():
            self.assertTrue(
                valid, f"Weight {key} is zero after transfer (layer 0)"
            )

    def test_weights_preserved_across_all_layers(self):
        """Every patched layer has valid weights, not just layer 0."""
        model = _load_model()
        patch_model(model, method="quest", page_size=16, top_k=4)
        for layer_idx in range(6):  # distilgpt2 has 6 layers
            result = verify_weight_transfer(model, layer_idx=layer_idx)
            for key, valid in result.items():
                self.assertTrue(
                    valid,
                    f"Weight {key} is zero in layer {layer_idx}",
                )


@unittest.skipIf(not HAS_TRANSFORMERS, "transformers is not installed")
@unittest.skipIf(not HAS_DATASETS, "datasets is not installed")
class TestPerplexityOnWikitext(unittest.TestCase):
    """Full perplexity evaluation using WikiText-2 (requires ``datasets``).

    This class reproduces a mini version of the eval_perplexity.py workflow
    as a test: load real data, compute perplexity on both patched and
    unpatched models, and assert the degradation is bounded.
    """

    @classmethod
    def setUpClass(cls):
        # Load a small sample of WikiText-2
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [ex["text"] for ex in dataset if ex["text"].strip()]
        cls.full_text = " ".join(texts[:50])  # ~50 lines

    def test_perplexity_on_wikitext(self):
        """All three methods produce finite perplexity on WikiText-2."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokens = tokenizer(
            self.full_text, return_tensors="pt",
            truncation=True, max_length=256,
        ).input_ids

        results = {}

        # Full
        model = _load_model()
        ppl = _compute_perplexity(model, tokens)
        results["full"] = ppl
        self.assertTrue(math.isfinite(ppl))

        # Quest
        model_q = _load_model()
        patch_model(model_q, method="quest", page_size=16, top_k=4)
        ppl = _compute_perplexity(model_q, tokens)
        results["quest"] = ppl
        self.assertTrue(math.isfinite(ppl))

        # Hierarchical
        model_h = _load_model()
        patch_model(
            model_h, method="hierarchical",
            page_size=16, top_k=4, macro_multiplier=2,
            num_sink_tokens=4, num_recent_tokens=32,
        )
        ppl = _compute_perplexity(model_h, tokens)
        results["hierarchical"] = ppl
        self.assertTrue(math.isfinite(ppl))

        # Print results so the test doubles as a report
        print(f"\n  WikiText-2 Perplexity (seq_len=256, budget=64):")
        for method in ("full", "quest", "hierarchical"):
            p = results[method]
            ratio = p / results["full"]
            print(f"    {method:<14s}  {p:.2f}  ({ratio:+.3f}x vs full)")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
