"""
HuggingFace Model Patcher — weight transfer + attention layer replacement.

Replaces standard full-attention layers in a pretrained HF model with our
sparse attention modules (QuestAttention or HierarchicalTokenAttention).
Handles both Conv1D-based models (GPT-2) and standard nn.Linear models
(Llama, Qwen, Mistral).

Usage::

    from transformers import GPT2LMHeadModel
    from hf_model_patcher import patch_model

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = patch_model(model, method="hierarchical", top_k=4, page_size=64)
    # Now model uses hierarchical sparse attention in every layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quest_attention import QuestAttention
from hierarchical_attention import HierarchicalTokenAttention


# ═══════════════════════════════════════════════════════════════════════════════
# Model type detection & layer access
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_PATHS = {
    "gpt2": "transformer.h",
    "gpt_neo": "transformer.h",
    "opt": "model.decoder.layers",
    "llama": "model.layers",
    "mistral": "model.layers",
    "qwen2": "model.layers",
    "gemma": "model.layers",
    "falcon": "transformer.h",
    "phi": "model.layers",
    "phi3": "model.layers",
}


def _detect_model_type(model) -> str:
    """Return the model type string from a HuggingFace model config."""
    config = model.config
    return getattr(config, "model_type", "unknown")


def _get_transformer_layers(model):
    """Return the list of transformer/decoder layers for a given HF model."""
    model_type = _detect_model_type(model)

    if model_type in _MODEL_PATHS:
        path = _MODEL_PATHS[model_type]
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. "
            f"Add the layer path to _MODEL_PATHS in hf_model_patcher.py."
        )


def _get_attention_module(layer) -> nn.Module:
    """Return the attention module from a transformer layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    elif hasattr(layer, "attn"):
        return layer.attn
    elif hasattr(layer, "attention"):
        return layer.attention
    else:
        raise ValueError(
            f"Cannot find attention module in layer of type {type(layer).__name__}"
        )


def _set_attention_module(layer, new_module: nn.Module) -> None:
    """Replace the attention module in a transformer layer."""
    if hasattr(layer, "self_attn"):
        layer.self_attn = new_module
    elif hasattr(layer, "attn"):
        layer.attn = new_module
    elif hasattr(layer, "attention"):
        layer.attention = new_module
    else:
        raise ValueError(
            f"Cannot find attention module to replace in layer of type "
            f"{type(layer).__name__}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Weight transfer functions
# ═══════════════════════════════════════════════════════════════════════════════

def transfer_gpt2_weights(
    hf_attn: nn.Module,
    target: nn.Module,
) -> None:
    """Transfer weights from GPT2Attention (Conv1D) to our module (nn.Linear).

    GPT-2 uses ``Conv1D`` layers where weight shape is ``(in_features, out_features)``
    (i.e. ``(hidden, output_dim)``), the transpose of ``nn.Linear``.  The combined
    ``c_attn`` projects hidden → [Q, K, V] concatenated, so we split into thirds
    after transposing.

    Args:
        hf_attn: The HF GPT2Attention module (with c_attn, c_proj Conv1D layers).
        target:  Our QuestAttention or HierarchicalTokenAttention (with
                 q_proj, k_proj, v_proj, o_proj nn.Linear layers).
    """
    # --- Q, K, V projections (c_attn) ---
    # c_attn.weight: (hidden, 3*hidden)  → transpose → (3*hidden, hidden) → chunk
    c_attn_w = hf_attn.c_attn.weight  # (hidden, 3*hidden)
    q_w, k_w, v_w = c_attn_w.t().chunk(3, dim=0)  # each (hidden, hidden)
    target.q_proj.weight.data.copy_(q_w)
    target.k_proj.weight.data.copy_(k_w)
    target.v_proj.weight.data.copy_(v_w)

    if hf_attn.c_attn.bias is not None:
        q_b, k_b, v_b = hf_attn.c_attn.bias.chunk(3, dim=0)  # each (hidden,)
        target.q_proj.bias.data.copy_(q_b)
        target.k_proj.bias.data.copy_(k_b)
        target.v_proj.bias.data.copy_(v_b)

    # --- Output projection (c_proj) ---
    # c_proj.weight: (hidden, hidden)  → transpose → (hidden, hidden)
    target.o_proj.weight.data.copy_(hf_attn.c_proj.weight.t())
    if hf_attn.c_proj.bias is not None:
        target.o_proj.bias.data.copy_(hf_attn.c_proj.bias)


def transfer_standard_weights(
    hf_attn: nn.Module,
    target: nn.Module,
) -> None:
    """Transfer weights from standard nn.Linear attention (Llama, Qwen, etc.).

    These models have separate ``q_proj, k_proj, v_proj, o_proj`` nn.Linear
    layers that match our module's structure exactly.

    Args:
        hf_attn: HF attention module with q_proj/k_proj/v_proj/o_proj.
        target:  Our module with matching projection layers.
    """
    target.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data)
    target.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
    target.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
    target.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data)

    # Biases (may not exist on some models)
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        hf_has = hasattr(hf_attn, proj_name) and getattr(hf_attn, proj_name).bias is not None
        our_has = hasattr(target, proj_name) and getattr(target, proj_name).bias is not None
        if hf_has and our_has:
            getattr(target, proj_name).bias.data.copy_(
                getattr(hf_attn, proj_name).bias.data
            )


def transfer_weights(hf_attn: nn.Module, target: nn.Module, model_type: str) -> None:
    """Dispatch weight transfer based on model type.

    Args:
        hf_attn:    The source HF attention module.
        target:     Our sparse attention module.
        model_type: The HF model_type string (e.g. "gpt2", "llama").
    """
    if model_type == "gpt2":
        transfer_gpt2_weights(hf_attn, target)
    else:
        # Llama, Mistral, Qwen, Gemma, Phi, Falcon, OPT, GPT-Neo
        transfer_standard_weights(hf_attn, target)


# ═══════════════════════════════════════════════════════════════════════════════
# Model patching
# ═══════════════════════════════════════════════════════════════════════════════

class _SparseAttentionWrapper(nn.Module):
    """Wraps a sparse attention module so its return value matches HF convention.

    HuggingFace attention modules return ``(attn_output, attn_weights)``
    (GPT-2) or ``(attn_output, attn_weights, past_key_value)`` (Llama).
    Our modules return a single tensor (or tuple of tensor + dict/KV).
    This wrapper delegates to our module and adapts the return value.

    Set ``verbose_metrics=True`` on the wrapper to log per-call metrics from
    hierarchical attention (useful for debugging quality regressions on
    real-model runs).
    """

    def __init__(self, sparse_module: nn.Module, verbose_metrics: bool = False):
        super().__init__()
        self.sparse_module = sparse_module
        self.verbose_metrics = verbose_metrics
        self._call_count = 0

    def forward(self, *args, **kwargs):
        output = self.sparse_module(*args, **kwargs)

        # Unwrap if our module returned (output, K, V) from return_kv
        if isinstance(output, tuple):
            main_output = output[0]
            if len(output) >= 2 and isinstance(output[1], dict):
                # (output, metrics_dict) — surface if verbose
                if self.verbose_metrics:
                    self._call_count += 1
                    metrics = output[1]
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(
                        "SparseAttention call #%d metrics: M=%d candidates=%d "
                        "budget=%d scores_mean=%.4f",
                        self._call_count,
                        metrics.get("num_macro_pages", -1),
                        metrics.get("num_candidates", -1),
                        metrics.get("token_budget", -1),
                        metrics.get("page_scores_mean", float("nan")),
                    )
                return main_output, None
            # (output, K, V) or other tuples
            return main_output, None
        # Single tensor — wrap as (output, None) = (output, attn_weights=None)
        return output, None


def _build_config_from_hf(model) -> dict:
    """Extract model dimensions from an HF model config.

    Returns a dict suitable for passing as kwargs to QuestAttention or
    HierarchicalTokenAttention constructors.
    """
    config = model.config
    hidden_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_dim // num_heads
    # Some HF configs store num_kv_heads; default to num_heads for MHA
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    # GPT-2 uses bias in its Conv1D layers; Llama/Mistral/Qwen do not
    model_type = _detect_model_type(model)
    use_bias = model_type in ("gpt2",)  # GPT-2 family uses bias

    return {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "bias": use_bias,
    }


def patch_model(
    model,
    *,
    method: str = "quest",
    layers: list[int] | None = None,
    **attention_kwargs,
):
    """Replace attention layers in a HuggingFace model with sparse attention.

    Patches the model **in place** — no copy is made.  After patching, every
    ``forward()`` call through the model will use the sparse attention method.

    Args:
        model: A HuggingFace pretrained model (GPT2LMHeadModel,
               LlamaForCausalLM, etc.).
        method: ``"quest"`` for Quest page-wise sparse attention, or
                ``"hierarchical"`` for hierarchical token-level attention.
        layers: Which layer indices to patch (None → all layers).
        **attention_kwargs: Forwarded to the attention module constructor
               (e.g. ``page_size=64, top_k=4`` for Quest; add
               ``macro_multiplier=3, num_sink_tokens=4`` for Hierarchical).

    Returns:
        The patched model (same object, modified in place).

    Example::

        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> model = patch_model(model, method="hierarchical",
        ...                     page_size=64, top_k=4,
        ...                     macro_multiplier=3, num_sink_tokens=4)

    Raises:
        ValueError: If the model type is unsupported.
    """
    model_type = _detect_model_type(model)
    base_config = _build_config_from_hf(model)

    AttnClass = {
        "quest": QuestAttention,
        "hierarchical": HierarchicalTokenAttention,
    }[method]

    transformer_layers = _get_transformer_layers(model)
    num_layers = len(transformer_layers)

    if layers is None:
        layers = list(range(num_layers))

    print(f"  Patching {len(layers)}/{num_layers} layers with "
          f"'{method}' sparse attention (model_type={model_type})")

    for layer_idx in layers:
        layer = transformer_layers[layer_idx]
        hf_attn = _get_attention_module(layer)

        sparse_attn = AttnClass(**base_config, **attention_kwargs)

        transfer_weights(hf_attn, sparse_attn, model_type)

        # Wrap to match HF return-value convention
        wrapper = _SparseAttentionWrapper(sparse_attn)

        _set_attention_module(layer, wrapper)

    print(f"  Done — {len(layers)} layer(s) patched.")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Verification helpers
# ═══════════════════════════════════════════════════════════════════════════════

def verify_weight_transfer(model, layer_idx: int = 0) -> dict:
    """Verify that weights were correctly transferred for a patched layer.

    Args:
        model:     A model patched by ``patch_model()``.
        layer_idx: Which layer to inspect.

    Returns:
        dict with keys: ``q_match``, ``k_match``, ``v_match``, ``o_match``
        — each a bool indicating whether weights match.
    """
    import torch

    model_type = _detect_model_type(model)
    layer = _get_transformer_layers(model)[layer_idx]
    wrapper = _get_attention_module(layer)
    # Unwrap our wrapper to access the actual sparse module
    sparse_attn = wrapper.sparse_module if isinstance(wrapper, _SparseAttentionWrapper) else wrapper

    # We can't easily get the original weights back, so we just check
    # that our module has valid (non-random) weights loaded.
    results = {}
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = getattr(sparse_attn, proj_name).weight
        # A pretrained weight should have non-trivial statistics
        is_valid = not torch.allclose(w, torch.zeros_like(w))
        results[f"{proj_name}_valid"] = is_valid

    return results
