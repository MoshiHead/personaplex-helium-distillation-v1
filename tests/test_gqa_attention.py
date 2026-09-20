# SPDX-License-Identifier: MIT
"""Streaming vs. non-streaming consistency for the student's GQA attention
(distill/gqa_attention.py) -- the same style of check
moshi/modules/streaming.py's own `test()` runs for its conv layers. Fast,
CPU-only, no checkpoints required.
"""

import torch

from distill.gqa_attention import GQAStreamingTransformer


def _build(d_model=64, num_heads=8, num_kv_heads=2, num_layers=2, context=64):
    torch.manual_seed(0)
    return GQAStreamingTransformer(
        d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads, num_layers=num_layers,
        dim_feedforward=128, causal=True, context=context,
    ).eval()


def test_streaming_matches_batch():
    model = _build()
    x = torch.randn(2, 20, 64)

    with torch.no_grad():
        full = model(x)

        streamed = []
        with model.streaming(2):
            for t in range(x.shape[1]):
                streamed.append(model(x[:, t:t + 1]))
        streamed = torch.cat(streamed, dim=1)

    assert full.shape == streamed.shape
    delta = (full - streamed).abs().max().item()
    assert delta < 1e-4, f"streaming drifted from batch forward by {delta}"


def test_gqa_head_grouping_matches_full_mha_when_kv_equals_q():
    """When num_kv_heads == num_heads, GQA degenerates to plain MHA (no pooling)."""
    model = _build(num_heads=4, num_kv_heads=4)
    layer = model.layers[0]
    attn = layer.self_attn
    assert attn.num_groups == 1
    x = torch.randn(1, 5, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_invalid_head_config_raises():
    import pytest

    with pytest.raises(AssertionError):
        GQAStreamingTransformer(
            d_model=64, num_heads=8, num_kv_heads=3, num_layers=1, dim_feedforward=128, context=16,
        )
