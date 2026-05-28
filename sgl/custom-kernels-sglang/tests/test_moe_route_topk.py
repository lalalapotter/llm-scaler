"""Correctness for moe_route_topk — fused softmax + top-K + (optional)
renormalize. Replaces (softmax, topk, sum, div) chain that fires once
per MoE layer at decode.
"""
import pytest
import torch

awq_fused_xpu = pytest.importorskip("awq_fused_xpu")
from awq_fused_xpu import moe_route_topk  # noqa: E402


def _reference_topk(logits, top_k, renormalize):
    probs = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(torch.float32), ids.to(torch.int32)


@pytest.mark.parametrize(
    "B,E,top_k",
    [
        pytest.param(1, 64, 8, id="B1_E64_topk8"),
        pytest.param(1, 128, 8, id="B1_E128_topk8"),
        pytest.param(1, 256, 8, id="B1_E256_topk8"),
        pytest.param(4, 128, 8, id="B4_E128_topk8"),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
def test_moe_route_topk_matches_torch_softmax_topk(B, E, top_k, renormalize,
                                                   xpu_device):
    logits = torch.randn(B, E, dtype=torch.bfloat16, device=xpu_device)
    weights, ids = moe_route_topk(logits, top_k, renormalize)
    ref_weights, ref_ids = _reference_topk(logits, top_k, renormalize)

    # Top-K id set must match (top_k tied entries are unlikely w/ randn,
    # but compare as sets to be safe). Weights for the matched ids must
    # also match.
    assert weights.shape == (B, top_k)
    assert ids.shape == (B, top_k)
    assert ids.dtype == torch.int32

    for b in range(B):
        got_set = set(ids[b].tolist())
        ref_set = set(ref_ids[b].tolist())
        assert got_set == ref_set, f"top-K ids differ at batch {b}"

    # Sort both by id for direct weight comparison.
    weights_sorted = torch.gather(
        weights, 1, torch.argsort(ids.long(), dim=-1)
    )
    ref_sorted = torch.gather(
        ref_weights, 1, torch.argsort(ref_ids.long(), dim=-1)
    )
    torch.testing.assert_close(weights_sorted, ref_sorted, rtol=2e-3, atol=2e-3)
