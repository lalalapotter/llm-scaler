"""Correctness for awq_moe_gemv — batched fused dequant+GEMV across
top-K selected experts, replacing the per-expert dequant + matmul
fallback used at decode.

Reference: independently dequant each expert, do its bf16 matmul, stack.
"""
import pytest
import torch

awq_fused_xpu = pytest.importorskip("awq_fused_xpu")
sgl_kernel = pytest.importorskip("sgl_kernel")

from awq_fused_xpu import awq_moe_gemv  # noqa: E402
from sgl_kernel import awq_dequantize  # noqa: E402


def _moe_reference(x_t, qw, sc, qz, expert_ids):
    outs = []
    for e in expert_ids.tolist():
        W = awq_dequantize(qw[e].contiguous(), sc[e].contiguous(),
                           qz[e].contiguous())
        outs.append(torch.matmul(x_t.unsqueeze(0), W).squeeze(0))
    return torch.stack(outs, dim=0)


# (E, K, N, top_k) — Qwen3.5-style MoE shapes (group=128).
@pytest.mark.parametrize(
    "E,K,N,top_k",
    [
        pytest.param(64, 2048, 1024, 4, id="E64_K2048_N1024_topk4"),
        pytest.param(128, 2560, 1280, 8, id="E128_K2560_N1280_topk8"),
    ],
)
def test_awq_moe_gemv_matches_per_expert_dequant_matmul(E, K, N, top_k,
                                                       xpu_device):
    G = 128
    qweight = torch.randint(0, 2**31 - 1, (E, K, N // 8), dtype=torch.int32,
                            device=xpu_device)
    qzeros = torch.randint(0, 2**31 - 1, (E, K // G, N // 8),
                           dtype=torch.int32, device=xpu_device)
    scales = torch.randn(E, K // G, N, dtype=torch.bfloat16,
                         device=xpu_device) * 0.01
    x_t = torch.randn(K, dtype=torch.bfloat16, device=xpu_device)
    expert_ids = torch.randperm(E, device=xpu_device)[:top_k].to(torch.int32)

    out = awq_moe_gemv(x_t, qweight, scales, qzeros, expert_ids)
    ref = _moe_reference(x_t, qweight, scales, qzeros, expert_ids)

    assert out.shape == (top_k, N)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
