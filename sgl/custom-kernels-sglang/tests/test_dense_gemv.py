"""Correctness for dense bf16 GEMV (M=1) — checks the SYCL kernel
matches torch.matmul on the linears AWQ leaves un-quantised in
Qwen3.5 (linear_attn / GDN projections, layer-0 MLP).
"""
import pytest
import torch

awq_fused_xpu = pytest.importorskip("awq_fused_xpu")
from awq_fused_xpu import dense_gemv  # noqa: E402

# (out=N, in=K) — matches PyTorch nn.Linear weight shape.
SHAPES = [
    pytest.param(2048,  2048,  id="GDN_out_proj"),
    pytest.param(8192,  2048,  id="GDN_in_proj_qkvz"),
    pytest.param(32,    2048,  id="GDN_in_proj_ba"),
    pytest.param(5120,  2048,  id="self_attn_qkv_proj"),
    pytest.param(2048,  6144,  id="layer0_down_proj"),
    pytest.param(12288, 2048,  id="layer0_in_proj_qkvz"),
]


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_gemv_matches_torch_matmul(N, K, dtype, xpu_device):
    x = torch.randn(1, K, dtype=dtype, device=xpu_device)
    W = torch.randn(N, K, dtype=dtype, device=xpu_device) * 0.01
    out = dense_gemv(x, W)
    ref = torch.matmul(x, W.t())
    assert out.shape == (1, N)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
