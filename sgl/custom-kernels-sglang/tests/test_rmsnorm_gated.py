"""Correctness for rmsnorm_gated — fused RMSNorm + sigmoid (swish) gate.

Equivalent to RMSNormGated(x, z) with is_rms_norm=True,
norm_before_gate=True, activation='swish', bias=None, group_size=None.
"""
import pytest
import torch
import torch.nn.functional as F

awq_fused_xpu = pytest.importorskip("awq_fused_xpu")
from awq_fused_xpu import rmsnorm_gated  # noqa: E402


def _reference(x, z, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    yn = xn * weight.float()
    return (yn * F.silu(z.float())).to(x.dtype)


# Real GDN shape (M=32, N=128) plus stresses on M and a couple of N values.
@pytest.mark.parametrize(
    "M,N",
    [
        pytest.param(32, 128, id="M32_N128_GDN"),
        pytest.param(128, 128, id="M128_N128"),
        pytest.param(2048, 128, id="M2048_N128"),
        pytest.param(32, 256, id="M32_N256"),
        pytest.param(64, 512, id="M64_N512"),
    ],
)
def test_rmsnorm_gated_matches_pytorch_reference(M, N, xpu_device):
    eps = 1e-6
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=xpu_device)
    z = torch.randn(M, N, dtype=dtype, device=xpu_device)
    w = torch.randn(N, dtype=dtype, device=xpu_device)

    out = rmsnorm_gated(x, z, w, eps)
    ref = _reference(x, z, w, eps)
    assert out.shape == x.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
