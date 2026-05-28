"""Correctness tests for awq_fused_xpu fused dequant+GEMV ops.

Three on-device variants of the same math, all should agree with
sgl_kernel.awq_dequantize + torch.matmul as the reference:

  * awq_gemv_fused          — non-repacked qweight (run-as-is)
  * awq_gemv_fused_repacked — qweight repacked once via awq_repack_packed_n
  * awq_gemv_fused_pp       — qweight as-is, scales/zeros prepacked to fp32

Layout matches sgl-kernel-xpu/awq_dequantize.cpp:
  qweight [K, N/8] int32, qzeros [K/group, N/8] int32, scales [K/group, N]
"""
import pytest
import torch

awq_fused_xpu = pytest.importorskip("awq_fused_xpu")
sgl_kernel = pytest.importorskip("sgl_kernel")

from awq_fused_xpu import (  # noqa: E402
    awq_gemv_fused,
    awq_gemv_fused_pp,
    awq_gemv_fused_repacked,
    awq_prepack_scales_zeros,
    awq_repack_packed_n,
)
from sgl_kernel import awq_dequantize  # noqa: E402

# Decode shapes from the Qwen3.5-AWQ MLP / attn projections.
SHAPES = [
    pytest.param(2048, 6144, id="K2048_N6144"),    # 2B gate/up
    pytest.param(6144, 2048, id="K6144_N2048"),    # 2B down
    pytest.param(2560, 10240, id="K2560_N10240"),  # 4B gate/up
    pytest.param(10240, 2560, id="K10240_N2560"),  # 4B down
    pytest.param(2048, 512, id="K2048_N512"),      # narrow attn out
    pytest.param(4096, 1024, id="K4096_N1024"),    # smaller down
]


def _make_inputs(K, N, group, device, dtype):
    qweight = torch.randint(
        0, 2**31 - 1, (K, N // 8), dtype=torch.int32, device=device
    )
    qzeros = torch.randint(
        0, 2**31 - 1, (K // group, N // 8), dtype=torch.int32, device=device
    )
    scales = torch.randn(K // group, N, dtype=dtype, device=device) * 0.01
    x = torch.randn(1, K, dtype=dtype, device=device)
    return x, qweight, scales, qzeros


def _reference(x, qweight, scales, qzeros):
    W = awq_dequantize(qweight, scales, qzeros)
    return torch.matmul(x, W)


@pytest.mark.parametrize("K,N", SHAPES)
def test_awq_gemv_fused_matches_dequant_matmul(K, N, xpu_device):
    x, qw, sc, qz = _make_inputs(K, N, group=128, device=xpu_device,
                                 dtype=torch.bfloat16)
    out = awq_gemv_fused(x, qw, sc, qz)
    ref = _reference(x, qw, sc, qz)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("K,N", SHAPES)
def test_awq_gemv_fused_repacked_matches_dequant_matmul(K, N, xpu_device):
    x, qw, sc, qz = _make_inputs(K, N, group=128, device=xpu_device,
                                 dtype=torch.bfloat16)
    qw_repacked = awq_repack_packed_n(qw)
    out = awq_gemv_fused_repacked(x, qw_repacked, sc, qz, N)
    ref = _reference(x, qw, sc, qz)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("K,N", SHAPES)
def test_awq_gemv_fused_pp_matches_dequant_matmul(K, N, xpu_device):
    x, qw, sc, qz = _make_inputs(K, N, group=128, device=xpu_device,
                                 dtype=torch.bfloat16)
    sc_fp32, z_fp32 = awq_prepack_scales_zeros(sc, qz)
    out = awq_gemv_fused_pp(x, qw, sc_fp32, z_fp32)
    ref = _reference(x, qw, sc, qz)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_awq_repack_packed_n_shape(xpu_device):
    K, N = 2048, 6144
    qw = torch.randint(0, 2**31 - 1, (K, N // 8), dtype=torch.int32,
                       device=xpu_device)
    out = awq_repack_packed_n(qw)
    # Layout transform: (K, N/8) -> (N/8, K/16, 16); same int32 element count.
    assert out.dtype == torch.int32
    assert out.numel() == qw.numel()


def test_awq_prepack_scales_zeros_shapes(xpu_device):
    K, N, G = 2048, 6144, 128
    sc = torch.randn(K // G, N, dtype=torch.bfloat16, device=xpu_device)
    qz = torch.randint(0, 2**31 - 1, (K // G, N // 8), dtype=torch.int32,
                       device=xpu_device)
    sc_fp32, z_fp32 = awq_prepack_scales_zeros(sc, qz)
    assert sc_fp32.dtype == torch.float32
    assert z_fp32.dtype == torch.float32
    assert sc_fp32.shape[-1] == N
    assert z_fp32.shape[-1] == N
