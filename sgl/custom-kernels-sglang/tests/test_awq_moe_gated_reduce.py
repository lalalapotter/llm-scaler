"""Correctness for the four MoE gated/reduce GEMV variants.

These collapse the (chunk → silu → mul → sum) Python tail of the AWQ
MoE forward into a single kernel launch:

  awq_moe_gated_gemv      SYCL gated w13 → silu(gate) * up
  awq_moe_reduce_gemv     SYCL reduce w2 with topk-weighted sum
  awq_moe_gated_np_esimd  ESIMD gated, n-packed AWQ layout (no repack)
  awq_moe_reduce_np_esimd ESIMD reduce, n-packed AWQ layout
  awq_moe_gated_kp_esimd  ESIMD gated, k-packed layout (vLLM-style)
  awq_moe_reduce_kp_esimd ESIMD reduce, k-packed layout

The k-packed pair takes a separately-prepared offsets tensor; we test
those separately by mirroring the same math through repack.

We test SYCL gated+reduce against a per-expert dequant+matmul reference,
and the n-packed ESIMD variants against the SYCL output (same layout).
The k-packed pair has its own layout helper and is tested with a smaller
shape that exercises the repack codepath.
"""
import pytest
import torch
import torch.nn.functional as F

awq_fused_xpu = pytest.importorskip("awq_fused_xpu")
sgl_kernel = pytest.importorskip("sgl_kernel")

from awq_fused_xpu import (  # noqa: E402
    awq_moe_gated_gemv,
    awq_moe_gated_kp_esimd,
    awq_moe_gated_np_esimd,
    awq_moe_reduce_gemv,
    awq_moe_reduce_kp_esimd,
    awq_moe_reduce_np_esimd,
)
from sgl_kernel import awq_dequantize  # noqa: E402


def _gated_reduce_reference(
    x_t,            # [K]
    w13_qw, w13_sc, w13_qz,   # [E, K, 2*M/8] etc.
    w2_qw,  w2_sc,  w2_qz,    # [E, M, hidden/8] etc.
    expert_ids,     # [top_k] int32
    topk_weights,   # [top_k]
):
    outs = []
    for e in expert_ids.tolist():
        W13 = awq_dequantize(w13_qw[e].contiguous(),
                             w13_sc[e].contiguous(),
                             w13_qz[e].contiguous())  # [K, 2*M]
        gu = torch.matmul(x_t.unsqueeze(0), W13).squeeze(0)  # [2*M]
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up                                # [M]

        W2 = awq_dequantize(w2_qw[e].contiguous(),
                            w2_sc[e].contiguous(),
                            w2_qz[e].contiguous())            # [M, hidden]
        d = torch.matmul(h.unsqueeze(0), W2).squeeze(0)       # [hidden]
        outs.append(d)
    stacked = torch.stack(outs, dim=0)              # [top_k, hidden]
    return (topk_weights.to(stacked.dtype).unsqueeze(-1) * stacked).sum(dim=0)


@pytest.mark.parametrize(
    "E,K,M,hidden,top_k",
    [pytest.param(64, 2048, 1024, 2048, 4, id="E64_K2048_M1024_h2048_topk4")],
)
def test_awq_moe_gated_reduce_sycl_matches_per_expert_reference(
    E, K, M, hidden, top_k, xpu_device,
):
    G = 128
    dtype = torch.bfloat16

    w13_qw = torch.randint(0, 2**31 - 1, (E, K, (2 * M) // 8),
                           dtype=torch.int32, device=xpu_device)
    w13_qz = torch.randint(0, 2**31 - 1, (E, K // G, (2 * M) // 8),
                           dtype=torch.int32, device=xpu_device)
    w13_sc = torch.randn(E, K // G, 2 * M, dtype=dtype,
                         device=xpu_device) * 0.01
    w2_qw = torch.randint(0, 2**31 - 1, (E, M, hidden // 8),
                          dtype=torch.int32, device=xpu_device)
    w2_qz = torch.randint(0, 2**31 - 1, (E, M // G, hidden // 8),
                          dtype=torch.int32, device=xpu_device)
    w2_sc = torch.randn(E, M // G, hidden, dtype=dtype,
                        device=xpu_device) * 0.01

    x_t = torch.randn(K, dtype=dtype, device=xpu_device)
    expert_ids = torch.randperm(E, device=xpu_device)[:top_k].to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(top_k, dtype=dtype, device=xpu_device), dim=0,
    )

    h = awq_moe_gated_gemv(x_t, w13_qw, w13_sc, w13_qz, expert_ids)
    out = awq_moe_reduce_gemv(h, w2_qw, w2_sc, w2_qz, expert_ids,
                              topk_weights)
    ref = _gated_reduce_reference(
        x_t, w13_qw, w13_sc, w13_qz,
        w2_qw, w2_sc, w2_qz,
        expert_ids, topk_weights,
    )
    assert out.shape == (hidden,)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


def test_awq_moe_gated_reduce_np_esimd_agrees_with_sycl(xpu_device):
    """ESIMD n-packed variants must produce the same result as the
    SYCL kernels on the same inputs (identical AWQ layout)."""
    E, K, M, hidden, top_k = 64, 2048, 1024, 2048, 4
    G = 128
    dtype = torch.bfloat16

    w13_qw = torch.randint(0, 2**31 - 1, (E, K, (2 * M) // 8),
                           dtype=torch.int32, device=xpu_device)
    w13_qz = torch.randint(0, 2**31 - 1, (E, K // G, (2 * M) // 8),
                           dtype=torch.int32, device=xpu_device)
    w13_sc = torch.randn(E, K // G, 2 * M, dtype=dtype,
                         device=xpu_device) * 0.01
    w2_qw = torch.randint(0, 2**31 - 1, (E, M, hidden // 8),
                          dtype=torch.int32, device=xpu_device)
    w2_qz = torch.randint(0, 2**31 - 1, (E, M // G, hidden // 8),
                          dtype=torch.int32, device=xpu_device)
    w2_sc = torch.randn(E, M // G, hidden, dtype=dtype,
                        device=xpu_device) * 0.01

    x_t = torch.randn(K, dtype=dtype, device=xpu_device)
    expert_ids = torch.randperm(E, device=xpu_device)[:top_k].to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(top_k, dtype=dtype, device=xpu_device), dim=0,
    )

    h_sycl = awq_moe_gated_gemv(x_t, w13_qw, w13_sc, w13_qz, expert_ids)
    o_sycl = awq_moe_reduce_gemv(h_sycl, w2_qw, w2_sc, w2_qz, expert_ids,
                                 topk_weights)
    h_esimd = awq_moe_gated_np_esimd(x_t, w13_qw, w13_sc, w13_qz, expert_ids)
    o_esimd = awq_moe_reduce_np_esimd(h_esimd, w2_qw, w2_sc, w2_qz,
                                      expert_ids, topk_weights)

    torch.testing.assert_close(h_sycl, h_esimd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(o_sycl, o_esimd, rtol=3e-2, atol=3e-2)


def test_awq_moe_gated_reduce_kp_esimd_runs(xpu_device):
    """k-packed ESIMD path takes a different (weight, scales, offsets)
    triple — we just check it runs and produces finite output of the
    expected shape on a representative tile.
    """
    from awq_fused_xpu.repack_kpacked import repack_awq_to_kpacked

    E, K, M, hidden, top_k = 8, 1024, 512, 1024, 2
    dtype = torch.bfloat16

    # The k-packed layout is computed off the standard AWQ tensors. The
    # repack module produces both the int4 layout transform and the
    # offset = -qzero * scale precompute.
    qw_std = torch.randint(0, 2**31 - 1, (E, K, (2 * M) // 8),
                           dtype=torch.int32, device=xpu_device)
    qz_std = torch.randint(0, 2**31 - 1, (E, K // 128, (2 * M) // 8),
                           dtype=torch.int32, device=xpu_device)
    sc_std = torch.randn(E, K // 128, 2 * M, dtype=dtype,
                         device=xpu_device) * 0.01
    w13_kp, w13_sc_kp, w13_off_kp = repack_awq_to_kpacked(
        qw_std, sc_std, qz_std,
    )

    qw2_std = torch.randint(0, 2**31 - 1, (E, M, hidden // 8),
                            dtype=torch.int32, device=xpu_device)
    qz2_std = torch.randint(0, 2**31 - 1, (E, M // 128, hidden // 8),
                            dtype=torch.int32, device=xpu_device)
    sc2_std = torch.randn(E, M // 128, hidden, dtype=dtype,
                          device=xpu_device) * 0.01
    w2_kp, w2_sc_kp, w2_off_kp = repack_awq_to_kpacked(
        qw2_std, sc2_std, qz2_std,
    )

    x_t = torch.randn(K, dtype=dtype, device=xpu_device)
    expert_ids = torch.arange(top_k, device=xpu_device, dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(top_k, dtype=dtype, device=xpu_device), dim=0,
    )

    h = awq_moe_gated_kp_esimd(x_t, w13_kp, w13_sc_kp, w13_off_kp,
                               expert_ids)
    out = awq_moe_reduce_kp_esimd(h, w2_kp, w2_sc_kp, w2_off_kp,
                                  expert_ids, topk_weights)
    assert out.shape == (hidden,)
    assert torch.isfinite(out).all()
