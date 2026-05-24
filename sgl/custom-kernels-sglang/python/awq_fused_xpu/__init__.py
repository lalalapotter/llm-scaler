import os as _os
import glob as _glob
import torch as _torch

_here = _os.path.dirname(__file__)
_so = _glob.glob(_os.path.join(_here, "_C*.so"))
if not _so:
    raise ImportError("awq_fused_xpu native library not found at " + _here)
_torch.ops.load_library(_so[0])

import torch


def awq_gemv_fused(x: torch.Tensor, qweight: torch.Tensor,
                   scales: torch.Tensor, qzeros: torch.Tensor) -> torch.Tensor:
    """Fused AWQ dequant + GEMV (non-repacked qweight).

    out = x @ dequant(qweight, scales, qzeros)
    """
    return torch.ops.awq_fused_xpu.awq_gemv_fused(x, qweight, scales, qzeros)


def awq_repack_packed_n(qweight: torch.Tensor) -> torch.Tensor:
    """One-time layout transform of AWQ qweight for the fused-repacked kernel.

    Input  qweight : [K, N/8]            int32 (standard AWQ layout)
    Output         : [N/8, K/16, 16]     int32 (block-tiled along K)

    Run once at model load.
    """
    return torch.ops.awq_fused_xpu.awq_repack_packed_n(qweight)


def awq_gemv_fused_repacked(x: torch.Tensor, qweight_repacked: torch.Tensor,
                            scales: torch.Tensor, qzeros: torch.Tensor,
                            N: int) -> torch.Tensor:
    """Fused AWQ dequant + GEMV on the repacked qweight.

    qweight_repacked must come from awq_repack_packed_n().
    """
    return torch.ops.awq_fused_xpu.awq_gemv_fused_repacked(
        x, qweight_repacked, scales, qzeros, N)


def awq_prepack_scales_zeros(scales: torch.Tensor, qzeros: torch.Tensor):
    """Prepack scales (bf16/fp16) and qzeros (int32 packed) into fp32 form.

    Returns (scales_fp32, zeros_fp32). Run once at model load.
    """
    return torch.ops.awq_fused_xpu.awq_prepack_scales_zeros(scales, qzeros)


def awq_gemv_fused_pp(x: torch.Tensor, qweight: torch.Tensor,
                      scales_fp32: torch.Tensor,
                      zeros_fp32: torch.Tensor) -> torch.Tensor:
    """Fused AWQ dequant + GEMV using prepacked fp32 scales/zeros."""
    return torch.ops.awq_fused_xpu.awq_gemv_fused_pp(
        x, qweight, scales_fp32, zeros_fp32)


def dense_gemv(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Dense bf16/fp16 GEMV for batch=1 decode.

    x : [M, K]   contiguous   bf16/fp16
    W : [N, K]   contiguous   bf16/fp16   (PyTorch nn.Linear weight)
    returns out : [M, N]      same dtype as x

    Computes  out[m, n] = sum_k x[m, k] * W[n, k]   (==  x @ W.T)
    """
    return torch.ops.awq_fused_xpu.dense_gemv(x, W)


def awq_moe_gemv(x: torch.Tensor, qweight: torch.Tensor,
                 scales: torch.Tensor, qzeros: torch.Tensor,
                 expert_ids: torch.Tensor) -> torch.Tensor:
    """Batched fused AWQ dequant + GEMV for MoE decode.

    x          : [K] or [1, K] (shared) or [B, K] (per-batch)
    qweight    : [E, K, N/8]           int32
    scales     : [E, K/group, N]       bf16/fp16
    qzeros     : [E, K/group, N/8]     int32
    expert_ids : [B]                   int32 (B = top_k)

    Returns out : [B, N]   same dtype as x.
    """
    return torch.ops.awq_fused_xpu.awq_moe_gemv(
        x, qweight, scales, qzeros, expert_ids)


def awq_moe_gated_gemv(x: torch.Tensor, qweight: torch.Tensor,
                       scales: torch.Tensor, qzeros: torch.Tensor,
                       expert_ids: torch.Tensor) -> torch.Tensor:
    """Fused w13 (gate+up) GEMV with silu(gate) * up activation.

    qweight stores [gate_proj | up_proj] concatenated along N. Returns
    silu(x @ gate) * (x @ up) directly — saves the chunk/silu/mul Python ops.

    x          : [K] or [1, K]                          bf16/fp16
    qweight    : [E, K, 2*moe_inter/8]                  int32
    scales     : [E, K/group, 2*moe_inter]              bf16/fp16
    qzeros     : [E, K/group, 2*moe_inter/8]            int32
    expert_ids : [B]                                    int32

    Returns out : [B, moe_inter]   same dtype as x.
    """
    return torch.ops.awq_fused_xpu.awq_moe_gated_gemv(
        x, qweight, scales, qzeros, expert_ids)


def awq_moe_reduce_gemv(x: torch.Tensor, qweight: torch.Tensor,
                        scales: torch.Tensor, qzeros: torch.Tensor,
                        expert_ids: torch.Tensor,
                        topk_weights: torch.Tensor) -> torch.Tensor:
    """Fused w2 GEMV with topk-weighted reduction.

    Computes  sum_b topk_weights[b] * (x[b] @ W2[expert_ids[b]])  in one launch.
    Replaces a (top_k, hidden) intermediate + Python `(ws * d).sum(0)` reduce.

    x            : [B, K]                               bf16/fp16
    qweight      : [E, K, hidden/8]                     int32
    scales       : [E, K/group, hidden]                 bf16/fp16
    qzeros       : [E, K/group, hidden/8]               int32
    expert_ids   : [B]                                  int32
    topk_weights : [B]                                  bf16/fp16

    Returns out : [hidden]   same dtype as x.
    """
    return torch.ops.awq_fused_xpu.awq_moe_reduce_gemv(
        x, qweight, scales, qzeros, expert_ids, topk_weights)


def awq_moe_gated_np_esimd(x: torch.Tensor, qweight: torch.Tensor,
                           scales: torch.Tensor, qzeros: torch.Tensor,
                           expert_ids: torch.Tensor) -> torch.Tensor:
    """ESIMD gated GEMV reading raw AWQ N-packed layout (no repack)."""
    return torch.ops.awq_fused_xpu.awq_moe_gated_np_esimd(
        x, qweight, scales, qzeros, expert_ids)


def awq_moe_reduce_np_esimd(hidden: torch.Tensor, qweight: torch.Tensor,
                            scales: torch.Tensor, qzeros: torch.Tensor,
                            expert_ids: torch.Tensor,
                            topk_weights: torch.Tensor) -> torch.Tensor:
    """ESIMD reduce GEMV (w2) reading raw AWQ N-packed layout."""
    return torch.ops.awq_fused_xpu.awq_moe_reduce_np_esimd(
        hidden, qweight, scales, qzeros, expert_ids, topk_weights)


def awq_moe_gated_kp_esimd(x: torch.Tensor, weight: torch.Tensor,
                           scales: torch.Tensor, offsets: torch.Tensor,
                           expert_ids: torch.Tensor) -> torch.Tensor:
    """vLLM-style ESIMD w13 GEMV (silu(gate)*up fused), K-packed layout.

    weight  : [E, K/8, 2*moe_inter]  int32   (1 int32 = 8 K-nibbles same N)
    scales  : [E, K/G, 2*moe_inter]  bf16/fp16
    offsets : [E, K/G, 2*moe_inter]  bf16/fp16  (= -qzero * scale per group)
    """
    return torch.ops.awq_fused_xpu.awq_moe_gated_kp_esimd(
        x, weight, scales, offsets, expert_ids)


def awq_moe_reduce_kp_esimd(hidden: torch.Tensor, weight: torch.Tensor,
                            scales: torch.Tensor, offsets: torch.Tensor,
                            expert_ids: torch.Tensor,
                            topk_weights: torch.Tensor) -> torch.Tensor:
    """vLLM-style ESIMD w2 GEMV with topk-weighted reduce, K-packed layout.

    weight  : [E, K_in/8, hidden]    int32
    scales  : [E, K_in/G, hidden]    bf16/fp16
    offsets : [E, K_in/G, hidden]    bf16/fp16
    """
    return torch.ops.awq_fused_xpu.awq_moe_reduce_kp_esimd(
        hidden, weight, scales, offsets, expert_ids, topk_weights)


def moe_route_topk(router_logits: torch.Tensor, top_k: int,
                   renormalize: bool = True):
    """Fused softmax + top-K (+ optional renormalize) MoE router.

    Replaces the (softmax, topk, sum, div) chain that fires once per MoE
    layer at decode and dominates host launch overhead on small (1, E) tensors.

    router_logits : [B, E]   bf16/fp16/fp32   (E <= 256, top_k <= 16)
    Returns (topk_weights [B, K] fp32, topk_ids [B, K] int32).

    Math: softmax(logits) -> top-K -> renormalize(?). When renormalize=True
    the global softmax denominator cancels and we divide by the top-K sum
    of raw exp() values.
    """
    return torch.ops.awq_fused_xpu.moe_route_topk(
        router_logits, top_k, renormalize)


def rmsnorm_gated(x: torch.Tensor, z: torch.Tensor,
                  weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm + gated SiLU.

    Equivalent to RMSNormGated(x, z) with is_rms_norm=True,
    norm_before_gate=True, activation='swish', bias=None, group_size=None.

    x, z : [..., N]   bf16/fp16, last dim = N
    weight : [N]      bf16/fp16
    out    : same shape as x

    Replaces the Triton LayerNormFn path which has ~100us / call of
    Python/autograd overhead in addition to the kernel itself.
    """
    return torch.ops.awq_fused_xpu.rmsnorm_gated(x, z, weight, eps)
