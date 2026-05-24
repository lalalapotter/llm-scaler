"""AWQ -> IPEX-style K-packed layout for vLLM-style ESIMD SLM kernels.

Source AWQ on-disk:
    qweight : [E, K,   N/8]  int32
                shift positions for output column o (within an 8-pack):
                pack_order[8] = {0, 4, 1, 5, 2, 6, 3, 7}
                bit_pos = 4 * pack_order[o]
    qzeros  : [E, K/G, N/8]  int32   (same packing)
    scales  : [E, K/G, N]    bf16

Target K-packed (row=K_packed, col=N), AWQ asymmetric semantics:
    weight_kp : [E, K/8, N]  int32
                each int32 contains 8 nibbles for K=[k0..k0+7] at the SAME N column
                shift positions: nibble for K=k0+i is at bits [4*i, 4*i+4)  (no pack-order shuffle)
    scales_kp : [E, K/G, N]  bf16  (= original AWQ scales unchanged)
    offset_kp : [E, K/G, N]  bf16  (= -qzero_nibble * scale, per-group)

Dequant formula in the kernel:
    nibble = (packed_int32 >> (4*i)) & 0xF
    dequant_value = nibble * scale[g] + offset[g]
                  = nibble * scale - qzero * scale
                  == (nibble - qzero) * scale  (same as AWQ formula)
"""
import torch

_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack_awq_int32_to_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """[..., N/8] int32 -> [..., N] uint8, AWQ pack order honored.

    Output nibble at column o comes from shift `4 * pack_order[o]`.
    """
    *prefix, packed_n = packed.shape
    out = torch.empty(*prefix, packed_n * 8, dtype=torch.uint8,
                      device=packed.device)
    for o in range(8):
        shift = 4 * _AWQ_ORDER[o]
        out[..., o::8] = ((packed >> shift) & 0xF).to(torch.uint8)
    return out


def _pack_kmajor_nibbles_to_int32(nibbles: torch.Tensor) -> torch.Tensor:
    """[..., K, N] uint8 -> [..., K/8, N] int32 with k=k0..k0+7 packed
    sequentially (k0 at bits[0:4], k0+1 at [4:8], ..., k0+7 at [28:32]).
    """
    assert nibbles.dtype == torch.uint8
    *prefix, K, N = nibbles.shape
    assert K % 8 == 0
    K8 = K // 8
    # Reshape to [..., K8, 8, N] then accumulate shift-OR per inner-K group.
    grouped = nibbles.view(*prefix, K8, 8, N).to(torch.int64)  # int64 for safe shift
    out = torch.zeros(*prefix, K8, N, dtype=torch.int64, device=nibbles.device)
    for i in range(8):
        out |= (grouped[..., i, :] & 0xF) << (4 * i)
    # Now mask back to int32 range. The high bit may set the sign for nibble=15
    # at position i=7, which becomes -1 in int32. We need to keep the bit
    # pattern: cast through uint32 then reinterpret.
    return out.to(torch.int32)  # truncates to lower 32 bits


def repack_awq_to_kpacked(qweight: torch.Tensor,
                           scales: torch.Tensor,
                           qzeros: torch.Tensor):
    """Convert AWQ MoE weights to K-packed layout used by vLLM-style ESIMD.

    Inputs:
        qweight : [E, K,   N/8] int32  (AWQ pack)
        scales  : [E, K/G, N]   bf16/fp16
        qzeros  : [E, K/G, N/8] int32  (AWQ pack)

    Returns (weight_kp, scales_kp, offsets_kp):
        weight_kp  : [E, K/8, N] int32   (1 int32 = 8 K-nibbles same N col)
        scales_kp  : [E, K/G, N] bf16/fp16   same as input scales
        offsets_kp : [E, K/G, N] bf16/fp16   = -qzero_nibble * scale per group
    """
    assert qweight.dtype == torch.int32 and qzeros.dtype == torch.int32
    assert qweight.dim() == 3 and qzeros.dim() == 3 and scales.dim() == 3
    E, K, packed_n = qweight.shape
    N = packed_n * 8
    num_groups = qzeros.shape[1]
    G = K // num_groups
    assert scales.shape == (E, num_groups, N)
    assert qzeros.shape == (E, num_groups, packed_n)
    assert K % 8 == 0

    # 1) Unpack qweight into [E, K, N] uint8 (each in [0,15]).
    qw_nibbles = _unpack_awq_int32_to_nibbles(qweight)  # [E, K, N]
    # 2) Re-pack along K (8 K rows -> 1 int32) for each (E, N) pair.
    weight_kp = _pack_kmajor_nibbles_to_int32(qw_nibbles)  # [E, K/8, N]

    # 3) Unpack qzeros into [E, K/G, N] uint8.
    qz_nibbles = _unpack_awq_int32_to_nibbles(qzeros)  # [E, K/G, N]

    # 4) Compute offsets = -qzero * scale (in scale dtype).
    sc_f32 = scales.to(torch.float32)
    qz_f32 = qz_nibbles.to(torch.float32)
    offsets_f32 = -qz_f32 * sc_f32                # [E, K/G, N]
    offsets_kp = offsets_f32.to(scales.dtype).contiguous()
    scales_kp = scales.contiguous()
    return weight_kp.contiguous(), scales_kp, offsets_kp


def repack_awq_w13_kpacked(qweight, scales, qzeros):
    """w13 (gate+up): N axis = 2 * moe_inter, K axis = hidden_size."""
    return repack_awq_to_kpacked(qweight, scales, qzeros)


def repack_awq_w2_kpacked(qweight, scales, qzeros):
    """w2 (down): K axis = moe_inter, N axis = hidden_size."""
    return repack_awq_to_kpacked(qweight, scales, qzeros)
