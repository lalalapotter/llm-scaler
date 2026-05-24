"""AWQ -> ESIMD layout repack helpers (for microbenchmark).

AWQ on-disk layout:
    qweight  : [E, K,   N/8]  int32, 8 nibbles per int32, pack order [0,4,1,5,2,6,3,7]
    qzeros   : [E, K/G, N/8]  int32, same pack order
    scales   : [E, K/G, N]    bf16/fp16

ESIMD-friendly layout (mimics moe_v2.cpp custom-esimd-kernels-vllm):
    weight   : [E, N, K/2]    uint8, low nibble = K_even, high nibble = K_odd
    scales   : [E, N, K/G]    bf16/fp16  (= original AWQ scale, just transposed)
    offsets  : [E, N, K/G]    bf16/fp16  (= -scale * qzero_value)

After this transformation, the ESIMD kernel computes:
    dequant(qw, qz, sc) = (qw - qz) * sc
                        = qw * sc - qz * sc
                        = qw * sc + offset
where offset = -qz * sc and qw is the unpacked nibble.
"""
import torch

_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack_awq_int32_to_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """[..., N/8] int32 -> [..., N] uint8 (nibbles unpacked in AWQ order).

    For each int32, the pack order is [0,4,1,5,2,6,3,7] in shift positions
    0,4,8,12,16,20,24,28. So the *output* nibble at column o is at shift
    `4 * AWQ_ORDER[o]`.
    """
    *prefix, packed_n = packed.shape
    out = torch.empty(*prefix, packed_n * 8, dtype=torch.uint8,
                      device=packed.device)
    for o in range(8):
        shift = 4 * _AWQ_ORDER[o]
        out[..., o::8] = ((packed >> shift) & 0xF).to(torch.uint8)
    return out


def repack_awq_to_esimd(qweight: torch.Tensor,
                        scales: torch.Tensor,
                        qzeros: torch.Tensor):
    """Convert AWQ MoE weights to ESIMD-friendly layout.

    Inputs:
        qweight : [E, K,   N/8] int32  (AWQ pack)
        scales  : [E, K/G, N]   bf16/fp16
        qzeros  : [E, K/G, N/8] int32  (AWQ pack)

    Returns (weight, scales_T, offsets):
        weight   : [E, N, K/2]  uint8  (low nibble K_even, high nibble K_odd)
        scales_T : [E, N, K/G]  bf16/fp16  (scales transposed to N-major)
        offsets  : [E, N, K/G]  bf16/fp16  (-scale * qzero, N-major)
    """
    assert qweight.dtype == torch.int32 and qzeros.dtype == torch.int32
    assert qweight.dim() == 3 and qzeros.dim() == 3 and scales.dim() == 3
    E, K, packed_n = qweight.shape
    N = packed_n * 8
    num_groups = qzeros.shape[1]
    G = K // num_groups
    assert scales.shape == (E, num_groups, N), \
        f"scales shape {scales.shape} != ({E},{num_groups},{N})"
    assert qzeros.shape == (E, num_groups, packed_n)

    # 1) Unpack qweight to [E, K, N] uint8 (each entry is a 4-bit value 0..15).
    qw_nibbles = _unpack_awq_int32_to_nibbles(qweight)  # [E, K, N]

    # 2) Pack along K into [E, K/2, N] uint8 (low nibble K_even, high K_odd).
    assert K % 2 == 0
    even = qw_nibbles[:, 0::2, :]  # [E, K/2, N]
    odd  = qw_nibbles[:, 1::2, :]  # [E, K/2, N]
    packed_K = (odd << 4) | even   # [E, K/2, N] uint8
    # 3) Transpose to [E, N, K/2].
    weight = packed_K.permute(0, 2, 1).contiguous()  # [E, N, K/2]

    # 4) Unpack qzeros [E, num_groups, N/8] -> [E, num_groups, N] uint8.
    qz_nibbles = _unpack_awq_int32_to_nibbles(qzeros)  # [E, num_groups, N]

    # 5) Compute offsets = -scale * qzero  in scales' dtype.
    sc_f32 = scales.to(torch.float32)
    qz_f32 = qz_nibbles.to(torch.float32)
    offsets_f32 = -sc_f32 * qz_f32     # [E, num_groups, N]
    # Transpose to N-major.
    scales_T = scales.permute(0, 2, 1).contiguous()      # [E, N, num_groups]
    offsets  = offsets_f32.permute(0, 2, 1).contiguous().to(scales.dtype)
    return weight, scales_T, offsets


def repack_awq_w13(qweight, scales, qzeros):
    """Convenience wrapper for w13 (gate+up). N axis = 2 * moe_inter."""
    return repack_awq_to_esimd(qweight, scales, qzeros)


def repack_awq_w2(qweight, scales, qzeros):
    """Convenience wrapper for w2 (down). K axis = moe_inter, N axis = hidden."""
    return repack_awq_to_esimd(qweight, scales, qzeros)
