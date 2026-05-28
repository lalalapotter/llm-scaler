"""Correctness for the eagle_ops paged-attention decode kernel.

We use a small head_dim=256, gqa_ratio=4 case (the only configuration
this kernel supports) and compare against a python reference that does
naive scaled-dot-product attention over the gathered K/V cache.

Also covers:
  - eagle_page_attn_decode_temp_size returns a positive integer (it's a
    pure-python sizing helper used to pre-allocate the temp_p scratch
    buffer for graph-stable replays).
"""
import math

import pytest
import torch

esimd = pytest.importorskip("custom_esimd_kernels_sglang")
from custom_esimd_kernels_sglang.ops import (  # noqa: E402
    eagle_page_attn_decode,
    eagle_page_attn_decode_temp_size,
)

HEAD_DIM = 256
GQA_RATIO = 4


def _reference_decode(query, key_cache, value_cache, block_table, seq_lens,
                      softmax_scale=None):
    """Naive paged attention decode reference.

    query        : [B, H_q, D] fp
    key_cache    : [num_pages, page_size, H_kv, D] fp
    value_cache  : [num_pages, page_size, H_kv, D] fp
    block_table  : [B, max_blocks] int32
    seq_lens     : [B] int32
    """
    B, H_q, D = query.shape
    num_pages, P, H_kv, _ = key_cache.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(query)
    for b in range(B):
        slen = int(seq_lens[b].item())
        n_blocks = (slen + P - 1) // P
        # Gather K/V for this batch row.
        keys, vals = [], []
        for blk in range(n_blocks):
            page_id = int(block_table[b, blk].item())
            take = P if blk < n_blocks - 1 else (slen - blk * P)
            keys.append(key_cache[page_id, :take])
            vals.append(value_cache[page_id, :take])
        K = torch.cat(keys, dim=0)   # [slen, H_kv, D]
        V = torch.cat(vals, dim=0)
        for h in range(H_q):
            kv_h = h // (H_q // H_kv)
            q = query[b, h]                         # [D]
            k = K[:, kv_h, :]                       # [slen, D]
            v = V[:, kv_h, :]
            attn = (k @ q) * softmax_scale          # [slen]
            attn = attn.float().softmax(dim=-1).to(q.dtype)
            out[b, h] = attn @ v
    return out


def test_eagle_page_attn_decode_temp_size_positive_int():
    sz = eagle_page_attn_decode_temp_size(
        batches=1, num_q_heads=8, num_kv_heads=2,
        head_dim=HEAD_DIM, max_seq_len=8192,
    )
    assert isinstance(sz, int)
    assert sz > 0


@pytest.mark.parametrize(
    "B,H_kv,seq_len,page_size,dtype",
    [
        pytest.param(1, 2, 256, 64, torch.float16, id="B1_H2_S256_P64_fp16"),
        pytest.param(1, 2, 256, 64, torch.bfloat16, id="B1_H2_S256_P64_bf16"),
        pytest.param(2, 2, 384, 64, torch.float16, id="B2_H2_S384_P64_fp16"),
    ],
)
def test_eagle_page_attn_decode_matches_naive_attention(
    B, H_kv, seq_len, page_size, dtype, xpu_device,
):
    H_q = H_kv * GQA_RATIO

    n_blocks_per_seq = (seq_len + page_size - 1) // page_size
    num_pages = max(B * n_blocks_per_seq, 8)

    query = torch.randn(B, H_q, HEAD_DIM, dtype=dtype, device=xpu_device) * 0.1
    key_cache = torch.randn(num_pages, page_size, H_kv, HEAD_DIM,
                            dtype=dtype, device=xpu_device) * 0.1
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros((B, n_blocks_per_seq), dtype=torch.int32,
                              device=xpu_device)
    page_pool = torch.randperm(num_pages, device=xpu_device)
    for b in range(B):
        block_table[b] = page_pool[b * n_blocks_per_seq:
                                   (b + 1) * n_blocks_per_seq].to(torch.int32)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=xpu_device)
    out = torch.empty(B, H_q, HEAD_DIM, dtype=dtype, device=xpu_device)

    temp_size = eagle_page_attn_decode_temp_size(
        B, H_q, H_kv, HEAD_DIM, max_seq_len=seq_len,
    )
    temp_p = torch.zeros((temp_size,), dtype=torch.float32, device=xpu_device)

    eagle_page_attn_decode(
        query, key_cache, value_cache, block_table, seq_lens, out,
        max_query_len=1, max_seq_len=seq_len, temp_p=temp_p,
    )

    ref = _reference_decode(query, key_cache, value_cache, block_table,
                            seq_lens)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
