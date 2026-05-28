"""Correctness for esimd_qkv_split_norm_rope — the fused QKV-split +
per-head RMSNorm + RoPE op invoked once per attention layer.

Reference is a Python implementation of the same math (split, RMSNorm
with the Qwen3 weight+1.0 convention, RoPE rotation). Hard-coded to
fp16 and head_dim=256 — that's all the ESIMD kernel supports.
"""
import pytest
import torch
import torch.nn.functional as F

esimd = pytest.importorskip("custom_esimd_kernels_sglang")
from custom_esimd_kernels_sglang import esimd_qkv_split_norm_rope  # noqa: E402

HEAD_DIM = 256


def _build_cos_sin_cache(max_pos, rotary_dim, device, base=10000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
    )
    pos = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # [max_pos, rotary_dim/2]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [max_pos, rotary_dim]
    return cache.to(torch.float16)


def _reference(qkv_state, norm_wq, norm_wk, positions, cos_sin,
               q_heads, kv_heads, attn_output_gate, rotary_dim):
    """Pure-python equivalent of the ESIMD kernel."""
    n_tokens = qkv_state.shape[0]
    head_dim = HEAD_DIM
    eps = 1e-6

    # Layout: [Q (q_heads * H), [Q-gate (q_heads * H)], K (kv_heads * H),
    #          V (kv_heads * H)]
    cur = 0
    q = qkv_state[:, cur:cur + q_heads * head_dim]
    cur += q_heads * head_dim
    if attn_output_gate:
        gate = qkv_state[:, cur:cur + q_heads * head_dim]
        cur += q_heads * head_dim
    else:
        gate = None
    k = qkv_state[:, cur:cur + kv_heads * head_dim]
    cur += kv_heads * head_dim
    v = qkv_state[:, cur:cur + kv_heads * head_dim]

    def rms(t, w):
        # Qwen3 convention: weight is centred at 0, formula uses (weight + 1)
        # per the kernel docstring. Reshape to per-head, normalise inner dim.
        t = t.reshape(n_tokens, -1, head_dim).float()
        var = t.pow(2).mean(dim=-1, keepdim=True)
        n = t * torch.rsqrt(var + eps)
        n = n * (w.float() + 1.0)
        return n.to(qkv_state.dtype)

    q = rms(q, norm_wq)
    k = rms(k, norm_wk)

    # RoPE on the first rotary_dim dims.
    cos = cos_sin[positions.long(), :rotary_dim // 2]
    sin = cos_sin[positions.long(), rotary_dim // 2:rotary_dim]

    def rope(t):  # t: [n_tokens, heads, head_dim] fp16
        t = t.float()
        a = t[..., :rotary_dim // 2]
        b = t[..., rotary_dim // 2:rotary_dim]
        cf = cos.float()[:, None, :]   # [n_tokens, 1, rotary_dim/2]
        sf = sin.float()[:, None, :]
        a_new = a * cf - b * sf
        b_new = a * sf + b * cf
        rotated = torch.cat([a_new, b_new], dim=-1)
        return torch.cat([rotated, t[..., rotary_dim:]], dim=-1).to(qkv_state.dtype)

    q_rot = rope(q).reshape(n_tokens, -1)
    k_rot = rope(k).reshape(n_tokens, -1)
    v_out = v
    gate_out = gate if gate is not None else torch.zeros_like(q_rot)
    return q_rot, gate_out, k_rot, v_out


@pytest.mark.parametrize(
    "n_tokens,q_heads,kv_heads,attn_output_gate",
    [
        pytest.param(1, 8, 2, False, id="T1_qH8_kvH2_nogate"),
        pytest.param(1, 8, 2, True,  id="T1_qH8_kvH2_gate"),
    ],
)
def test_esimd_qkv_split_norm_rope_matches_python_reference(
    n_tokens, q_heads, kv_heads, attn_output_gate, xpu_device,
):
    rotary_dim = 256
    max_pos = 1024

    hidden_q = q_heads * HEAD_DIM
    hidden_kv = kv_heads * HEAD_DIM
    hidden_total = hidden_q + (hidden_q if attn_output_gate else 0) + 2 * hidden_kv

    qkv_state = (
        torch.randn(n_tokens, hidden_total, dtype=torch.float16,
                    device=xpu_device) * 0.5
    )
    q_out = torch.empty(n_tokens, hidden_q, dtype=torch.float16,
                        device=xpu_device)
    gate_out = torch.empty(n_tokens, hidden_q, dtype=torch.float16,
                           device=xpu_device)
    k_out = torch.empty(n_tokens, hidden_kv, dtype=torch.float16,
                        device=xpu_device)
    v_out = torch.empty(n_tokens, hidden_kv, dtype=torch.float16,
                        device=xpu_device)

    norm_wq = torch.randn(HEAD_DIM, dtype=torch.float16, device=xpu_device) * 0.1
    norm_wk = torch.randn(HEAD_DIM, dtype=torch.float16, device=xpu_device) * 0.1
    positions = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int32,
                              device=xpu_device)
    cos_sin = _build_cos_sin_cache(max_pos, rotary_dim, xpu_device)

    esimd_qkv_split_norm_rope(
        qkv_state, q_out, gate_out, k_out, v_out,
        norm_wq, norm_wk, positions,
        q_heads, kv_heads, attn_output_gate, rotary_dim, cos_sin,
    )

    q_ref, gate_ref, k_ref, v_ref = _reference(
        qkv_state, norm_wq, norm_wk, positions, cos_sin,
        q_heads, kv_heads, attn_output_gate, rotary_dim,
    )

    torch.testing.assert_close(q_out, q_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_out, v_ref, rtol=5e-2, atol=5e-2)
    if attn_output_gate:
        torch.testing.assert_close(gate_out, gate_ref, rtol=5e-2, atol=5e-2)
