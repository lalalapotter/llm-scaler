# custom-kernels-sglang

Unified SYCL/ESIMD kernel package used by the sglang XPU stack on Intel
Panther Lake (PTL) iGPU for Qwen3.5-35B-A3B-AWQ decode.

## Layout

```
custom-kernels-sglang/
├── csrc/
│   ├── awq_gemv.sycl                 # Dense AWQ int4 fused dequant + GEMV
│   ├── awq_moe_gemv.sycl             # MoE AWQ int4 fused dequant + GEMV
│   ├── awq_moe_esimd.sycl            # ESIMD MoE AWQ (kpacked variant)
│   ├── awq_moe_esimd_npacked.sycl    # ESIMD MoE AWQ (npacked variant)
│   ├── dense_gemv.sycl               # bf16 dense GEMV (M=1, templated)
│   ├── moe_route_topk.sycl           # Fused softmax + top-K + renormalize
│   ├── rmsnorm_gated.sycl            # Fused RMSNorm + sigmoid gate
│   ├── eagle/                        # ESIMD eagle ops (page_attn_decode etc.)
│   │   ├── eagle.sycl
│   │   ├── eagle.kernels.{fp16,bf16}.h
│   │   ├── extend.kernels.{fp16,bf16}.h
│   │   └── page.attn.h
│   └── xpu/                          # ESIMD norm/gemv/qkv kernel sources
│       ├── esimd_kernel.sycl         #   esimd_qkv_split_norm_rope etc.
│       ├── torch_extension.cc
│       └── esimd_kernels/*.h         #   12 helper headers
├── include/kernel_ops.h
├── python/
│   ├── awq_fused_xpu/                # Python bindings for awq kernels
│   │   ├── __init__.py
│   │   ├── repack.py
│   │   └── repack_kpacked.py
│   └── custom_esimd_kernels_sglang/  # Python bindings for ESIMD kernels
│       ├── __init__.py               #   PTL-tolerant best-effort loader
│       └── ops.py                    #   wraps torch.ops.custom_esimd_kernels_sglang.*
│                                     #   and torch.ops.eagle_ops.*
├── tests/                             # Per-op correctness tests (pytest)
├── build_ext.py                       # Vendored PyTorch BuildExtension w/ SYCL fixes
├── setup.py
└── README.md
```

## Built extensions

| Extension | Sources | Provides |
|---|---|---|
| `awq_fused_xpu._C` | All `csrc/*.sycl` (top level) | `awq_gemv_fused`, `awq_moe_gated_gemv`, `awq_moe_reduce_gemv`, `awq_moe_gated_kp_esimd`, `awq_moe_reduce_kp_esimd`, `awq_moe_gated_np_esimd`, `awq_moe_reduce_np_esimd`, `dense_gemv`, `moe_route_topk`, `rmsnorm_gated`, ... |
| `custom_esimd_kernels_sglang.custom_esimd_kernels` | `csrc/xpu/esimd_kernel.sycl` + `torch_extension.cc` | `esimd_qkv_split_norm_rope` (and several siblings unused on this path) |
| `custom_esimd_kernels_sglang.eagle_ops` | `csrc/eagle/eagle.sycl` | `page_attn_decode` (8-arg, external `temp_p`), `chunk_gated_delta_rule_extend` |

## Where the ESIMD pieces come from

`csrc/eagle/`, `csrc/xpu/`, `include/`, and `python/custom_esimd_kernels_sglang/`
were originally maintained as `custom-esimd-kernels-vllm` in the
[xiangyuT/llm-scaler dev/sgl_qwen3.5_0512](https://github.com/xiangyuT/llm-scaler/tree/dev/sgl_qwen3.5_0512)
branch. They have been:

* renamed to `custom_esimd_kernels_sglang` (the namespace, the Python
  package, and the matching `import` sites in sglang) since the consumer
  is now sglang, not vllm,
* trimmed to just the two extensions sglang's Qwen3.5 path uses (the
  upstream tree also ships extensions that depend on XMX/DPAS — BMG only —
  or implement MoE-batch helpers that our path does not touch).

The eagle_ops carries the 5.12-5.15 graph-friendly changes that let
`SGLANG_XPU_ENABLE_GRAPH=1` capture+replay decode with stable `data_ptr`:

| Commit | Subject |
|---|---|
| `188c0c3` | template `page_attn_decode` on storage dtype (fp16/bf16) |
| `6790951` | optional external `temp_p` buffer (graph-stable scratch) |
| `86b1739` | take separate K/V tensors, drop merged-layout gather |
| `cca9d9b` | PTL/XeLPG-tolerant package loading |
