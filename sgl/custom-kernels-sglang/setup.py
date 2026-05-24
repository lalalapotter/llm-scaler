"""custom-kernels-sglang — unified SYCL/ESIMD kernel package for sglang
on Intel PTL iGPU.

Combines two formerly separate trees:

  * awq_fused_xpu      In-house SYCL kernels (AWQ dequant+GEMV, MoE GEMV,
                       dense GEMV, MoE routing, RMSNorm-gated, ESIMD MoE).
  * custom_esimd_kernels_sglang  Selected ESIMD kernels mirrored from
                       xiangyuT/llm-scaler's custom-esimd-kernels-vllm,
                       trimmed to only the two extensions sglang's
                       Qwen3.5 path actually uses (renamed from *_vllm
                       since the upstream is now consumed by sglang).

Built extensions:
  - awq_fused_xpu._C
  - custom_esimd_kernels_sglang.custom_esimd_kernels  (esimd_qkv_split_norm_rope ...)
  - custom_esimd_kernels_sglang.eagle_ops             (page_attn_decode, chunk_gated_delta_rule_extend)
"""
import os
from pathlib import Path

# Default to PTL-only AOT codegen. The vendored build_ext.py reads
# TORCH_XPU_ARCH_LIST and otherwise hands every architecture in
# torch.xpu.get_arch_list() to icpx -fsycl-targets=spir64_gen.
# That triggers a VISA/IGC compiler crash in the device-link of the
# ESIMD MoE npacked kernel on at least one of the non-PTL targets,
# so unless the caller overrides this environment variable we restrict
# AOT to ptl-h. Use setdefault so external CI / cross-build flows can
# still override.
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "ptl-h")

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension

# Use the in-tree build_ext.py — a vendored PyTorch BuildExtension with
# fixes for SYCL/ESIMD compilation (e.g. multi-arch AOT, ESIMD doubleGRF
# flag wiring). Identical to torch.utils.cpp_extension.BuildExtension API.
from build_ext import BuildExtension

root = Path(__file__).parent.resolve()
torch_include = str(Path(torch.__file__).parent / "include")

# ---------------------------------------------------------------------------
# awq_fused_xpu — in-house SYCL kernels
# ---------------------------------------------------------------------------
awq_sources = [
    "csrc/awq_gemv.sycl",
    "csrc/awq_moe_gemv.sycl",
    "csrc/awq_moe_esimd.sycl",
    "csrc/awq_moe_esimd_npacked.sycl",
    "csrc/dense_gemv.sycl",
    "csrc/moe_route_topk.sycl",
    "csrc/rmsnorm_gated.sycl",
]

awq_ext = SyclExtension(
    name="awq_fused_xpu._C",
    sources=awq_sources,
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "sycl": ["-O3", "-fsycl", "-ffast-math",
                 "-fsycl-device-code-split=per_kernel",
                 f"-I{torch_include}"],
    },
    extra_link_args=["-Wl,-rpath,$ORIGIN/../torch/lib"],
    py_limited_api=False,
)

# ---------------------------------------------------------------------------
# custom_esimd_kernels_sglang.custom_esimd_kernels — ESIMD norm/gemv/qkv path
# ---------------------------------------------------------------------------
esimd_core_ext = SyclExtension(
    name="custom_esimd_kernels_sglang.custom_esimd_kernels",
    sources=[
        "csrc/xpu/esimd_kernel.sycl",
        "csrc/xpu/torch_extension.cc",
    ],
    include_dirs=[
        root / "include",
        root / "csrc",
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                 f"-I{torch_include}"],
    },
    extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
    py_limited_api=False,
)

# ---------------------------------------------------------------------------
# custom_esimd_kernels_sglang.eagle_ops — page_attn_decode + GDN extend
#
# Carries the 5.12-5.15 graph-friendly changes from the upstream xiangyuT
# tree:
#   - templated on storage dtype (fp16/bf16)
#   - optional external temp_p scratch buffer
#   - separate K/V tensors (no merged-layout gather)
# ---------------------------------------------------------------------------
eagle_ext = SyclExtension(
    name="custom_esimd_kernels_sglang.eagle_ops",
    sources=[
        "csrc/eagle/eagle.sycl",
    ],
    include_dirs=[
        root / "csrc" / "eagle",
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++20"],
        "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                 f"-I{torch_include}"],
    },
    extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
    py_limited_api=False,
)

setup(
    name="custom-kernels-sglang",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[awq_ext, esimd_core_ext, eagle_ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
