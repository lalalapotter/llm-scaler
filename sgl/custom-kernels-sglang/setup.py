"""custom-kernels-sglang — unified SYCL/ESIMD kernel package for sglang
on Intel PTL iGPU.

This commit lays down only the package scaffold (build infrastructure +
empty extension list). Subsequent commits add:

  - awq_fused_xpu (in-house SYCL kernels)
  - custom_esimd_kernels_sglang.custom_esimd_kernels (mirrored ESIMD)
  - custom_esimd_kernels_sglang.eagle_ops (page_attn_decode + extend)
"""
import os

# Default to PTL-only AOT codegen. The vendored build_ext.py reads
# TORCH_XPU_ARCH_LIST and otherwise hands every architecture in
# torch.xpu.get_arch_list() to icpx -fsycl-targets=spir64_gen.
# That triggers a VISA/IGC compiler crash in the device-link of the
# ESIMD MoE npacked kernel on at least one of the non-PTL targets,
# so unless the caller overrides this environment variable we restrict
# AOT to ptl-h. Use setdefault so external CI / cross-build flows can
# still override.
os.environ.setdefault("TORCH_XPU_ARCH_LIST", "ptl-h")

from setuptools import find_packages, setup

# Use the in-tree build_ext.py — a vendored PyTorch BuildExtension with
# fixes for SYCL/ESIMD compilation (e.g. multi-arch AOT, ESIMD doubleGRF
# flag wiring). Identical to torch.utils.cpp_extension.BuildExtension API.
from build_ext import BuildExtension

setup(
    name="custom-kernels-sglang",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
