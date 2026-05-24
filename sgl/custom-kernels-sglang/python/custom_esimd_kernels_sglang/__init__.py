"""custom_esimd_kernels_sglang — PTL iGPU (XeLPG) tolerant __init__.

Some compiled extensions require XMX (DPAS) intrinsics which are not
available on XeLPG (Panther Lake iGPU). On such hosts we build only a
subset of the extensions; the remaining ones are absent. Everything Qwen3.5
dense production needs (``esimd_qkv_split_norm_rope`` from
``custom_esimd_kernels`` and ``eagle_page_attn_decode`` from ``eagle_ops``)
lives in DPAS-free extensions.

Each ext import and each op re-export is wrapped in a best-effort block so
a missing extension does not prevent the package from loading. Missing ops
surface as AttributeError only at call time.
"""
import logging

import torch

_log = logging.getLogger(__name__)
_MISSING_EXTS = []


def _try_import_ext(name):
    """Import a compiled extension and also re-dlopen it with RTLD_GLOBAL.

    Without RTLD_GLOBAL the TORCH_LIBRARY static initializers in some of
    these .so files do not register their ops into torch.ops.<namespace>.
    Confirmed via nm: symbols TORCH_LIBRARY_FRAGMENT_static_init_<ns> exist
    but are not invoked under the default Python ext loader's RTLD_LOCAL.
    """
    import ctypes
    import os
    try:
        mod = __import__(f"custom_esimd_kernels_sglang.{name}")
        sub = getattr(mod, name)
        so_path = getattr(sub, "__file__", None)
        if so_path and os.path.exists(so_path):
            # idempotent: CDLL caches by path internally
            ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        return True
    except ImportError as e:
        _MISSING_EXTS.append((name, str(e)))
        return False


# Compiled extension modules (each registers ops into torch.ops).
# This trimmed PTL-sglang build only ships the two extensions sglang
# actually consumes; the rest live in the upstream xiangyuT/llm-scaler tree.
_try_import_ext("custom_esimd_kernels")        # esimd_qkv_split_norm_rope etc.
_try_import_ext("eagle_ops")                   # page_attn_decode + chunk_gated_delta_rule_extend

if _MISSING_EXTS:
    _log.info(
        "custom_esimd_kernels_sglang: extensions unavailable on this host: %s. "
        "Ops from those extensions will raise AttributeError at call time.",
        ", ".join(n for n, _ in _MISSING_EXTS),
    )

# Python wrappers. ops.py does not eagerly call any kernel; each function
# only touches torch.ops.custom_esimd_kernels_sglang.<name> when invoked.
# Re-export them individually so a missing op (e.g. an MoE helper that the
# skipped extension would have registered) doesn't block loading the rest.
from custom_esimd_kernels_sglang import ops as _ops_mod  # noqa: E402

_EXPORTS = [
    # custom_esimd_kernels (esimd_kernel.sycl) — used by sglang qwen3_5
    "esimd_qkv_split_norm_rope",
    # eagle_ops (eagle.sycl) — used by sglang xpu_backend + gdn_triton
    "eagle_page_attn_decode",
    "eagle_page_attn_decode_temp_size",
]

_MISSING_OPS = []
for _name in _EXPORTS:
    try:
        globals()[_name] = getattr(_ops_mod, _name)
    except AttributeError:
        _MISSING_OPS.append(_name)

if _MISSING_OPS:
    _log.info(
        "custom_esimd_kernels_sglang: %d python ops not available on this host "
        "(e.g. %s%s)",
        len(_MISSING_OPS),
        ", ".join(_MISSING_OPS[:3]),
        " ..." if len(_MISSING_OPS) > 3 else "",
    )
