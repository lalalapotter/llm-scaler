# custom-kernels-sglang

Unified SYCL/ESIMD kernel package used by the sglang XPU stack on Intel
Panther Lake (PTL) iGPU for Qwen3.5-35B-A3B-AWQ decode.

## Build

```
TORCH_XPU_ARCH_LIST=ptl-h pip install -e . --no-build-isolation --no-deps
```

`setup.py` defaults `TORCH_XPU_ARCH_LIST=ptl-h` if unset; override only
when cross-building.

## Layout

```
custom-kernels-sglang/
├── csrc/                # SYCL / ESIMD kernel sources (added by later commits)
├── python/              # Python bindings (added by later commits)
├── tests/               # Per-op correctness tests
├── build_ext.py         # Vendored PyTorch BuildExtension with SYCL fixes
├── setup.py
└── README.md
```

This commit lays down only the build scaffold; the actual kernel
extensions are added in subsequent commits.
