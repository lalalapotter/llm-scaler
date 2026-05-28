"""Shared test fixtures for custom-kernels-sglang.

Run with:
    cd custom-kernels-sglang
    pytest tests/

Tests skip cleanly if the XPU device or the in-tree extensions are not
available; they require an Intel GPU and `pip install -e .` of this
package to have completed.
"""
import pytest
import torch


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@pytest.fixture(scope="session")
def xpu_device() -> torch.device:
    if not _xpu_available():
        pytest.skip("torch.xpu not available")
    return torch.device("xpu")


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)
