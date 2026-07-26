"""Tests for share_neighbors that need no oracle and have no keras-hexagdly
equivalent to share: structural checks against pytorch-only internals
(_materialize_shared_kernels), and Conv3d smoke tests (forward+backward,
finite output) across every mode/kernel_size/parity combination.

Everything oracle-driven -- the pure-Python first-principles share_neighbors
checks and the hand-verified Conv2d/Conv3d/MaxPool2d/MaxPool3d ground truth
-- lives in the shared `hexagdly-oracle` package
(https://github.com/YugnatD/hexagdly-oracle)'s tests/test_layers.py,
not here, so it is never duplicated (or silently allowed to diverge).
Cross-framework agreement with keras-hexagdly is checked in
hexagdly-oracle's tests/test_cross_framework.py.
"""

import pytest
import torch

import pytorch_hexagdly as ph


class TestParityCorrectness:
    """Verify that even/odd column centres read the same physical neighbourhood.

    For a correct weight-sharing map the two column parities must group the same
    *physical* hexes together. Because HexagDLy shifts odd columns down by half a
    cell, that physical invariance is achieved with an *identical* weight map for
    both parities — the row zig-zag is handled entirely by the slice/stride
    machinery (the same design kernel_size=1 uses). Hence no separate kernel_i_odd
    variant is created; asserting the maps are equal is asserting correctness.
    """

    @pytest.mark.parametrize("mode", ["ring", "diag", "sym"])
    def test_parity_maps_identical_for_n2(self, mode):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors=mode)
        conv._materialize_shared_kernels()
        # Even and odd maps coincide, so the odd variant is never materialized.
        assert not hasattr(conv, "kernel1_odd"), (
            f"{mode} k=2: even/odd maps are identical, no kernel1_odd expected"
        )


class TestConv3dShareNeighbors:
    """Smoke tests: Conv3d shares _HARDCODED_MAPS with Conv2d, so exercise every
    mode / kernel size on both even- and odd-width inputs (odd width triggers both
    column parities) to guard against crashes and NaNs in the 3d code path."""

    @pytest.mark.parametrize("mode", ["ring", "diag", "sym"])
    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("width", [16, 17])
    def test_forward_and_backward(self, mode, k, width):
        conv = ph.Conv3d(1, 1, kernel_size=(1, k), stride=1, bias=False, share_neighbors=mode)
        x = torch.randn(2, 1, 4, 13, width)
        out = conv(x)
        assert out.isfinite().all(), f"{mode} k={k} width={width}: non-finite output"
        assert out.shape[0] == 2 and out.shape[-1] == width
        out.sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).any(), (
            f"{mode} k={k} width={width}: no gradient flowed to shared_weights"
        )
