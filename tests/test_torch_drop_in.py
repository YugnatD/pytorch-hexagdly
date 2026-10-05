"""Tests for using the hex layers as drop-in replacements for the torch.nn ones.

Conv2d, MaxPool2d and AvgPool2d accept ``padding`` like ``torch.nn.Conv2d`` /
``MaxPool2d`` / ``AvgPool2d``, so a model can pick either family of layers and
build them from one set of keyword arguments. Also covers grids no wider
than the stride, and saving models with torch.save. See CHANGELOG.md [0.3.0].

AvgPool2d's values are checked against the oracle mean in hexagdly-oracle's
tests/test_avgpool2d.py; the checks here need no oracle.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import pytorch_hexagdly as ph

H, W, C = 9, 8, 3


@pytest.fixture
def x():
    return torch.randn(2, C, H, W, generator=torch.Generator().manual_seed(0))


# ---- padding ------------------------------------------------------------------


POOLS = [ph.MaxPool2d, ph.AvgPool2d]


def _build(layer_cls, **kwargs):
    if layer_cls is ph.Conv2d:
        return layer_cls(C, 4, kernel_size=1, **kwargs)
    return layer_cls(kernel_size=1, **kwargs)


@pytest.mark.parametrize("layer_cls", [ph.Conv2d, *POOLS])
@pytest.mark.parametrize("padding", ["same", 0, 1, 2, (1, 1)])
def test_padding_is_accepted_and_changes_nothing(x, layer_cls, padding):
    torch.manual_seed(0)
    default = _build(layer_cls)
    torch.manual_seed(0)
    padded = _build(layer_cls, padding=padding)
    assert torch.equal(default(x), padded(x))


@pytest.mark.parametrize("layer_cls", [ph.Conv2d, *POOLS])
@pytest.mark.parametrize("padding", ["valid", "full", -1, True, 1.5, (1, -1)])
def test_padding_that_asks_for_another_grid_is_rejected(layer_cls, padding):
    with pytest.raises(ValueError, match="padding"):
        _build(layer_cls, padding=padding)


def test_positional_arguments_are_unchanged():
    """padding is appended after the existing arguments, so positional calls
    written for earlier versions keep their meaning."""
    conv = ph.Conv2d(2, 3, 2, 2, False)
    assert (conv.hexbase_size, conv.hexbase_stride, conv.bias) == (2, 2, False)
    pool = ph.MaxPool2d(2, 3)
    assert (pool.hexbase_size, pool.hexbase_stride) == (2, 3)


@pytest.mark.parametrize("stride", [1, 2])
def test_same_keyword_arguments_build_either_family(x, stride):
    """The keyword arguments a torch.nn model passes to its conv and pool
    layers build the hex layers too, as a CTLearn-style backbone does when it
    switches layer classes."""
    for conv_cls, max_cls, avg_cls in [
        (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d),
        (ph.Conv2d, ph.MaxPool2d, ph.AvgPool2d),
    ]:
        model = nn.Sequential(
            conv_cls(C, 4, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(),
            max_cls(kernel_size=2, stride=stride, padding=0),
            conv_cls(4, 4, kernel_size=1, stride=1, bias=False),
            avg_cls(kernel_size=2, stride=1),
        )
        out = model(x)
        assert out.shape[:2] == (2, 4)
        assert torch.isfinite(out).all()


# ---- grids no wider than the stride ---------------------------------------------


@pytest.mark.parametrize(
    "make_layer",
    [
        lambda k, s: ph.Conv2d(C, 4, kernel_size=k, stride=s),
        lambda k, s: ph.MaxPool2d(k, s),
        lambda k, s: ph.AvgPool2d(k, s),
    ],
    ids=["conv2d", "maxpool2d", "avgpool2d"],
)
@pytest.mark.parametrize("width, stride", [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)])
@pytest.mark.parametrize("kernel_size", [1, 2])
def test_grid_no_wider_than_the_stride(make_layer, width, stride, kernel_size):
    """With width <= stride the even-column group of the output is empty.
    Computing it anyway crashed, as it does in upstream HexagDLy. The single
    output column must match the first output column of the same grid widened
    with zero columns, which takes the ordinary path. (AvgPool2d divides by
    the in-grid count, which the widening changes, so it compares with a
    widened grid of the same width-independent mean: a constant.)"""
    g = torch.Generator().manual_seed(1)
    narrow = torch.randn(2, C, 7, width, generator=g)
    layer = make_layer(kernel_size, stride)
    narrow_out = layer(narrow)
    assert narrow_out.shape[-1] == 1
    if isinstance(layer, ph.AvgPool2d):
        const = layer(torch.full((2, C, 7, width), 2.5))
        np.testing.assert_allclose(const.numpy(), 2.5, rtol=1e-6)
        return
    wide = torch.cat([narrow, torch.zeros(2, C, 7, 2 * stride + 1)], dim=-1)
    torch.testing.assert_close(narrow_out[..., 0], layer(wide)[..., 0], atol=1e-5, rtol=0)


@pytest.mark.parametrize("width, stride", [(1, 1), (2, 2)])
def test_conv3d_grid_no_wider_than_the_stride(width, stride):
    x3 = torch.randn(2, C, 3, 7, width)
    out = ph.Conv3d(C, 4, kernel_size=(1, 2), stride=(1, stride))(x3)
    assert out.shape[-1] == 1
    assert torch.isfinite(out).all()


# ---- AvgPool2d ----------------------------------------------------------------


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_avgpool_of_a_constant_is_that_constant_everywhere(kernel_size, stride):
    """Holds at the grid border only if the divisor counts exactly the in-grid
    cells of each window."""
    out = ph.AvgPool2d(kernel_size, stride)(torch.full((1, 2, H, W), 3.5))
    np.testing.assert_allclose(out.numpy(), 3.5, rtol=1e-6)


def _ones_custom_kernel(kernel_size, stride):
    """Conv2d_CustomKernel summing a single channel over the hex window -- an
    independent path to the same window sums (a dense conv per sub-kernel
    rather than the depthwise sum AvgPool2d uses)."""
    sub_kernels = [
        np.ones((1, 1, 1 + 2 * kernel_size - i, 1 if i == 0 else 2), "float32")
        for i in range(kernel_size + 1)
    ]
    return ph.Conv2d_CustomKernel(sub_kernels=sub_kernels, stride=stride)


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_avgpool_matches_window_sum_over_window_count(x, kernel_size, stride):
    window_sum = _ones_custom_kernel(kernel_size, stride)
    count = window_sum(torch.ones(1, 1, H, W))
    expected = torch.cat([window_sum(x[:, c : c + 1]) / count for c in range(C)], dim=1)
    torch.testing.assert_close(ph.AvgPool2d(kernel_size, stride)(x), expected)


def test_avgpool_interior_cell_averages_its_seven_cells():
    """Hand check: a unit impulse is shared by the 7 windows it belongs to."""
    impulse = torch.zeros(1, 1, H, W)
    impulse[0, 0, 4, 4] = 1.0
    out = ph.AvgPool2d(1)(impulse)[0, 0]
    assert torch.isclose(out[4, 4], torch.tensor(1 / 7))
    assert int(torch.count_nonzero(out)) == 7
    torch.testing.assert_close(out[out != 0], torch.full((7,), 1 / 7))


@pytest.mark.parametrize("stride", [1, 2])
def test_avgpool_output_grid_matches_maxpool2d(x, stride):
    """Same window walk as MaxPool2d, so the same output grid."""
    assert ph.AvgPool2d(2, stride)(x).shape == ph.MaxPool2d(2, stride)(x).shape


def test_avgpool_passes_gradients_and_keeps_dtype():
    x = torch.randn(2, C, H, W, dtype=torch.float64, requires_grad=True)
    out = ph.AvgPool2d(2, 2)(x)
    assert out.dtype == torch.float64
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Every input cell is in at least one window at stride 1, so all of it
    # receives gradient there.
    x.grad = None
    ph.AvgPool2d(1)(x).sum().backward()
    assert (x.grad > 0).all()


# ---- saving -------------------------------------------------------------------


def _hex_model():
    return nn.Sequential(
        ph.Conv2d(C, 4, kernel_size=2, stride=1, padding=1),
        nn.ReLU(),
        ph.MaxPool2d(kernel_size=1, stride=2, padding=0),
        ph.Conv2d(4, 4, kernel_size=1, stride=2, bias=False),
        ph.AvgPool2d(kernel_size=1, stride=1),
    )


@pytest.mark.parametrize(
    "make_layer",
    [
        lambda: ph.Conv2d(C, 4, 1),
        lambda: ph.Conv2d(C, 4, 1, share_neighbors="ring"),
        lambda: ph.Conv3d(C, 4, 1),
        lambda: ph.MaxPool2d(1),
        lambda: ph.MaxPool3d(1),
        lambda: ph.AvgPool2d(1),
    ],
    ids=["conv2d", "conv2d_shared", "conv3d", "maxpool2d", "maxpool3d", "avgpool2d"],
)
def test_every_layer_can_be_pickled(make_layer, tmp_path):
    """torch.save(model) pickles every module. MaxPool2d/3d used to store
    F.max_pool2d/3d directly, which can't be pickled."""
    path = tmp_path / "layer.pt"
    torch.save(make_layer(), path)
    torch.load(path, weights_only=False)


def test_full_model_save_load_round_trip(x, tmp_path):
    model = _hex_model()
    path = tmp_path / "model.pt"
    torch.save(model, path)
    restored = torch.load(path, weights_only=False)
    assert torch.equal(restored(x), model(x))


def test_state_dict_round_trip(x):
    model = _hex_model()
    restored = _hex_model()
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored(x), model(x))
