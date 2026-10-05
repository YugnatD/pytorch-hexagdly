# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-10-05

### Added

- **`AvgPool2d`**: hexagonal average pooling, with the same window and the
  same `kernel_size` / `stride` arguments as `MaxPool2d`. Each output cell is
  the mean over the cells of its window that lie inside the grid: positions
  past the border count in neither the sum nor the divisor (like
  `nn.AvgPool2d(..., count_include_pad=False)`), so a constant image stays
  constant up to the border. Same semantics as `keras_hexagdly.AvgPool2d`
  (keras-hexagdly 0.5.0), and checked against the same oracle mean in
  `hexagdly-oracle`'s `tests/test_layers.py` for both ports.
- **`padding` argument on `Conv2d`, `MaxPool2d` and `AvgPool2d`**, so calls
  written for `nn.Conv2d` / `nn.MaxPool2d` / `nn.AvgPool2d` (for example
  `padding=kernel_size // 2`) build the hex layers unchanged. The hex grid is
  always padded so that every cell is a window centre: `"same"` (the
  default) and non-negative integer paddings are accepted and change
  nothing, and `"valid"` raises a `ValueError` instead of being silently
  ignored. Appended after the existing arguments, so positional calls keep
  their meaning.
- **`kernel_size` and `stride` are checked when a layer is built.** A
  non-positive or non-integer value (or a `Conv3d`/`MaxPool3d` pair that
  isn't a 2-tuple) now raises a `ValueError` right away, instead of being
  accepted and failing at the first forward pass with an unrelated-looking
  `ZeroDivisionError` or `TypeError`. Same checks as keras_hexagdly.
- **`share_neighbors=True`** now means `"ring"`, as in keras_hexagdly.

### Fixed

- **`Conv2d`, `Conv3d`, `MaxPool2d` (and `MaxPool3d`) crashed on grids no
  wider than the stride**, and on single-column grids at stride 1. The
  even-column group of the output is empty there, and computing it anyway
  ran a convolution or pooling on a zero-width input. It is now skipped in
  both stride paths. Inherited from upstream HexagDLy, which fails the same
  way; the same fix as keras-hexagdly 0.5.0.
- **`torch.save(model)` failed on any model containing `MaxPool2d` or
  `MaxPool3d`** (`AttributeError: Can't pickle local object
  'boolean_dispatch.<locals>.fn'`). The layers stored `F.max_pool2d/3d`
  directly, and those are locally defined dispatch functions that can't be
  pickled. They now go through a module-level wrapper. Saving a
  `state_dict` was never affected.
- **A strided layer reused on another grid size crashed or misindexed.**
  With `stride > 1`, the slices and paddings of each sub-kernel were
  computed on the first call only and reused for every later input, whatever
  its height and width. They are now recomputed whenever the grid size
  changes, as keras_hexagdly already does. A model always fed the same image
  size, as in training, was not affected.
- **`share_neighbors=False` raised a `ValueError`** ("must be False or one
  of ..., got False"), although 0.2.2 documented it as still working. It now
  means off, like `None`.

## [0.2.3] - 2026-08-28

### Fixed

- **`Conv2d`/`Conv3d`'s default (Kaiming-normal) weight initialization was
  scaled too large, and grew larger with `kernel_size`.** Both layers split
  each output channel's receptive field across several separately-shaped
  sub-kernel tensors (`kernel0`, `kernel1`, ... -- or, with
  `share_neighbors`, one `shared_weights` tensor covering fewer distinct
  values than physical taps) whose contributions are *summed* to produce
  the output. Calling `nn.init.kaiming_normal_()` on each piece
  independently -- as this package originally did, matching upstream
  `hexagdly`'s identical per-sub-kernel `kaiming_normal_()` calls --
  calibrates each piece's variance as if it alone were the entire
  receptive field. The summed output ends up with variance inflated by
  roughly the number of sub-kernels: measured output std for a single
  untrained layer, on unit-variance input, held steady near the correct
  Kaiming target of `sqrt(2)` after the fix (1.30-1.48 across
  `kernel_size` 1-3 and both plain and `share_neighbors` modes), where it
  previously grew with kernel size.
  - This is a real, upstream-inherited bug, not a deliberate design choice
    -- `Conv2d_CustomKernel`/`Conv3d_CustomKernel` are unaffected (they
    require explicit caller-supplied `sub_kernels`, no default init path).
    The same bug (and the same fix) was found and applied independently in
    the Keras port, `keras_hexagdly` 0.4.1 -- see its CHANGELOG.
  - Fixed by initializing from one pool sized by the *combined* fan-in
    across all of a layer's sub-kernels/shared-weight groups, then
    slicing pieces off it -- see `_combined_fan_in_pool` in `__init__.py`.
    `debug=True` (all-ones init) is unaffected.
  - Verified against the shared `hexagdly-oracle` test suite (878 passed,
    7 skipped, 0 failed) -- unchanged from before the fix, since it only
    changes initial weight *values*, not the convolution math or weight
    *shapes*.
  - No change to saved/loaded model weights, since a trained checkpoint's
    weight values come from training, not the initializer.

## [0.2.2] - 2026-07-03

### Changed

- `share_neighbors` default changed from `False` to `None` on `Conv2d` and `Conv3d`.
  `None` and `False` are both falsy so existing code passing `share_neighbors=False`
  explicitly continues to work without change.

## [0.2.1] - 2026-07-03

### Fixed

- `share_neighbors` with `kernel_size=2`: the odd-column weight-group maps for
  the `ring`, `diag`, and `sym` modes had their off-axis (`sub1`) rows
  cyclically rotated, so odd-column centres grouped the wrong physical hexes
  (e.g. `diag` pairs were no longer 180° antipodes). The odd-column map is now
  identical to the even-column one — the half-cell column shift is handled by
  the slice/stride machinery alone, matching the `kernel_size=1` design.
  Verified against an independent cube-coordinate hex model. Affects both
  `Conv2d` and `Conv3d`.

### Added

- `Conv3d` smoke tests for every `share_neighbors` mode / kernel size on both
  even- and odd-width inputs (the 3d + weight-sharing path was previously
  untested).

## [0.1.0] - 2026-07-02

Initial release of `pytorch-hexagdly`, a fork of
[HexagDLy](https://github.com/ai4iacts/hexagdly).

### Added

- `share_neighbors` parameter on `Conv2d` and `Conv3d`: ties hexagonal kernel
  weights by ring (ring 0 = center, ring *r* = the 6*r* cells at hex-distance
  *r*) instead of giving every cell an independent weight, like TDSCAN.
- `depth_padding` parameter on `Conv3d`: `"same"` zero-pads the depth/time
  axis so the temporal kernel is centred and output depth equals input depth,
  instead of upstream's `"valid"`-only behaviour.
- `ring_maps_2d(n)`: utility function returning the per-sub-kernel ring-index
  maps for kernel size `n`, derived empirically via impulse responses.
- PyPI packaging: `pyproject.toml` with hatchling backend, proper package
  layout under `src/pytorch_hexagdly/`, `__version__`, and `__all__`.
- GitHub Actions CI: test workflow (Python 3.10/3.11/3.12) and publish workflow
  (tag-triggered, PyPI Trusted Publishing / OIDC).
