# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
