# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
