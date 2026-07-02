# Notice

`pytorch-hexagdly` is a fork of
[HexagDLy](https://github.com/ai4iacts/hexagdly) by Tim Lukas Holch and
Constantin Steppa (ai4iacts), originally developed for hexagonal convolution
and pooling on PyTorch in the context of Imaging Atmospheric Cherenkov
Telescope analysis with H.E.S.S.

This package extends the original `HexBase` sub-kernel decomposition with two
new features that have no equivalent in upstream HexagDLy:

- `share_neighbors`: ties the weights of a hexagonal kernel by ring (ring 0 =
  center, ring *r* = the 6*r* cells at hex-distance *r*), instead of giving
  every cell its own weight.
- `depth_padding` (`Conv3d` only): `"same"` zero-pads the depth/time axis so
  the temporal kernel is centred and output depth equals input depth, instead of
  HexagDLy's `"valid"`-only behaviour.

This work was developed as part of the SST-1M Collaboration / HEPIA TDSCAN
triggering project.

## License

Both the original HexagDLy code and this fork are distributed under the MIT
license; see [LICENSE](LICENSE). The original copyright notice
(Copyright (c) 2018 ai4iacts) is preserved alongside the copyright notice for
this fork, as required by the MIT license.

## Citing

If you use this package, please cite the original HexagDLy paper:

```bibtex
@article{hexagdly_paper,
    title = "HexagDLy—Processing hexagonally sampled data with CNNs in PyTorch",
    author = "Constantin Steppa and Tim L. Holch",
    journal = "SoftwareX",
    volume = "9",
    pages = "193 - 198",
    year = "2019",
    issn = "2352-7110",
    doi = "https://doi.org/10.1016/j.softx.2019.02.010",
    url = "https://www.sciencedirect.com/science/article/pii/S2352711018302723",
}
```
