# pytorch-hexagdly — Hexagonal Convolutions for PyTorch

`pytorch-hexagdly` is a fork of [HexagDLy](https://github.com/ai4iacts/hexagdly)
that extends the original hexagonal convolution and pooling layers for PyTorch with
two new features: **ring-shared weights** (`share_neighbors`) and **depth-axis same-padding**
(`depth_padding="same"` on `Conv3d`).

- [Getting Started](#getting-started)
- [New Features](#new-features)
- [Preparing the Data](#preparing-the-data)
- [How to use pytorch-hexagdly](#how-to-use-pytorch-hexagdly)
- [General Concept](#general-concept)
- [Disclaimer](#disclaimer)
- [Citing HexagDLy](#citation)


## Getting Started

### Pip Installation

```
pip install pytorch-hexagdly
```

```python
import pytorch_hexagdly
```

To get the dependencies needed to run the provided [unit tests](tests) and
[notebooks](notebooks), add the `dev` option:

```
pip install pytorch-hexagdly[dev]
```

### Manual Installation

Requires a working installation of [PyTorch](https://github.com/pytorch/pytorch).
Clone the repository and install in editable mode:

```
git clone https://github.com/YugnatD/pytorch-hexagdly
cd pytorch-hexagdly
pip install -e .
```


## New Features

### `share_neighbors` — ring-shared kernel weights

Available on `Conv2d` and `Conv3d`. When set to `True`, all cells at the same
hexagonal ring distance share a single weight, reducing the number of learnable
parameters. Ring 0 is the center pixel; ring *r* covers the 6*r* cells at
hex-distance *r*. This mirrors the TDSCAN triggering approach.

```python
import torch
import pytorch_hexagdly

conv = pytorch_hexagdly.Conv2d(1, 8, kernel_size=2, stride=1, share_neighbors=True)
x = torch.randn(1, 1, 21, 21)
print(conv(x).shape)
```

### `depth_padding="same"` — temporal same-padding for `Conv3d`

When `depth_padding="same"`, the depth/time axis is zero-padded symmetrically so
the output depth equals the input depth. The default is `"valid"` (upstream behaviour).

```python
conv3d = pytorch_hexagdly.Conv3d(1, 4, kernel_size=(3, 1), stride=1,
                                  depth_padding="same")
x = torch.randn(1, 1, 10, 21, 21)
print(conv3d(x).shape)  # depth dimension preserved
```


## How to use pytorch-hexagdly

As `pytorch-hexagdly` is based on PyTorch, it is of advantage to be familiar with
PyTorch's functionalities and concepts. Before applying it, ensure that the input
data has the correct hexagonal layout. An [example notebook](notebooks/how_to_apply_adressing_scheme.ipynb)
illustrates the steps to get data into the correct format.

Basic example:

```python
import torch
import pytorch_hexagdly

kernel_size, stride = 1, 4
in_channels, out_channels = 1, 3

hexconv = pytorch_hexagdly.Conv2d(in_channels, out_channels, kernel_size, stride)
input = torch.rand(1, 1, 21, 21)
output = hexconv(input)
```

HexagDLy uses an addressing scheme to map hexagonal grid data to a square tensor.
The layout from top to bottom (along tensor index 2) must be of zig-zag-edge shape
and from left to right (along tensor index 3) of armchair-edge shape.

Additional examples for basic use-cases are shown in the [notebooks](notebooks) folder.


## General Concept

As common deep learning frameworks process data on square grids, hexagonally sampled
data must be mapped to a square tensor. This conversion is non-trivial due to the
different symmetries of square (4-fold) vs hexagonal (6-fold) grids.

HexagDLy solves this by splitting each convolution kernel into sub-kernels that
together cover the true neighbours of a data point in the hexagonal grid. A full
hexagonal convolution with size 1 (next-neighbour kernel) decomposes into three
sub-convolutions with two different sub-kernels applied to three differently padded
versions of the input.

![kerne size+stride](figures/kernel_size+stride.png "Examples of different kernels of different size and strides.")

**Please note**: Operations are only performed where the center point of a kernel is
located within the input tensor. This could result in output columns of different
length; in such cases the output will be sliced according to the shortest column.

![violating_symmetry](figures/violating_symmetry.png "Squeezing hexagonal data in a square grid and applying square convolution kernels disregards the symmetry of the hexagonal lattice.")

![explicit_next_neighbour_conv](figures/explicit_next_neighbour_conv.png "Schematic description of the individual sub-convolutions and combination of the individual outputs to perform a hexagonal convolution.")


## Disclaimer

`pytorch-hexagdly` is built as an easy-to-use prototyping tool to design convolutional
neural networks for hexagonally sampled data. The implemented methods aim for
flexibility rather than performance. Once a model is optimized, hard-coding kernel
size, stride and input dimensions will make the implementation faster.


## Authors

**Fork (`pytorch-hexagdly`)**
* **Tanguy Dietrich** — HEPIA / SST-1M Collaboration

**Original HexagDLy**
* **Tim Lukas Holch**
* **Constantin Steppa**

See [NOTICE.md](NOTICE.md) for full attribution.


## License

MIT license — see [LICENSE](LICENSE).


## Citation

If this work has helped your research, please cite the original HexagDLy paper:

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
    keywords = "Convolutional neural networks, Hexagonal grid, PyTorch, Astroparticle physics",
    abstract = "HexagDLy is a Python-library extending the PyTorch deep learning framework with convolution and pooling operations on hexagonal grids. It aims to ease the access to convolutional neural networks for applications that rely on hexagonally sampled data as, for example, commonly found in ground-based astroparticle physics experiments."
}
```

HexagDLy was developed as part of a research study in ground-based gamma-ray astronomy
published in [Astroparticle Physics](https://doi.org/10.1016/j.astropartphys.2018.10.003).


## Acknowledgments

The original HexagDLy project evolved by exploring new analysis techniques for Imaging
Atmospheric Cherenkov Telescopes with H.E.S.S. The fork was developed in the context of
the SST-1M Collaboration / HEPIA TDSCAN triggering project.
