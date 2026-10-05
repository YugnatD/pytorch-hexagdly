r"""
This file contains utilities to set up hexagonal convolution and pooling
kernels in PyTorch. The size of the input is abitrary, whereas the layout
from top to bottom (along tensor index 2) has to be of zig-zag-edge shape
and from left to right (along tensor index 3) of armchair-edge shape as
shown below.
 __    __                                 __ __ __ __
/11\__/31\__  . . .                      |11|21|31|41| . . .
\__/21\__/41\                            |__|__|__|__|
/12\__/32\__/ . . .        _______|\     |12|22|32|42| . . .
\__/22\__/42\             |         \    |__|__|__|__|
   \__/  \__/             |_______  /
 .  .  .  .  .                    |/       .  .  .  .  .
 .  .  .  .    .                           .  .  .  .    .
 .  .  .  .      .                         .  .  .  .      .

For more information visit https://github.com/ai4iacts/hexagdly

"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

# Read from the installed package's own metadata (set from pyproject.toml's
# `version` at build time) rather than a second hardcoded string here -- the
# Keras port (keras_hexagdly) had exactly this constant drift stale across
# three releases in a row before switching to the same approach; see its
# CHANGELOG [0.4.2].
try:
    __version__ = _version("pytorch-hexagdly")
except PackageNotFoundError:
    # Running from source without an installed/editable package (e.g. a
    # bare checkout on sys.path with no `pip install`).
    __version__ = "unknown"

__all__ = [
    "AvgPool2d",
    "Conv2d",
    "Conv2d_CustomKernel",
    "Conv3d",
    "Conv3d_CustomKernel",
    "MaxPool2d",
    "MaxPool3d",
    "ring_maps_2d",
    "diag_maps_2d",
    "sym_maps_2d",
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

SHARE_NEIGHBORS_MODES = ("ring", "diag", "sym")


def _check_positive_int(name, value):
    """Reject a non-positive or non-integer kernel_size/stride up front.

    Without this, a bad value (0, negative, or a float) doesn't fail until the
    first forward pass, with an error that depends on where in the sub-kernel
    decomposition it breaks (ZeroDivisionError, a TypeError about lists...).
    Upstream HexagDLy has the same gap. keras_hexagdly checks the same way.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _check_positive_int_or_pair(name, value):
    """Like _check_positive_int, but also accepts a (depth, hex) tuple, the
    Conv3d/MaxPool3d convention."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"{name} must be a positive integer or a (depth, hex) tuple, got {value!r}."
            )
        _check_positive_int(f"{name}[0]", value[0])
        _check_positive_int(f"{name}[1]", value[1])
    else:
        _check_positive_int(name, value)


def _normalize_share_neighbors(value):
    """None/False -> None (off); True -> "ring", as in keras_hexagdly; a mode
    string -> itself, validated."""
    if value is None or value is False:
        return None
    if value is True:
        return "ring"
    if value in SHARE_NEIGHBORS_MODES:
        return value
    raise ValueError(
        f"share_neighbors must be None, False, True or one of {SHARE_NEIGHBORS_MODES}, "
        f"got {value!r}"
    )


def _check_padding(padding):
    """Validate the ``padding`` argument of the 2D layers and return it.

    The hex layers always pad the grid themselves, so every input cell is a
    window centre at stride 1 (what ``torch.nn`` calls ``"same"``). ``padding``
    is accepted so calls written for ``torch.nn.Conv2d`` / ``MaxPool2d`` /
    ``AvgPool2d`` work unchanged: ``"same"`` and integer paddings, such as the
    usual ``padding=kernel_size // 2``, are accepted and change nothing.
    ``"valid"`` asks for a different output grid, which the hex layers can't
    produce, so it is rejected rather than ignored.
    """
    if padding == "same":
        return padding
    ints = padding if isinstance(padding, tuple) else (padding,)
    if all(isinstance(p, int) and not isinstance(p, bool) and p >= 0 for p in ints):
        return padding
    raise ValueError(
        f"padding must be 'same' or non-negative integers, got {padding!r}. The hex grid is "
        "always padded so that every cell is a window centre; 'valid' isn't supported."
    )


def _combined_fan_in_pool(out_channels, total_fan_in, debug):
    """Draw one correctly-scaled pool of weight values, to be sliced into a
    hex conv's several sub-kernel/shared-weight-group pieces.

    A hex convolution splits each output channel's receptive field across
    several separately-shaped tensors (kernel0, kernel1, ... -- or, with
    share_neighbors, one shared_weights tensor covering fewer distinct
    values than physical taps) whose contributions are SUMMED to produce
    the output. Calling kaiming_normal_ independently on each piece --
    which used its own (too-small) shape to compute fan_in -- calibrated
    each piece's variance as if it alone were the entire receptive field,
    so the summed output ends up with variance inflated by roughly the
    number of pieces (same bug also present in upstream hexagdly's
    identical per-sub-kernel kaiming_normal_ calls; fixed independently in
    keras_hexagdly -- see that port's CHANGELOG [0.4.1] for the full
    measurement and explanation).

    Since kaiming_normal_ draws i.i.d. elements, a contiguous slice of a
    pool sized by the TRUE combined fan-in is itself correctly scaled -- so
    init_parameters draws one (out_channels, total_fan_in) pool here and
    slices+reshapes pieces off it for each sub-kernel/shared-weights
    tensor, instead of initializing each piece from its own shape.

    debug=True keeps using all-ones (a pool of ones sliced anywhere is
    still all 1s, so no special-casing needed there).
    """
    if debug:
        return torch.ones(out_channels, total_fan_in)
    pool = torch.empty(out_channels, total_fan_in)
    nn.init.kaiming_normal_(pool)
    return pool


class HexBase:
    def __init__(self):
        super(HexBase, self).__init__()
        self.hexbase_size = None
        self.depth_size = None
        self.hexbase_stride = None
        self.depth_stride = None
        self._plan_shape = None
        self.odd_columns_slices = []
        self.odd_columns_pads = []
        self.even_columns_slices = []
        self.even_columns_pads = []
        self.dimensions = None
        self.combine = None
        self.process = None
        self.kwargs = dict()

    def shape_for_odd_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        pads[0] = kernel_number
        # right
        pads[1] = max(0, kernel_number - ((input_size[-1] - 1) % (2 * self.hexbase_stride)))
        # top
        pads[2] = self.hexbase_size - int(kernel_number / 2)
        # bottom
        constraint = (
            input_size[-2]
            - 1
            - int((input_size[-2] - 1 - int(self.hexbase_stride / 2)) / self.hexbase_stride)
            * self.hexbase_stride
        )
        bottom = (self.hexbase_size - int((kernel_number + 1) / 2)) - constraint
        if bottom >= 0:
            pads[3] = bottom
        else:
            slices[1] = bottom

        return slices, pads

    def shape_for_even_columns(self, input_size, kernel_number):
        slices = [None, None, None, None]
        pads = [0, 0, 0, 0]
        # left
        left = kernel_number - self.hexbase_stride
        if left >= 0:
            pads[0] = left
        else:
            slices[2] = -left
        # right
        pads[1] = max(
            0,
            kernel_number
            - ((input_size[-1] - 1 - self.hexbase_stride) % (2 * self.hexbase_stride)),
        )
        # top
        top_shift = -(kernel_number % 2) if (self.hexbase_stride % 2) == 1 else 0
        top = (
            (self.hexbase_size - int(kernel_number / 2)) + top_shift - int(self.hexbase_stride / 2)
        )
        if top >= 0:
            pads[2] = top
        else:
            slices[0] = -top
        # bottom
        bottom_shift = 0 if (self.hexbase_stride % 2) == 1 else -(kernel_number % 2)
        pads[3] = max(
            0,
            self.hexbase_size
            - int(kernel_number / 2)
            + bottom_shift
            - ((input_size[-2] - int(self.hexbase_stride / 2) - 1) % self.hexbase_stride),
        )

        return slices, pads

    def get_padded_input(self, input, pads):
        if self.dimensions == 2:
            return nn.ZeroPad2d(tuple(pads))(input)
        elif self.dimensions == 3:
            return nn.ConstantPad3d(tuple(pads + [0, 0]), 0)(input)

    def get_sliced_input(self, input, slices):
        if self.dimensions == 2:
            return input[:, :, slices[0] : slices[1], slices[2] : slices[3]]
        elif self.dimensions == 3:
            return input[:, :, :, slices[0] : slices[1], slices[2] : slices[3]]

    def get_dilation(self, dilation_2d):
        if self.dimensions == 2:
            return dilation_2d
        elif self.dimensions == 3:
            return tuple([1] + list(dilation_2d))

    def get_stride(self):
        if self.dimensions == 2:
            return (self.hexbase_stride, 2 * self.hexbase_stride)
        elif self.dimensions == 3:
            return (self.depth_stride, self.hexbase_stride, 2 * self.hexbase_stride)

    def get_ordered_output(self, input, order):
        if self.dimensions == 2:
            return input[:, :, :, order]
        elif self.dimensions == 3:
            return input[:, :, :, :, order]

    def _odd_col_kernel(self, i):
        """Return the odd-column variant of sub-kernel i if present, else the standard one.

        Odd-indexed sub-kernels in diag/sym modes are materialized with two sets of
        weights — one for even-column centres and one for odd-column centres.
        The standard kernel_i holds the even-column version; kernel_i_odd the other.
        """
        attr = "kernel" + str(i) + "_odd"
        if hasattr(self, attr):
            return getattr(self, attr)
        return getattr(self, "kernel" + str(i))

    # general implementation of an operation with a hexagonal kernel
    def operation_with_arbitrary_stride(self, input):
        assert input.size(-2) - (self.hexbase_stride // 2) >= 0, (
            "Too few rows to apply hex conv with the stide that is set"
        )
        odd_columns = None
        even_columns = None
        # With no more columns than the stride, the even-column group of the
        # output is empty and the odd group is the whole output. Computing the
        # empty group anyway runs an op on a zero-width input, which fails (as
        # it does in upstream HexagDLy), so it is skipped.
        has_even_columns = input.size(-1) > self.hexbase_stride

        # The per-sub-kernel slices/pads depend on the grid's height and width,
        # so they are recomputed whenever those change, not only on the first
        # call: reusing them for another size crashes or misindexes. (getattr:
        # layers pickled by an older version don't have _plan_shape yet.)
        plan_shape = tuple(input.shape[-2:])
        if plan_shape != getattr(self, "_plan_shape", None):
            self.odd_columns_slices = []
            self.odd_columns_pads = []
            self.even_columns_slices = []
            self.even_columns_pads = []
            for i in range(self.hexbase_size + 1):
                slices, pads = self.shape_for_odd_columns(input.size(), i)
                self.odd_columns_slices.append(slices)
                self.odd_columns_pads.append(pads)
                slices, pads = self.shape_for_even_columns(input.size(), i)
                self.even_columns_slices.append(slices)
                self.even_columns_pads.append(pads)
            self._plan_shape = plan_shape

        for i in range(self.hexbase_size + 1):
            dilation_base = (1, 1) if i == 0 else (1, 2 * i)

            # odd_columns stream: centres land on odd output columns → even-col kernel map
            ki_odd_stream = getattr(self, "kernel" + str(i))
            # even_columns stream: centres land on even output columns → odd-col kernel map
            ki_even_stream = self._odd_col_kernel(i)

            if odd_columns is None:
                odd_columns = self.process(
                    self.get_padded_input(
                        self.get_sliced_input(input, self.odd_columns_slices[i]),
                        self.odd_columns_pads[i],
                    ),
                    ki_odd_stream,
                    dilation=self.get_dilation(dilation_base),
                    stride=self.get_stride(),
                    **self.kwargs,
                )
            else:
                odd_columns = self.combine(
                    odd_columns,
                    self.process(
                        self.get_padded_input(
                            self.get_sliced_input(input, self.odd_columns_slices[i]),
                            self.odd_columns_pads[i],
                        ),
                        ki_odd_stream,
                        dilation=self.get_dilation(dilation_base),
                        stride=self.get_stride(),
                    ),
                )

            if not has_even_columns:
                continue
            if even_columns is None:
                even_columns = self.process(
                    self.get_padded_input(
                        self.get_sliced_input(input, self.even_columns_slices[i]),
                        self.even_columns_pads[i],
                    ),
                    ki_even_stream,
                    dilation=self.get_dilation(dilation_base),
                    stride=self.get_stride(),
                    **self.kwargs,
                )
            else:
                even_columns = self.combine(
                    even_columns,
                    self.process(
                        self.get_padded_input(
                            self.get_sliced_input(input, self.even_columns_slices[i]),
                            self.even_columns_pads[i],
                        ),
                        ki_even_stream,
                        dilation=self.get_dilation(dilation_base),
                        stride=self.get_stride(),
                    ),
                )

        if not has_even_columns:
            return odd_columns

        concatenated_columns = torch.cat((odd_columns, even_columns), 1 + self.dimensions)

        n_odd_columns = odd_columns.size(-1)
        n_even_columns = even_columns.size(-1)
        if n_odd_columns == n_even_columns:
            order = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            order = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            order.append(n_even_columns)

        return self.get_ordered_output(concatenated_columns, order)

    # a slightly faster, case specific implementation of the hexagonal convolution
    def operation_with_single_hexbase_stride(self, input):
        columns_mod2 = input.size(-1) % 2
        # A single-column grid has no even columns, so the odd-kernel
        # even-column group is empty -- skipped for the same reason as in
        # operation_with_arbitrary_stride.
        has_even_columns = input.size(-1) > 1
        odd_kernels_odd_columns = []
        odd_kernels_even_columns = []
        even_kernels_all_columns = []

        even_kernels_all_columns = self.process(
            self.get_padded_input(input, [0, 0, self.hexbase_size, self.hexbase_size]),
            self.kernel0,
            stride=(1, 1) if self.dimensions == 2 else (self.depth_stride, 1, 1),
            **self.kwargs,
        )
        if self.hexbase_size >= 1:
            # odd-column centres use kernel1 (even-col map);
            # even-column centres use _odd_col_kernel(1) (odd-col map).
            odd_kernels_odd_columns = self.process(
                self.get_padded_input(
                    input, [1, columns_mod2, self.hexbase_size, self.hexbase_size - 1]
                ),
                self.kernel1,
                dilation=self.get_dilation((1, 2)),
                stride=self.get_stride(),
            )
            if has_even_columns:
                odd_kernels_even_columns = self.process(
                    self.get_padded_input(
                        input,
                        [0, 1 - columns_mod2, self.hexbase_size - 1, self.hexbase_size],
                    ),
                    self._odd_col_kernel(1),
                    dilation=self.get_dilation((1, 2)),
                    stride=self.get_stride(),
                )

        if self.hexbase_size > 1:
            for i in range(2, self.hexbase_size + 1):
                if i % 2 == 0:
                    even_kernels_all_columns = self.combine(
                        even_kernels_all_columns,
                        self.process(
                            self.get_padded_input(
                                input,
                                [
                                    i,
                                    i,
                                    self.hexbase_size - int(i / 2),
                                    self.hexbase_size - int(i / 2),
                                ],
                            ),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=(1, 1) if self.dimensions == 2 else (self.depth_stride, 1, 1),
                        ),
                    )
                else:
                    x = self.hexbase_size + int((1 - i) / 2)
                    odd_kernels_odd_columns = self.combine(
                        odd_kernels_odd_columns,
                        self.process(
                            self.get_padded_input(input, [i, i - 1 + columns_mod2, x, x - 1]),
                            getattr(self, "kernel" + str(i)),
                            dilation=self.get_dilation((1, 2 * i)),
                            stride=self.get_stride(),
                        ),
                    )
                    if has_even_columns:
                        odd_kernels_even_columns = self.combine(
                            odd_kernels_even_columns,
                            self.process(
                                self.get_padded_input(input, [i - 1, i - columns_mod2, x - 1, x]),
                                self._odd_col_kernel(i),
                                dilation=self.get_dilation((1, 2 * i)),
                                stride=self.get_stride(),
                            ),
                        )

        if not has_even_columns:
            return self.combine(even_kernels_all_columns, odd_kernels_odd_columns)

        odd_kernels_concatenated_columns = torch.cat(
            (odd_kernels_odd_columns, odd_kernels_even_columns), 1 + self.dimensions
        )

        n_odd_columns = odd_kernels_odd_columns.size(-1)
        n_even_columns = odd_kernels_even_columns.size(-1)
        if n_odd_columns == n_even_columns:
            order = [int(i + x * n_even_columns) for i in range(n_even_columns) for x in range(2)]
        else:
            order = [int(i + x * n_odd_columns) for i in range(n_even_columns) for x in range(2)]
            order.append(n_even_columns)

        return self.combine(
            even_kernels_all_columns,
            self.get_ordered_output(odd_kernels_concatenated_columns, order),
        )


# ----------------------------------------------------------------------------
# Weight-sharing modes for share_neighbors:
#   "ring" — one weight per hex ring (n+1 weights total). Derived empirically:
#            a single-tap impulse through the conv reveals each cell's physical
#            (row, col) offset; ring = smallest kernel size whose support contains it.
#   "diag" — antipodal pairs: cells at offset (dr,dc) and (-dr,-dc) share a weight.
#            Gives 1 + 3*n*(n+1)/2 weights.
#   "sym"  — adjacent pairs along the outer boundary of each ring, grouped by 60°
#            rotational symmetry. Same weight count as "diag".
# All three modes use the same impulse-response infrastructure (_tap_offset) to get
# physical offsets; only the grouping rule differs.
# ----------------------------------------------------------------------------

_WEIGHT_MAP_CACHE = {}

# Hardcoded weight-group index maps for "diag" and "sym" at kernel sizes 1 and 2.
# Keyed by (mode, kernel_size, col_parity) where col_parity = input_col % 2.
# The off-axis sub-kernels (sub1) shift by one row depending on whether the center
# pixel sits in an even or odd column of the input tensor — this is the same
# zig-zag offset that HexagDLy uses for all its sub-kernel decompositions.
# sub0 and sub2+ are identical for both parities; only sub1 differs.
# Values computed algorithmically for each parity and verified.

_HARDCODED_MAPS = {
    # --- ring n=1 (identical for both parities: center=0, all 6 neighbors=1) ---
    ("ring", 1, 0): (
        [np.array([[1], [0], [1]]), np.array([[1, 1], [1, 1]])],
        2,
    ),
    ("ring", 1, 1): (
        [np.array([[1], [0], [1]]), np.array([[1, 1], [1, 1]])],
        2,
    ),
    # --- ring n=2 ---
    # From user-validated pointy-top ASCII:
    #           [ B ]         A=1 (ring-1, 6 cells): (+1,0),(-1,0),(0,-1),(0,+1),(-1,-1),(-1,+1)
    #        [ B ] [ B ]      B=2 (ring-2, 12 cells): all others
    #     [ B ] [ A ] [ B ]
    #        [ A ] [ A ]
    #     [ B ] [ X ] [ B ]
    #        [ A ] [ A ]
    #     [ B ] [ A ] [ B ]
    #        [ B ] [ B ]
    #           [ B ]
    ("ring", 2, 0): (
        [
            np.array([[2], [1], [0], [1], [2]]),  # sub0
            np.array([[2, 2], [1, 1], [1, 1], [2, 2]]),  # sub1 even
            np.array([[2, 2], [2, 2], [2, 2]]),  # sub2
        ],
        3,
    ),
    ("ring", 2, 1): (
        [
            np.array([[2], [1], [0], [1], [2]]),  # sub0
            np.array([[2, 2], [1, 1], [1, 1], [2, 2]]),  # sub1 odd
            np.array([[2, 2], [2, 2], [2, 2]]),  # sub2
        ],
        3,
    ),
    # --- diag n=1 (identical for both parities) ---
    ("diag", 1, 0): (
        [np.array([[3], [0], [3]]), np.array([[1, 2], [2, 1]])],
        4,
    ),
    ("diag", 1, 1): (
        [np.array([[3], [0], [3]]), np.array([[1, 2], [2, 1]])],
        4,
    ),
    # --- diag n=2 ---
    # 9 visual-antipodal pairs from the flat-top hex kernel (read from pointy-top ASCII):
    #   E=1: (+2,0)<->(-2,0)     C=5: (+1,0)<->(-1,0)     D=9: (0,-2)<->(0,+2)
    #   F=2: (+1,-1)<->(-2,+1)   H=4: (+1,-2)<->(-1,+2)
    #   I=3: (+1,+1)<->(-2,-1)   G=6: (+1,+2)<->(-1,-2)
    #   B=7: (0,-1)<->(-1,+1)    A=8: (0,+1)<->(-1,-1)
    # Orphans (-2,±1) [even] or (+2,±1) [odd] pair with (+1,∓1) or (-1,±1) respectively.
    ("diag", 2, 0): (
        [
            np.array([[1], [5], [0], [5], [1]]),  # sub0
            np.array([[3, 2], [8, 7], [7, 8], [2, 3]]),  # sub1 even
            np.array([[6, 4], [9, 9], [4, 6]]),  # sub2
        ],
        10,
    ),
    ("diag", 2, 1): (
        [
            np.array([[1], [5], [0], [5], [1]]),  # sub0
            np.array([[3, 2], [8, 7], [7, 8], [2, 3]]),  # sub1 odd
            np.array([[6, 4], [9, 9], [4, 6]]),  # sub2
        ],
        10,
    ),
    # --- sym n=1 (identical for both parities) ---
    ("sym", 1, 0): (
        [np.array([[2], [0], [3]]), np.array([[2, 1], [3, 1]])],
        4,
    ),
    ("sym", 1, 1): (
        [np.array([[2], [0], [3]]), np.array([[2, 1], [3, 1]])],
        4,
    ),
    # --- sym n=2 ---
    # 9 adjacent pairs from user-validated pointy-top ASCII:
    #           [ G ]         G=7: (+2,0)<->(+1,-1)
    #        [ G ] [ F ]      F=6: (+1,+1)<->(+1,+2)
    #     [ H ] [ A ] [ F ]   H=8: (+1,-2)<->(0,-2)    A=1: (+1,0)<->(0,-1)
    #        [ A ] [ B ]      B=2: (0,+1)<->(-1,+1)
    #     [ H ] [ X ] [ E ]   E=5: (0,+2)<->(-1,+2)
    #        [ C ] [ B ]      C=3: (-1,-1)<->(-1,0)
    #     [ I ] [ C ] [ E ]   I=9: (-1,-2)<->(-2,-1)[even] or (-1,-2)<->(+2,-1)[odd]
    #        [ I ] [ D ]      D=4: (-2,+1)<->(-2,0)[even] or (+2,+1)<->(-2,0)[odd]
    #           [ D ]
    ("sym", 2, 0): (
        [
            np.array([[4], [3], [0], [1], [7]]),  # sub0
            np.array([[9, 4], [3, 2], [1, 2], [7, 6]]),  # sub1 even
            np.array([[9, 5], [8, 5], [8, 6]]),  # sub2
        ],
        10,
    ),
    ("sym", 2, 1): (
        [
            np.array([[4], [3], [0], [1], [7]]),  # sub0
            np.array([[9, 4], [3, 2], [1, 2], [7, 6]]),  # sub1 odd
            np.array([[9, 5], [8, 5], [8, 6]]),  # sub2
        ],
        10,
    ),
}


def _tap_offset(n, i, r, c):
    """Physical (dr, dc) offset of sub-kernel cell (i, r, c) for kernel size n."""
    g = 6 * n + 11
    cen = g // 2
    imp = torch.zeros(1, 1, g, g)
    imp[0, 0, cen, cen] = 1.0
    sub_kernels = []
    for k in range(n + 1):
        kh = 2 * n + 1 - k
        kw = 1 if k == 0 else 2
        a = np.zeros((1, 1, kh, kw), dtype=np.float32)
        if k == i:
            a[0, 0, r, c] = 1.0
        sub_kernels.append(a)
    layer = Conv2d_CustomKernel(sub_kernels=sub_kernels, stride=1)
    out = layer(imp).detach().numpy()[0, 0]
    pos = np.argwhere(np.isclose(out, 1.0))
    return int(pos[0][0] - cen), int(pos[0][1] - cen)


def _all_offsets(n):
    """Return dict mapping (i, r, c) -> (dr, dc) for all sub-kernel cells of size n."""
    offsets = {}
    for i in range(n + 1):
        rows = 2 * n + 1 - i
        cols = 1 if i == 0 else 2
        for r in range(rows):
            for c in range(cols):
                offsets[(i, r, c)] = _tap_offset(n, i, r, c)
    return offsets


def ring_maps_2d(n):
    """Return ``(weight_maps, num_weights)`` using ring grouping for kernel size ``n``.

    Ring r contains the 6r cells at hex-distance r from center (ring 0 = center).
    Total weights = n + 1.
    """
    return _get_weight_maps(n, "ring")


def diag_maps_2d(n, col_parity=0):
    """Return ``(weight_maps, num_weights)`` using diagonal (antipodal) grouping.

    Cells at offsets (dr, dc) and (-dr, -dc) share a weight. Center is alone.
    Total weights = 1 + 3*n*(n+1)//2.
    ``col_parity`` (0 or 1) selects the map for even/odd input columns.
    """
    return _get_parity_maps(n, "diag", col_parity)


def sym_maps_2d(n, col_parity=0):
    """Return ``(weight_maps, num_weights)`` using 60° rotational symmetry grouping.

    Within each ring, cells are sorted by angle and paired as consecutive adjacent
    neighbors (each pair spans 60°). Center is alone.
    Total weights = 1 + 3*n*(n+1)//2.
    ``col_parity`` (0 or 1) selects the map for even/odd input columns.
    """
    return _get_parity_maps(n, "sym", col_parity)


def _get_parity_maps(n, mode, col_parity):
    """Return (weight_maps, num_weights) for a given mode and column parity."""
    key = (mode, n, col_parity % 2)
    if key in _HARDCODED_MAPS:
        return _HARDCODED_MAPS[key]
    if n not in (1, 2):
        raise NotImplementedError(
            f"share_neighbors={mode!r} is only implemented for kernel_size 1 and 2, "
            f"got kernel_size={n}."
        )
    raise KeyError(f"No hardcoded map for {key}")


def _get_weight_maps(n, mode):
    """Dispatcher: compute and cache (weight_maps, num_weights) for given mode.

    For 'diag' and 'sym', returns the even-column (parity=0) maps.
    Use _get_parity_maps() directly when parity matters.
    """
    key = (n, mode)
    if key in _WEIGHT_MAP_CACHE:
        return _WEIGHT_MAP_CACHE[key]

    offsets = _all_offsets(n)

    if mode == "ring":
        # Use hardcoded maps when available (n=1,2); fall back to algorithmic for n>2.
        if (mode, n, 0) in _HARDCODED_MAPS:
            result = _HARDCODED_MAPS[(mode, n, 0)]
            _WEIGHT_MAP_CACHE[key] = result
            return result

        # Algorithmic fallback for n>2: build ring support sets.
        support = {}
        for ks in range(1, n + 1):
            offs = set()
            for i in range(ks + 1):
                rows = 2 * ks + 1 - i
                cols = 1 if i == 0 else 2
                for r in range(rows):
                    for c in range(cols):
                        offs.add(_tap_offset(ks, i, r, c))
            support[ks] = offs

        def group_of(off):
            if off == (0, 0):
                return 0
            for ks in range(1, n + 1):
                if off in support[ks]:
                    return ks
            raise ValueError(f"offset {off} not within kernel size {n}")

    elif mode in ("diag", "sym"):
        result = _get_parity_maps(n, mode, 0)
        _WEIGHT_MAP_CACHE[key] = result
        return result

    else:
        raise ValueError(
            f"Unknown share_neighbors mode: {mode!r}. Choose from {SHARE_NEIGHBORS_MODES}."
        )

    # Build the integer index maps (same shape as each sub-kernel).
    weight_maps = []
    for i in range(n + 1):
        rows = 2 * n + 1 - i
        cols = 1 if i == 0 else 2
        m = np.zeros((rows, cols), dtype=np.int64)
        for r in range(rows):
            for c in range(cols):
                m[r, c] = group_of(offsets[(i, r, c)])
        weight_maps.append(m)

    num_weights = int(max(m.max() for m in weight_maps) + 1)
    result = (weight_maps, num_weights)
    _WEIGHT_MAP_CACHE[key] = result
    return result


class Conv2d(HexBase, nn.Module):
    r"""Applies a 2D hexagonal convolution`

    Args:
        in_channels:        int: number of input channels
        out_channels:       int: number of output channels
        kernel_size:        int: number of layers with neighbouring pixels
                                 covered by the pooling kernel
        stride:             int: length of strides
        bias:               bool: add bias if True (default)
        debug:              bool: switch to debug mode
                                False: weights are initalised with
                                       kaiming normal, bias with 0.01 (default)
                                True: weights / bias are set to 1.
        share_neighbors:    str or None: weight-sharing mode (default: None).
                                None:    independent weight per kernel cell.
                                "ring":  one weight per hexagonal ring.
                                "diag":  antipodal cell pairs share a weight.
                                "sym":   adjacent 60° pairs share a weight.
        padding:            str or int: 'same' (default) or a non-negative int,
                                 accepted for torch.nn.Conv2d compatibility.
                                 The hex grid is always padded the same way,
                                 whatever the value; 'valid' raises.

    Examples::

    >>> conv2d = pytorch_hexagdly.Conv2d(1,3,2,1)
    >>> input = torch.randn(1, 1, 4, 2)
    >>> output = conv2d(input)
    >>> print(output)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        bias=True,
        debug=False,
        share_neighbors=None,
        padding="same",
    ):
        super(Conv2d, self).__init__()
        _check_positive_int("kernel_size", kernel_size)
        _check_positive_int("stride", stride)
        share_neighbors = _normalize_share_neighbors(share_neighbors)
        self.padding = _check_padding(padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hexbase_size = kernel_size
        self.hexbase_stride = stride
        self.debug = debug
        self.bias = bias
        self.share_neighbors = share_neighbors
        self.dimensions = 2
        self.process = F.conv2d
        self.combine = torch.add

        if share_neighbors:
            # Always use per-parity maps so both even and odd column centres
            # get the correct weight grouping.
            maps_even, self.num_shared_weights = _get_parity_maps(
                self.hexbase_size, share_neighbors, 0
            )
            maps_odd, _ = _get_parity_maps(self.hexbase_size, share_neighbors, 1)
            self._weight_idx = [torch.as_tensor(m, dtype=torch.long) for m in maps_even]
            # Only store the odd variant when the two parities actually differ
            if any(not np.array_equal(maps_even[i], maps_odd[i]) for i in range(len(maps_even))):
                self._weight_idx_odd = [torch.as_tensor(m, dtype=torch.long) for m in maps_odd]
            else:
                self._weight_idx_odd = None
            self.shared_weights = Parameter(
                torch.Tensor(out_channels, in_channels, self.num_shared_weights)
            )
        else:
            for i in range(self.hexbase_size + 1):
                setattr(
                    self,
                    "kernel" + str(i),
                    Parameter(
                        torch.Tensor(
                            out_channels,
                            in_channels,
                            1 + 2 * self.hexbase_size - i,
                            1 if i == 0 else 2,
                        )
                    ),
                )
        if self.bias:
            self.bias_tensor = Parameter(torch.Tensor(out_channels))
            self.kwargs = {"bias": self.bias_tensor}
        else:
            self.kwargs = {"bias": None}
        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        # Total physical taps across kernel0..kernel{hexbase_size} -- the TRUE
        # combined receptive field whose summed output needs variance-preserving
        # init. See _combined_fan_in_pool for why this must be combined, not
        # computed per sub-kernel/shared-weights-group.
        total_taps = sum(
            (1 + 2 * self.hexbase_size - i) * (1 if i == 0 else 2)
            for i in range(self.hexbase_size + 1)
        )
        pool = _combined_fan_in_pool(self.out_channels, total_taps * self.in_channels, debug)

        if self.share_neighbors:
            count = self.in_channels * self.num_shared_weights
            with torch.no_grad():
                self.shared_weights.copy_(
                    pool[:, :count].reshape(
                        self.out_channels, self.in_channels, self.num_shared_weights
                    )
                )
            if self.bias:
                nn.init.constant_(self.kwargs["bias"], 1.0 if debug else 0.01)
            return

        offset = 0
        for i in range(self.hexbase_size + 1):
            kh = 1 + 2 * self.hexbase_size - i
            kw = 1 if i == 0 else 2
            count = self.in_channels * kh * kw
            with torch.no_grad():
                getattr(self, "kernel" + str(i)).copy_(
                    pool[:, offset : offset + count].reshape(
                        self.out_channels, self.in_channels, kh, kw
                    )
                )
            offset += count
        if self.bias:
            nn.init.constant_(getattr(self, "kwargs")["bias"], 1.0 if debug else 0.01)

    def _materialize_shared_kernels(self):
        """Broadcast shared_weights into dense sub-kernels for both column parities.

        For ring mode (parity-independent): sets kernel_i for all i.
        For diag/sym modes: sets kernel_i (even-column map) AND kernel_i_odd
        (odd-column map) for every odd-indexed sub-kernel i (i=1,3,...).
        Even-indexed sub-kernels are parity-independent and only get kernel_i.

        The two sets are then consumed by operation_with_single_hexbase_stride
        and operation_with_arbitrary_stride, which pick the right one per call.
        """
        dev = self.shared_weights.device
        for i in range(self.hexbase_size + 1):
            idx_even = self._weight_idx[i].to(dev)
            flat = torch.index_select(self.shared_weights, 2, idx_even.reshape(-1))
            setattr(
                self,
                "kernel" + str(i),
                flat.reshape(self.out_channels, self.in_channels, *idx_even.shape),
            )
            # For odd-indexed sub-kernels in parity-sensitive modes, also build
            # the odd-column version so operation_with_*_stride can reference it.
            if self._weight_idx_odd is not None and i % 2 == 1:
                idx_odd = self._weight_idx_odd[i].to(dev)
                flat_odd = torch.index_select(self.shared_weights, 2, idx_odd.reshape(-1))
                setattr(
                    self,
                    "kernel" + str(i) + "_odd",
                    flat_odd.reshape(self.out_channels, self.in_channels, *idx_odd.shape),
                )

    def forward(self, input):
        if self.share_neighbors:
            self._materialize_shared_kernels()
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={hexbase_size}"
            ", stride={hexbase_stride}"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d_CustomKernel(HexBase, nn.Module):
    r"""Applies a 2D hexagonal convolution with custom kernels`

    Args:
        sub_kernels:        list:   list containing sub-kernels as numpy arrays
        stride:             int:    length of strides
        bias:               array:  numpy array with biases (default: None)
        requires_grad:      bool:   trainable parameters if True (default: False)
        debug:              bool:   If True a kernel of size one with all values
                                    set to 1 will be applied as well as no bias
                                    (default: False)

    Examples::

    Given in the online repository https://github.com/ai4iacts/hexagdly
    """

    def __init__(self, sub_kernels=[], stride=1, bias=None, requires_grad=False, debug=False):
        super(Conv2d_CustomKernel, self).__init__()
        _check_positive_int("stride", stride)
        self.sub_kernels = sub_kernels
        self.bias_array = bias
        self.hexbase_stride = stride
        self.requires_grad = requires_grad
        self.debug = debug
        self.dimensions = 2
        self.process = F.conv2d
        self.combine = torch.add

        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        if debug or len(self.sub_kernels) == 0:
            print("The debug kernel is used for {name}!".format(name=self.__class__.__name__))
            self.sub_kernels = [
                np.array([[[[1], [1], [1]]]]),
                np.array([[[[1, 1], [1, 1]]]]),
            ]
        self.hexbase_size = len(self.sub_kernels) - 1
        self.check_sub_kernels()

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                Parameter(
                    torch.from_numpy(self.sub_kernels[i]).type(torch.FloatTensor),
                    requires_grad=self.requires_grad,
                ),
            )

        if not debug and self.bias_array is not None:
            self.check_bias()
            self.bias_tensor = Parameter(
                torch.from_numpy(self.bias_array).type(torch.FloatTensor),
                requires_grad=self.requires_grad,
            )
            self.kwargs = {"bias": self.bias_tensor}
            self.bias = True
        else:
            self.bias = False
            if self.bias_array is not None:
                print(
                    "{name}: Bias is not used in debug mode!".format(name=self.__class__.__name__)
                )

    def check_sub_kernels(self):
        for i in range(self.hexbase_size + 1):
            assert type(self.sub_kernels[i]).__module__ == np.__name__, (
                "sub-kernels must be given as numpy arrays"
            )
            assert len(self.sub_kernels[i].shape) == 4, (
                "sub-kernels must be of rank 4 for a 2d convolution"
            )
            if i == 0:
                assert self.sub_kernels[i].shape[3] == 1, "first sub-kernel must have only 1 column"
                assert self.sub_kernels[i].shape[2] == 2 * self.hexbase_size + 1, (
                    "first sub-kernel must have 2* (kernel size) + 1 rows"
                )
                self.out_channels = self.sub_kernels[i].shape[0]
                self.in_channels = self.sub_kernels[i].shape[1]
            else:
                assert self.sub_kernels[i].shape[3] == 2, (
                    "sub-kernel {}: all but the first sub-kernel must have 2 columns".format(i)
                )
                assert self.sub_kernels[i].shape[2] == 2 * self.hexbase_size + 1 - i, (
                    "{}. sub-kernel must have 2* (kernel size) + 1 - {} rows".format(i, i)
                )
                assert self.sub_kernels[i].shape[0] == self.out_channels, (
                    "sub-kernel {}: out channels are not consistent".format(i)
                )
                assert self.sub_kernels[i].shape[1] == self.in_channels, (
                    "sub-kernel {}: in channels are not consistent".format(i)
                )

    def check_bias(self):
        assert type(self.bias_array).__module__ == np.__name__, (
            "bias must be given as a numpy array"
        )
        assert len(self.bias_array.shape) == 1, "bias must be of rank 1"
        assert self.bias_array.shape[0] == self.out_channels, (
            "bias must have length equal to number of out channels"
        )

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={hexbase_size}"
            ", stride={hexbase_stride}"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv3d(HexBase, nn.Module):
    r"""Applies a 3D hexagonal convolution`

    Args:
        in_channels:        int: number of input channels
        out_channels:       int: number of output channels
        kernel_size:        int, tuple: number of layers with neighbouring pixels
                                        covered by the pooling kernel
                                int: same number of layers in all dimensions
                                tuple of two ints:
                                    1st int: layers in depth
                                    2nd int: layers in hexagonal base
        stride:             int, tuple: length of strides
                                int: same lenght of strides in each dimension
                                tuple of two ints:
                                    1st int: length of strides in depth
                                    2nd int: length of strides in hexagonal base
        bias:               bool: add bias if True (default)
        debug:              bool: switch to debug mode
                                False: weights are initalised with
                                       kaiming normal, bias with 0.01 (default)
                                True: weights / bias are set to 1.
        share_neighbors:    str or None: weight-sharing mode (default: None).
                                None:    independent weight per kernel cell.
                                "ring":  one weight per hexagonal ring.
                                "diag":  antipodal cell pairs share a weight.
                                "sym":   adjacent 60° pairs share a weight.
        depth_padding:      str: 'valid' (default) or 'same' — 'same' zero-pads
                                 the depth axis so output depth equals input depth

    Examples::

    >>> conv3d = pytorch_hexagdly.Conv3d((1,1), (2,2))
    >>> input = torch.randn(1, 1, 6, 5, 4)
    >>> output = conv3d(input)
    >>> print(output)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        bias=True,
        debug=False,
        share_neighbors=None,
        depth_padding="valid",
    ):
        super(Conv3d, self).__init__()
        if depth_padding not in ("valid", "same"):
            raise ValueError("depth_padding must be 'valid' or 'same'.")
        _check_positive_int_or_pair("kernel_size", kernel_size)
        _check_positive_int_or_pair("stride", stride)
        share_neighbors = _normalize_share_neighbors(share_neighbors)
        self.depth_padding = depth_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.hexbase_size = kernel_size
            self.depth_size = kernel_size
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "Need a tuple of two ints to set kernel size"
            self.hexbase_size = kernel_size[1]
            self.depth_size = kernel_size[0]
        if isinstance(stride, int):
            self.hexbase_stride = stride
            self.depth_stride = stride
        elif isinstance(stride, tuple):
            assert len(stride) == 2, "Need a tuple of two ints to set stride"
            self.hexbase_stride = stride[1]
            self.depth_stride = stride[0]
        self.debug = debug
        self.bias = bias
        self.share_neighbors = share_neighbors
        self.dimensions = 3
        self.process = F.conv3d
        self.combine = torch.add

        if share_neighbors:
            # Share over hex axes only; depth (time) stays independent ->
            # shared_weights is (out, in, depth, num_shared_weights).
            maps_even, self.num_shared_weights = _get_parity_maps(
                self.hexbase_size, share_neighbors, 0
            )
            maps_odd, _ = _get_parity_maps(self.hexbase_size, share_neighbors, 1)
            self._weight_idx = [torch.as_tensor(m, dtype=torch.long) for m in maps_even]
            if any(not np.array_equal(maps_even[i], maps_odd[i]) for i in range(len(maps_even))):
                self._weight_idx_odd = [torch.as_tensor(m, dtype=torch.long) for m in maps_odd]
            else:
                self._weight_idx_odd = None
            self.shared_weights = Parameter(
                torch.Tensor(out_channels, in_channels, self.depth_size, self.num_shared_weights)
            )
        else:
            for i in range(self.hexbase_size + 1):
                setattr(
                    self,
                    "kernel" + str(i),
                    Parameter(
                        torch.Tensor(
                            out_channels,
                            in_channels,
                            self.depth_size,
                            1 + 2 * self.hexbase_size - i,
                            1 if i == 0 else 2,
                        )
                    ),
                )
        if self.bias:
            self.bias_tensor = Parameter(torch.Tensor(out_channels))
            self.kwargs = {"bias": self.bias_tensor}
        else:
            self.kwargs = {"bias": None}

        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        # Total physical taps across kernel0..kernel{hexbase_size} (including
        # the depth axis) -- see Conv2d.init_parameters / _combined_fan_in_pool.
        total_taps = self.depth_size * sum(
            (1 + 2 * self.hexbase_size - i) * (1 if i == 0 else 2)
            for i in range(self.hexbase_size + 1)
        )
        pool = _combined_fan_in_pool(self.out_channels, total_taps * self.in_channels, debug)

        if self.share_neighbors:
            count = self.in_channels * self.depth_size * self.num_shared_weights
            with torch.no_grad():
                self.shared_weights.copy_(
                    pool[:, :count].reshape(
                        self.out_channels,
                        self.in_channels,
                        self.depth_size,
                        self.num_shared_weights,
                    )
                )
            if self.bias:
                nn.init.constant_(self.kwargs["bias"], 1.0 if debug else 0.01)
            return

        offset = 0
        for i in range(self.hexbase_size + 1):
            kh = 1 + 2 * self.hexbase_size - i
            kw = 1 if i == 0 else 2
            count = self.in_channels * self.depth_size * kh * kw
            with torch.no_grad():
                getattr(self, "kernel" + str(i)).copy_(
                    pool[:, offset : offset + count].reshape(
                        self.out_channels, self.in_channels, self.depth_size, kh, kw
                    )
                )
            offset += count
        if self.bias:
            nn.init.constant_(getattr(self, "kwargs")["bias"], 1.0 if debug else 0.01)

    def _materialize_shared_kernels(self):
        """Broadcast shared_weights into dense sub-kernels; depth axis left independent.

        Same parity logic as Conv2d._materialize_shared_kernels: odd-indexed sub-kernels
        in diag/sym modes get both kernel_i (even-col map) and kernel_i_odd (odd-col map).
        """
        dev = self.shared_weights.device
        for i in range(self.hexbase_size + 1):
            idx_even = self._weight_idx[i].to(dev)
            flat = torch.index_select(self.shared_weights, 3, idx_even.reshape(-1))
            setattr(
                self,
                "kernel" + str(i),
                flat.reshape(self.out_channels, self.in_channels, self.depth_size, *idx_even.shape),
            )
            if self._weight_idx_odd is not None and i % 2 == 1:
                idx_odd = self._weight_idx_odd[i].to(dev)
                flat_odd = torch.index_select(self.shared_weights, 3, idx_odd.reshape(-1))
                setattr(
                    self,
                    "kernel" + str(i) + "_odd",
                    flat_odd.reshape(
                        self.out_channels, self.in_channels, self.depth_size, *idx_odd.shape
                    ),
                )

    def forward(self, input):
        if self.share_neighbors:
            # input shape: (batch, channels, depth, rows, cols) — col parity from last dim
            self._materialize_shared_kernels()
        if self.depth_padding == "same":
            # Symmetric zero-pad the depth axis (NCDHW -> axis 2) so the temporal
            # kernel is centred and output depth == input depth, like TDSCAN.
            pad = (self.depth_size - 1) // 2
            top = pad
            bot = self.depth_size - 1 - pad
            input = F.pad(input, [0, 0, 0, 0, top, bot])  # pads last dims; here D
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size=({depth_size}, {hexbase_size})"
            ", stride=({depth_stride}, {hexbase_stride})"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv3d_CustomKernel(HexBase, nn.Module):
    r"""Applies a 3D hexagonal convolution with custom kernels`

    Args:
        sub_kernels:        list: list containing sub-kernels as numpy arrays
        stride:             stride:             int, tuple: length of strides
                                int: same lenght of strides in each dimension
                                tuple of two ints:
                                    1st int: length of strides in depth
                                    2nd int: length of strides in hexagonal base
        requires_grad:      bool:   trainable parameters if True (default: False)
        debug:              bool:   If True a kernel of size one with all values
                                    set to 1 will be applied as well as no bias
                                    (default: False)

    Examples::

    Given in the online repository https://github.com/ai4iacts/hexagdly
    """

    def __init__(self, sub_kernels=[], stride=1, bias=None, requires_grad=False, debug=False):
        super(Conv3d_CustomKernel, self).__init__()
        _check_positive_int_or_pair("stride", stride)
        self.sub_kernels = sub_kernels
        self.bias_array = bias
        if isinstance(stride, int):
            self.hexbase_stride = stride
            self.depth_stride = stride
        elif isinstance(stride, tuple):
            assert len(stride) == 2, "Need a tuple of two ints to set stride"
            self.hexbase_stride = stride[1]
            self.depth_stride = stride[0]
        self.requires_grad = requires_grad
        self.debug = debug
        self.dimensions = 3
        self.process = F.conv3d
        self.combine = torch.add

        self.init_parameters(self.debug)

    def init_parameters(self, debug):
        if debug or len(self.sub_kernels) == 0:
            print("The debug kernel is used for {name}!".format(name=self.__class__.__name__))
            self.sub_kernels = [
                np.array([[[[[1], [1], [1]]]]]),
                np.array([[[[[1, 1], [1, 1]]]]]),
            ]
        self.hexbase_size = len(self.sub_kernels) - 1
        self.check_sub_kernels()

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                Parameter(
                    torch.from_numpy(self.sub_kernels[i]).type(torch.FloatTensor),
                    requires_grad=self.requires_grad,
                ),
            )

        if not debug and self.bias_array is not None:
            self.check_bias()
            self.bias_tensor = Parameter(
                torch.from_numpy(self.bias_array).type(torch.FloatTensor),
                requires_grad=self.requires_grad,
            )
            self.kwargs = {"bias": self.bias_tensor}
            self.bias = True
        else:
            self.bias = False
            print("No bias is used for {name}!".format(name=self.__class__.__name__))

    def check_sub_kernels(self):
        for i in range(self.hexbase_size + 1):
            assert type(self.sub_kernels[i]).__module__ == np.__name__, (
                "sub-kernels must be given as numpy arrays"
            )
            assert len(self.sub_kernels[i].shape) == 5, (
                "sub-kernels must be of rank 5 for a 3d convolution"
            )
            if i == 0:
                assert self.sub_kernels[i].shape[4] == 1, "first sub-kernel must have only 1 column"
                assert self.sub_kernels[i].shape[3] == 2 * self.hexbase_size + 1, (
                    "first sub-kernel must have 2* (kernel size) + 1 rows"
                )
                self.out_channels = self.sub_kernels[i].shape[0]
                self.in_channels = self.sub_kernels[i].shape[1]
                self.depth_size = self.sub_kernels[i].shape[2]
            else:
                assert self.sub_kernels[i].shape[4] == 2, (
                    "sub-kernel {}: all but the first sub-kernel must have 2 columns".format(i)
                )
                assert self.sub_kernels[i].shape[3] == 2 * self.hexbase_size + 1 - i, (
                    "{}th sub-kernel must have 2* (kernel size) + 1 - {} rows".format(i, i)
                )
                assert self.sub_kernels[i].shape[0] == self.out_channels, (
                    "sub-kernel {}: out channels are not consistent".format(i)
                )
                assert self.sub_kernels[i].shape[1] == self.in_channels, (
                    "sub-kernel {}: out channels are not consistent".format(i)
                )
                assert self.sub_kernels[i].shape[2] == self.depth_size, (
                    "sub-kernel {}: depths are not consistent".format(i)
                )

    def check_bias(self):
        assert type(self.bias_array).__module__ == np.__name__, (
            "bias must be given as a numpy array"
        )
        assert len(self.bias_array.shape) == 1, "bias must be of rank 1"
        assert self.bias_array.shape[0] == self.out_channels, (
            "bias must have length equal to number of out channels"
        )

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size=({depth_size}, {hexbase_size})"
            ", stride=({depth_stride}, {hexbase_stride})"
        )
        if self.bias is False:
            s += ", bias=False"
        if self.debug is True:
            s += ", debug=True"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


def _max_pool2d(*args, **kwargs):
    """F.max_pool2d, behind a module-level name.

    F.max_pool2d/3d are locally defined dispatch functions, so a layer that
    stores one of them directly as an attribute can't be pickled, and
    torch.save(model) fails.
    """
    return F.max_pool2d(*args, **kwargs)


def _max_pool3d(*args, **kwargs):
    """F.max_pool3d, behind a module-level name -- see _max_pool2d."""
    return F.max_pool3d(*args, **kwargs)


class MaxPool2d(HexBase, nn.Module):
    r"""Applies a 2D hexagonal max pooling`

    Args:
        kernel_size:        int: number of layers with neighbouring pixels
                                 covered by the pooling kernel
        stride:             int: length of strides
        padding:            str or int: 'same' (default) or a non-negative int,
                                 accepted for torch.nn.MaxPool2d compatibility.
                                 The hex grid is always padded the same way,
                                 whatever the value; 'valid' raises.

    Examples::

        >>> maxpool2d = pytorch_hexagdly.MaxPool2d(1,2)
        >>> input = torch.randn(1, 1, 4, 2)
        >>> output = maxpool2d(input)
        >>> print(output)
    """

    def __init__(self, kernel_size=1, stride=1, padding="same"):
        super(MaxPool2d, self).__init__()
        _check_positive_int("kernel_size", kernel_size)
        _check_positive_int("stride", stride)
        self.padding = _check_padding(padding)
        self.hexbase_size = kernel_size
        self.hexbase_stride = stride
        self.dimensions = 2
        self.process = _max_pool2d
        self.combine = torch.max

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                (1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2),
            )

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = "{name}(kernel_size={hexbase_size}, stride={hexbase_stride})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


def _sum_pool2d(input, window, dilation=1, stride=1):
    """Sum of each window, per channel -- the sum counterpart of F.max_pool2d.

    F.avg_pool2d can't dilate, and the hex sub-kernels are dilated, so this is
    a depthwise convolution with an all-ones kernel instead.
    """
    channels = input.size(1)
    ones = input.new_ones(channels, 1, *window)
    return F.conv2d(input, ones, dilation=dilation, stride=stride, groups=channels)


class AvgPool2d(HexBase, nn.Module):
    r"""Applies a 2D hexagonal average pooling`

    Each output cell is the mean over the cells of its hexagonal window (the
    same window as :class:`MaxPool2d`) that lie inside the grid. Positions past
    the grid border count in neither the sum nor the divisor, like
    ``torch.nn.AvgPool2d(..., count_include_pad=False)``. Cells inside the grid
    always count, including empty (zero) ones such as the cells around a mapped
    camera image -- the same convention :class:`Conv2d` and :class:`MaxPool2d`
    apply to them. This matches ``keras_hexagdly.AvgPool2d``.

    Args:
        kernel_size:        int: number of layers with neighbouring pixels
                                 covered by the pooling kernel
        stride:             int: length of strides
        padding:            str or int: 'same' (default) or a non-negative int,
                                 accepted for torch.nn.AvgPool2d compatibility.
                                 The hex grid is always padded the same way,
                                 whatever the value; 'valid' raises.

    Examples::

        >>> avgpool2d = pytorch_hexagdly.AvgPool2d(1,2)
        >>> input = torch.randn(1, 1, 4, 2)
        >>> output = avgpool2d(input)
        >>> print(output)
    """

    def __init__(self, kernel_size=1, stride=1, padding="same"):
        super(AvgPool2d, self).__init__()
        _check_positive_int("kernel_size", kernel_size)
        _check_positive_int("stride", stride)
        self.padding = _check_padding(padding)
        self.hexbase_size = kernel_size
        self.hexbase_stride = stride
        self.dimensions = 2
        self.process = _sum_pool2d
        self.combine = torch.add

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                (1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2),
            )

    def _window_sum(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def forward(self, input):
        # Summing a tensor of ones over the same windows counts the in-grid
        # cells of each window, since the hex padding is zeros. The window
        # centre is always in the grid, so the count is never zero.
        total = self._window_sum(input)
        count = self._window_sum(torch.ones_like(input[:, :1]))
        return total / count

    def __repr__(self):
        s = "{name}(kernel_size={hexbase_size}, stride={hexbase_stride})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MaxPool3d(HexBase, nn.Module):
    r"""Applies a 3D hexagonal max pooling`

    Args:
        kernel_size:        int, tuple: number of layers with neighbouring pixels
                                        covered by the pooling kernel
                                int: same number of layers in all dimensions
                                tuple of two ints:
                                    1st int: layers in depth
                                    2nd int: layers in hexagonal base
        stride:             int, tuple: length of strides
                                int: same lenght of strides in each dimension
                                tuple of two ints:
                                    1st int: length of strides in depth
                                    2nd int: length of strides in hexagonal base

    Examples::

        >>> maxpool3d = pytorch_hexagdly.MaxPool3d((1,1), (2,2))
        >>> input = torch.randn(1, 1, 6, 5, 4)
        >>> output = maxpool3d(input)
        >>> print(output)
    """

    def __init__(self, kernel_size=1, stride=1):
        super(MaxPool3d, self).__init__()
        _check_positive_int_or_pair("kernel_size", kernel_size)
        _check_positive_int_or_pair("stride", stride)
        if isinstance(kernel_size, int):
            self.hexbase_size = kernel_size
            self.depth_size = kernel_size
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "Too many parameters"
            self.hexbase_size = kernel_size[1]
            self.depth_size = kernel_size[0]
        if isinstance(stride, int):
            self.hexbase_stride = stride
            self.depth_stride = stride
        elif isinstance(stride, tuple):
            assert len(stride) == 2, "Too many parameters"
            self.hexbase_stride = stride[1]
            self.depth_stride = stride[0]
        self.dimensions = 3
        self.process = _max_pool3d
        self.combine = torch.max

        for i in range(self.hexbase_size + 1):
            setattr(
                self,
                "kernel" + str(i),
                (self.depth_size, 1 + 2 * self.hexbase_size - i, 1 if i == 0 else 2),
            )

    def forward(self, input):
        if self.hexbase_stride == 1:
            return self.operation_with_single_hexbase_stride(input)
        else:
            return self.operation_with_arbitrary_stride(input)

    def __repr__(self):
        s = (
            "{name}(kernel_size=({depth_size}, {hexbase_size})"
            ", stride=({depth_stride}, {hexbase_stride}))"
        )
        return s.format(name=self.__class__.__name__, **self.__dict__)
