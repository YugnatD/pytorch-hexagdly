"""
Tests for share_neighbors weight-sharing modes using a pure-Python oracle.

The oracle is built from the hex neighbor offsets derived by firing single-pixel
impulses through the library and recording exactly which input pixel each kernel
cell reads for each output column parity. This gives us an independent reference
implementation that computes expected outputs without using the weight
materialization path at all.

Hexagonal neighbor offsets (offsets are relative to OUTPUT pixel, verified by impulse):

    Even column center (col % 2 == 0):    Odd column center (col % 2 == 1):
        (-1,-1) (-1,0) (-1,+1)                    (-1,0)
         (0,-1) CENTER  (0,+1)            (0,-1)  CENTER  (0,+1)
                (+1,0)                    (+1,-1)  (+1,0)  (+1,+1)

Non-shared sub-kernel cell → input offset (dr,dc) per output column parity:
    kernel0[r=0]: both parities → (-1, 0)   ← pixel above center
    kernel0[r=1]: both parities → ( 0, 0)   ← center
    kernel0[r=2]: both parities → (+1, 0)   ← pixel below center
    kernel1[r=0,c=0]: parity=0 → (-1,-1),  parity=1 → ( 0,-1)
    kernel1[r=0,c=1]: parity=0 → (-1,+1),  parity=1 → ( 0,+1)
    kernel1[r=1,c=0]: parity=0 → ( 0,-1),  parity=1 → (+1,-1)
    kernel1[r=1,c=1]: parity=0 → ( 0,+1),  parity=1 → (+1,+1)

Shared-weight group assignment per mode (kernel_size=1):

    ring  (2 weights):
        all 6 neighbors = group 1,  center = group 0

    diag  (4 weights) — antipodal pairs:
        even col: (-1,-1)=1, (-1,0)=3, (-1,+1)=2, (0,-1)=2, (0,+1)=1, (+1,0)=3
        odd  col: (-1,0)=3,  (0,-1)=1, (0,+1)=2,  (+1,-1)=2,(+1,0)=3, (+1,+1)=1

    sym   (4 weights) — adjacent 60° pairs:
        even col: (-1,-1)=2, (-1,0)=2, (-1,+1)=1, (0,-1)=3, (0,+1)=1, (+1,0)=3
        odd  col: (-1,0)=2,  (0,-1)=2, (0,+1)=1,  (+1,-1)=3,(+1,0)=3, (+1,+1)=1
"""

import torch

import pytorch_hexagdly as ph

# ---------------------------------------------------------------------------
# Neighbor tables — verified by impulse response (kernel_size=1)
# ---------------------------------------------------------------------------
#
# HexagDLy stores hex data in a square tensor where odd columns (col%2==1)
# are shifted DOWN by half a cell relative to even columns:
#
#   col:   0     1     2     3
#         ___         ___
#        / A \       / C \        ← even cols (0, 2) are "up"
#   ___  \___/  ___  \___/
#  / B \       / D \             ← odd cols  (1, 3) are "down"
#  \___/       \___/
#
# Because of this shift, the 6 neighbors of a center pixel depend on whether
# it sits in an even or odd column:
#
#   EVEN column center:          ODD column center:
#
#       TL  T  TR                       T
#        L [C]  R                  L   [C]   R
#            B                    BL   B    BR
#
#   where the offsets (dr = row delta, dc = col delta) are:
#
#   position   even (col%2==0)   odd (col%2==1)
#   ────────   ───────────────   ──────────────
#   TL         (-1, -1)          — (no top-left for odd)
#   T          (-1,  0)          (-1,  0)
#   TR         (-1, +1)          — (no top-right for odd)
#   L          ( 0, -1)          ( 0, -1)
#   R          ( 0, +1)          ( 0, +1)
#   BL         — (no bot-left)   (+1, -1)
#   B          (+1,  0)          (+1,  0)
#   BR         — (no bot-right)  (+1, +1)
#
# ---------------------------------------------------------------------------
# Group assignments per sharing mode (each number = one shared weight):
#
#   ring (2 weights)        diag (4 weights)        sym (4 weights)
#   center=0, ring=1        center=0, 3 axis-pairs   center=0, 3 adj-pairs
#
#   EVEN:                   EVEN:                   EVEN:
#       1   1   1               1   3   2               2   2   1
#       1  [0]  1               2  [0]  1               3  [0]  1
#           1                       3                       3
#
#   ODD:                    ODD:                    ODD:
#           1                       3                       2
#       1  [0]  1               1  [0]  2               2  [0]  1
#       1   1   1               2   3   1               3   3   1
#
# ---------------------------------------------------------------------------

# fmt: off
_RING_NEIGHBORS = {
    # even col: TL=1, T=1, TR=1, L=1, R=1, B=1
    0: {(-1,-1):1, (-1, 0):1, (-1,+1):1,
        ( 0,-1):1,             ( 0,+1):1,
                   (+1, 0):1},
    # odd col:  T=1, L=1, R=1, BL=1, B=1, BR=1
    1: {           (-1, 0):1,
        ( 0,-1):1,             ( 0,+1):1,
        (+1,-1):1, (+1, 0):1, (+1,+1):1},
}

_DIAG_NEIGHBORS = {
    # even col: TL=1, T=3, TR=2, L=2, R=1, B=3
    0: {(-1,-1):1, (-1, 0):3, (-1,+1):2,
        ( 0,-1):2,             ( 0,+1):1,
                   (+1, 0):3},
    # odd col:  T=3, L=1, R=2, BL=2, B=3, BR=1
    1: {           (-1, 0):3,
        ( 0,-1):1,             ( 0,+1):2,
        (+1,-1):2, (+1, 0):3, (+1,+1):1},
}

_SYM_NEIGHBORS = {
    # even col: TL=2, T=2, TR=1, L=3, R=1, B=3
    0: {(-1,-1):2, (-1, 0):2, (-1,+1):1,
        ( 0,-1):3,             ( 0,+1):1,
                   (+1, 0):3},
    # odd col:  T=2, L=2, R=1, BL=3, B=3, BR=1
    1: {           (-1, 0):2,
        ( 0,-1):2,             ( 0,+1):1,
        (+1,-1):3, (+1, 0):3, (+1,+1):1},
}
# fmt: on

# ---------------------------------------------------------------------------
# Non-shared offset table — verified by impulse response (kernel_size=1)
# ---------------------------------------------------------------------------
#
# In non-shared mode, each sub-kernel cell has an independent weight.
# sub0 is a 3×1 column (center axis); sub1 is a 2×2 off-axis block.
#
# sub0 is parity-independent — it always reads the center column:
#
#   kernel0[row=0] → T  (one row above, same column)
#   kernel0[row=1] → C  (center pixel)
#   kernel0[row=2] → B  (one row below, same column)
#
# sub1 shifts by 1 row when the output center is in an odd column
# (same asymmetry as the diag/sym neighbor offset change above):
#
#   cell             even col (parity=0)   odd col (parity=1)
#   ──────────────   ───────────────────   ──────────────────
#   kernel1[r=0,c=0]   TL  (-1,-1)          L   ( 0,-1)
#   kernel1[r=0,c=1]   TR  (-1,+1)          R   ( 0,+1)
#   kernel1[r=1,c=0]   L   ( 0,-1)          BL  (+1,-1)
#   kernel1[r=1,c=1]   R   ( 0,+1)          BR  (+1,+1)
#
# fmt: off
_NOSHARE_OFFSETS = {
    # sub0 — same for both parities
    (0, 0, 0): {0: (-1, 0), 1: (-1, 0)},   # T  above center
    (0, 1, 0): {0: ( 0, 0), 1: ( 0, 0)},   # C  center
    (0, 2, 0): {0: (+1, 0), 1: (+1, 0)},   # B  below center
    # sub1 — shifts one row down for odd-column centres
    (1, 0, 0): {0: (-1,-1), 1: ( 0,-1)},   # even→TL  odd→L
    (1, 0, 1): {0: (-1,+1), 1: ( 0,+1)},   # even→TR  odd→R
    (1, 1, 0): {0: ( 0,-1), 1: (+1,-1)},   # even→L   odd→BL
    (1, 1, 1): {0: ( 0,+1), 1: (+1,+1)},   # even→R   odd→BR
}
# fmt: on

# ---------------------------------------------------------------------------
# kernel_size=2 neighbor tables — verified by impulse response
# ---------------------------------------------------------------------------
#
# A kernel_size=2 hex kernel covers 19 cells (rings 0, 1, 2).
# The 5×5 offset grid below shows which group each cell belongs to.
# '.' = position not covered by this kernel size.
#
#   ring (3 weights: center=0, ring-1=1, ring-2=2)
#   From user-validated ASCII:
#           [ B ]         ring-1 (A=1): (-1,-1),(-1,0),(-1,+1),(0,-1),(0,+1),(+1,0)
#        [ B ] [ B ]      ring-2 (B=2): all 12 other cells
#     [ B ] [ A ] [ B ]
#        [ A ] [ A ]
#     [ B ] [ X ] [ B ]
#        [ A ] [ A ]
#     [ B ] [ A ] [ B ]
#        [ B ] [ B ]
#           [ B ]
#
#   even:  .  2  2  2  .        odd:  .  .  2  .  .
#          2  1  1  1  2              2  1  1  1  2
#          2  1 [0] 1  2              2  1 [0] 1  2
#          2  2  1  2  2              2  2  1  2  2
#          .  .  2  .  .              .  2  2  2  .
#
# fmt: off
_RING2_NEIGHBORS = {
    0: {(-2,-1):2, (-2, 0):2, (-2,+1):2,
        (-1,-2):2, (-1,-1):1, (-1, 0):1, (-1,+1):1, (-1,+2):2,
        ( 0,-2):2, ( 0,-1):1, ( 0, 0):0, ( 0,+1):1, ( 0,+2):2,
        (+1,-2):2, (+1,-1):2, (+1, 0):1, (+1,+1):2, (+1,+2):2,
                              (+2, 0):2},
    1: {                      (-2, 0):2,
        (-1,-2):2, (-1,-1):1, (-1, 0):1, (-1,+1):1, (-1,+2):2,
        ( 0,-2):2, ( 0,-1):1, ( 0, 0):0, ( 0,+1):1, ( 0,+2):2,
        (+1,-2):2, (+1,-1):2, (+1, 0):1, (+1,+1):2, (+1,+2):2,
                   (+2,-1):2, (+2, 0):2, (+2,+1):2},
}
# fmt: on

#   diag (10 weights: center=0, visual-antipodal pairs 1..9)
#   Pairs from pointy-top ASCII (same letter = same group):
#           [ E ]         E=1: (+2,0)<->(-2,0)
#        [ F ] [ I ]      F=2: (+1,-1)<->(-2,+1)    I=3: (+1,+1)<->(-2,-1)
#     [ H ] [ C ] [ G ]   H=4: (+1,-2)<->(-1,+2)    C=5: (+1,0)<->(-1,0)   G=6: (+1,+2)<->(-1,-2)
#        [ B ] [ A ]      B=7: (0,-1)<->(-1,+1)     A=8: (0,+1)<->(-1,-1)
#     [ D ] [ X ] [ D ]   D=9: (0,-2)<->(0,+2)
#        [ A ] [ B ]
#     [ G ] [ C ] [ H ]
#        [ I ] [ F ]
#           [ E ]
#
#   even:  .  3  1  2  .        odd:  .  .  1  .  .
#          6  8  5  7  4              6  2  5  3  4
#          9  7 [0] 8  9              9  7 [0] 8  9
#          4  2  5  3  6              4  8  5  7  6
#          .  .  1  .  .              .  3  1  2  .
#
# fmt: off
_DIAG2_NEIGHBORS = {
    0: {(-2,-1):3, (-2, 0):1, (-2,+1):2,
        (-1,-2):6, (-1,-1):8, (-1, 0):5, (-1,+1):7, (-1,+2):4,
        ( 0,-2):9, ( 0,-1):7, ( 0, 0):0, ( 0,+1):8, ( 0,+2):9,
        (+1,-2):4, (+1,-1):2, (+1, 0):5, (+1,+1):3, (+1,+2):6,
                              (+2, 0):1},
    1: {                      (-2, 0):1,
        (-1,-2):6, (-1,-1):8, (-1, 0):5, (-1,+1):7, (-1,+2):4,
        ( 0,-2):9, ( 0,-1):7, ( 0, 0):0, ( 0,+1):8, ( 0,+2):9,
        (+1,-2):4, (+1,-1):2, (+1, 0):5, (+1,+1):3, (+1,+2):6,
                   (+2,-1):3, (+2, 0):1, (+2,+1):2},
}
# fmt: on

#   sym (10 weights: center=0, adjacent pairs from user-validated ASCII)
#           [ G ]         G=7: (+2,0)<->(+1,-1)
#        [ G ] [ F ]      F=6: (+1,+1)<->(+1,+2)
#     [ H ] [ A ] [ F ]   H=8: (+1,-2)<->(0,-2)    A=1: (+1,0)<->(0,-1)
#        [ A ] [ B ]      B=2: (0,+1)<->(-1,+1)
#     [ H ] [ X ] [ E ]   E=5: (0,+2)<->(-1,+2)
#        [ C ] [ B ]      C=3: (-1,-1)<->(-1,0)
#     [ I ] [ C ] [ E ]   I=9: (-1,-2)<->(-2,-1)[even] / (-1,-2)<->(+2,-1)[odd]
#        [ I ] [ D ]      D=4: (-2,+1)<->(-2,0)[even] / (+2,+1)<->(-2,0)[odd]
#           [ D ]
#
#   even:  .  9  4  4  .        odd:  .  .  4  .  .
#          9  3  3  2  5              9  3  3  2  5
#          8  1 [0] 2  5              8  1 [0] 2  5
#          8  7  1  6  6              8  7  1  6  6
#          .  .  7  .  .              .  9  7  4  .
#
# fmt: off
_SYM2_NEIGHBORS = {
    0: {(-2,-1):9, (-2, 0):4, (-2,+1):4,
        (-1,-2):9, (-1,-1):3, (-1, 0):3, (-1,+1):2, (-1,+2):5,
        ( 0,-2):8, ( 0,-1):1, ( 0, 0):0, ( 0,+1):2, ( 0,+2):5,
        (+1,-2):8, (+1,-1):7, (+1, 0):1, (+1,+1):6, (+1,+2):6,
                              (+2, 0):7},
    1: {                      (-2, 0):4,
        (-1,-2):9, (-1,-1):3, (-1, 0):3, (-1,+1):2, (-1,+2):5,
        ( 0,-2):8, ( 0,-1):1, ( 0, 0):0, ( 0,+1):2, ( 0,+2):5,
        (+1,-2):8, (+1,-1):7, (+1, 0):1, (+1,+1):6, (+1,+2):6,
                   (+2,-1):9, (+2, 0):7, (+2,+1):4},
}
# fmt: on

# ---------------------------------------------------------------------------
# Non-shared offset table for kernel_size=2 — verified by impulse response
# ---------------------------------------------------------------------------
#
# sub0 (5×1) and sub2 (3×2) are parity-independent.
# sub1 (4×2) shifts by exactly 1 row for odd-column centres (same rule as k=1).
#
#   cell             even col (parity=0)   odd col (parity=1)
#   ──────────────   ───────────────────   ──────────────────
#   kernel0[r=0]     (-2, 0)               (-2, 0)
#   kernel0[r=1]     (-1, 0)               (-1, 0)
#   kernel0[r=2]     ( 0, 0) center        ( 0, 0) center
#   kernel0[r=3]     (+1, 0)               (+1, 0)
#   kernel0[r=4]     (+2, 0)               (+2, 0)
#   kernel1[r=0,c=0] (-2,-1)               (-1,-1)  ← shift
#   kernel1[r=0,c=1] (-2,+1)               (-1,+1)  ← shift
#   kernel1[r=1,c=0] (-1,-1)               ( 0,-1)  ← shift
#   kernel1[r=1,c=1] (-1,+1)               ( 0,+1)  ← shift
#   kernel1[r=2,c=0] ( 0,-1)               (+1,-1)  ← shift
#   kernel1[r=2,c=1] ( 0,+1)               (+1,+1)  ← shift
#   kernel1[r=3,c=0] (+1,-1)               (+2,-1)  ← shift
#   kernel1[r=3,c=1] (+1,+1)               (+2,+1)  ← shift
#   kernel2[r=0,c=0] (-1,-2)               (-1,-2)
#   kernel2[r=0,c=1] (-1,+2)               (-1,+2)
#   kernel2[r=1,c=0] ( 0,-2)               ( 0,-2)
#   kernel2[r=1,c=1] ( 0,+2)               ( 0,+2)
#   kernel2[r=2,c=0] (+1,-2)               (+1,-2)
#   kernel2[r=2,c=1] (+1,+2)               (+1,+2)
#
# fmt: off
_NOSHARE2_OFFSETS = {
    # sub0 — parity-independent
    (0, 0, 0): {0: (-2, 0), 1: (-2, 0)},
    (0, 1, 0): {0: (-1, 0), 1: (-1, 0)},
    (0, 2, 0): {0: ( 0, 0), 1: ( 0, 0)},
    (0, 3, 0): {0: (+1, 0), 1: (+1, 0)},
    (0, 4, 0): {0: (+2, 0), 1: (+2, 0)},
    # sub1 — shifts one row down for odd-column centres
    (1, 0, 0): {0: (-2,-1), 1: (-1,-1)},
    (1, 0, 1): {0: (-2,+1), 1: (-1,+1)},
    (1, 1, 0): {0: (-1,-1), 1: ( 0,-1)},
    (1, 1, 1): {0: (-1,+1), 1: ( 0,+1)},
    (1, 2, 0): {0: ( 0,-1), 1: (+1,-1)},
    (1, 2, 1): {0: ( 0,+1), 1: (+1,+1)},
    (1, 3, 0): {0: (+1,-1), 1: (+2,-1)},
    (1, 3, 1): {0: (+1,+1), 1: (+2,+1)},
    # sub2 — parity-independent
    (2, 0, 0): {0: (-1,-2), 1: (-1,-2)},
    (2, 0, 1): {0: (-1,+2), 1: (-1,+2)},
    (2, 1, 0): {0: ( 0,-2), 1: ( 0,-2)},
    (2, 1, 1): {0: ( 0,+2), 1: ( 0,+2)},
    (2, 2, 0): {0: (+1,-2), 1: (+1,-2)},
    (2, 2, 1): {0: (+1,+2), 1: (+1,+2)},
}
# fmt: on


# ---------------------------------------------------------------------------
# Pure-Python oracles
# ---------------------------------------------------------------------------


def oracle_k2(grid, weights, neighbor_table):
    """Same as oracle() but works for any kernel size — neighbor_table already
    contains all offsets including ring-2."""
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            for (dr, dc), g in neighbor_table[parity].items():
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[g] * grid[ir][ic]
    return out


def oracle_noshare_k2(grid, weights):
    """oracle_noshare() for kernel_size=2 using _NOSHARE2_OFFSETS."""
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            for cell, parity_map in _NOSHARE2_OFFSETS.items():
                dr, dc = parity_map[parity]
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[cell] * grid[ir][ic]
    return out


def oracle(grid, weights, neighbor_table):
    """
    Compute hexagonal conv output for a 2-D grid using the given neighbor table.

    Args:
        grid:            list[list[float]] — input values, shape (rows, cols)
        weights:         dict[int, float]  — {group_index: weight_value}
        neighbor_table:  dict keyed by col_parity (0 or 1), each value is a
                         dict of {(dr, dc): group_index} for the 6 neighbors

    Returns:
        list[list[float]] — output values, same shape as grid
    """
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            # center — always group 0
            out[row][col] += weights[0] * grid[row][col]
            # neighbors
            for (dr, dc), g in neighbor_table[parity].items():
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[g] * grid[ir][ic]
    return out


def oracle_noshare(grid, weights):
    """
    Compute hexagonal conv output for non-shared mode (kernel_size=1).

    Args:
        grid:    list[list[float]] — input values, shape (rows, cols)
        weights: dict[(sub_i, row, col), float] — one weight per kernel cell

    Returns:
        list[list[float]] — output values, same shape as grid
    """
    n_rows, n_cols = len(grid), len(grid[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    for row in range(n_rows):
        for col in range(n_cols):
            parity = col % 2
            for cell, parity_map in _NOSHARE_OFFSETS.items():
                dr, dc = parity_map[parity]
                ir, ic = row + dr, col + dc
                if 0 <= ir < n_rows and 0 <= ic < n_cols:
                    out[row][col] += weights[cell] * grid[ir][ic]
    return out


def to_tensor(grid):
    return torch.tensor([grid], dtype=torch.float32).unsqueeze(0)  # (1,1,H,W)


def make_conv(mode, kernel_size=1):
    return ph.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False, share_neighbors=mode)


def make_conv_noshare(kernel_size=1):
    return ph.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=False)


def set_weights(conv, weights):
    with torch.no_grad():
        for g, w in weights.items():
            conv.shared_weights[0, 0, g] = w


def set_weights_noshare(conv, weights):
    """Set independent kernel weights. weights keyed by (sub_i, row, col)."""
    with torch.no_grad():
        for (sub_i, r, c), w in weights.items():
            getattr(conv, f"kernel{sub_i}")[0, 0, r, c] = w


# ---------------------------------------------------------------------------
# Test grids
# ---------------------------------------------------------------------------

# Small 3x2 grid — easy to trace by hand
GRID_SMALL = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

# Larger 5x4 grid — exercises interior pixels that have all 6 neighbors
# Values chosen to make every group contribution distinct
GRID_LARGE = [
    [2.0, 1.0, 5.0, 4.0],
    [4.0, 3.0, 2.0, 9.0],
    [2.0, 7.0, 1.0, 1.0],
    [2.0, 4.0, 4.0, 9.0],
    [1.0, 9.0, 4.0, 9.0],
]

WEIGHTS_RING = {0: 2.0, 1: 3.0}
WEIGHTS_DIAG = {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}
WEIGHTS_SYM = {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}

# Non-shared k=1: 7 independent weights (3 in sub0 + 4 in sub1).
# Distinct primes so any wrong routing produces a detectably wrong result.
WEIGHTS_NOSHARE = {
    (0, 0, 0): 2.0,  # sub0 top
    (0, 1, 0): 11.0,  # sub0 center
    (0, 2, 0): 3.0,  # sub0 bottom
    (1, 0, 0): 5.0,  # sub1 r0c0
    (1, 0, 1): 7.0,  # sub1 r0c1
    (1, 1, 0): 13.0,  # sub1 r1c0
    (1, 1, 1): 17.0,  # sub1 r1c1
}

# kernel_size=2 grids and weights
# 7x6 grid — interior pixels (rows 2-4, cols 2-3) have all 19 neighbors present
GRID_K2 = [
    [2.0, 1.0, 5.0, 4.0, 3.0, 7.0],
    [4.0, 3.0, 2.0, 9.0, 1.0, 5.0],
    [2.0, 7.0, 1.0, 1.0, 8.0, 2.0],
    [2.0, 4.0, 4.0, 9.0, 3.0, 6.0],
    [1.0, 9.0, 4.0, 9.0, 2.0, 4.0],
    [3.0, 5.0, 2.0, 6.0, 7.0, 1.0],
    [4.0, 2.0, 8.0, 3.0, 1.0, 9.0],
]

# ring k=2: 3 weights
WEIGHTS_RING2 = {0: 2.0, 1: 3.0, 2: 5.0}

# diag/sym k=2: 10 weights — distinct primes to catch any wrong group assignment
WEIGHTS_DIAG2 = {
    0: 2.0,
    1: 3.0,
    2: 5.0,
    3: 7.0,
    4: 11.0,
    5: 13.0,
    6: 17.0,
    7: 19.0,
    8: 23.0,
    9: 29.0,
}
WEIGHTS_SYM2 = {
    0: 2.0,
    1: 3.0,
    2: 5.0,
    3: 7.0,
    4: 11.0,
    5: 13.0,
    6: 17.0,
    7: 19.0,
    8: 23.0,
    9: 29.0,
}

# Non-shared k=2: 19 independent weights (5 in sub0 + 8 in sub1 + 6 in sub2).
WEIGHTS_NOSHARE2 = {
    (0, 0, 0): 2.0,
    (0, 1, 0): 3.0,
    (0, 2, 0): 5.0,  # sub0
    (0, 3, 0): 7.0,
    (0, 4, 0): 11.0,
    (1, 0, 0): 13.0,
    (1, 0, 1): 17.0,  # sub1
    (1, 1, 0): 19.0,
    (1, 1, 1): 23.0,
    (1, 2, 0): 29.0,
    (1, 2, 1): 31.0,
    (1, 3, 0): 37.0,
    (1, 3, 1): 41.0,
    (2, 0, 0): 43.0,
    (2, 0, 1): 47.0,  # sub2
    (2, 1, 0): 53.0,
    (2, 1, 1): 59.0,
    (2, 2, 0): 61.0,
    (2, 2, 1): 67.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRingOracle:
    def _run(self, grid, weights):
        expected = oracle(grid, weights, _RING_NEIGHBORS)
        conv = make_conv("ring")
        set_weights(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_small_grid(self):
        expected, got = self._run(GRID_SMALL, WEIGHTS_RING)
        for r in range(len(GRID_SMALL)):
            for c in range(len(GRID_SMALL[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"ring small [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_large_grid(self):
        expected, got = self._run(GRID_LARGE, WEIGHTS_RING)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"ring large [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_weight_count(self):
        assert make_conv("ring").shared_weights.shape == (1, 1, 2)

    def test_all_neighbors_same_weight(self):
        # With w0=0, w1=1: output[r,c] = number of valid neighbors of (r,c)
        weights = {0: 0.0, 1: 1.0}
        expected, got = self._run(GRID_LARGE, weights)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3


class TestDiagOracle:
    def _run(self, grid, weights):
        expected = oracle(grid, weights, _DIAG_NEIGHBORS)
        conv = make_conv("diag")
        set_weights(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_small_grid(self):
        expected, got = self._run(GRID_SMALL, WEIGHTS_DIAG)
        for r in range(len(GRID_SMALL)):
            for c in range(len(GRID_SMALL[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"diag small [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_large_grid(self):
        expected, got = self._run(GRID_LARGE, WEIGHTS_DIAG)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"diag large [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_weight_count(self):
        assert make_conv("diag").shared_weights.shape == (1, 1, 4)

    def test_antipodal_symmetry(self):
        # With uniform input (all ones) and w0=0, all weights equal:
        # every interior pixel should see 6 neighbors — same sum regardless of parity
        grid = [[1.0] * 6 for _ in range(7)]
        weights = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
        expected, got = self._run(grid, weights)
        # interior pixels (row 1-5, col 1-4) should all equal 6.0
        for r in range(1, 6):
            for c in range(1, 5):
                assert abs(got[r][c] - 6.0) < 1e-3, (
                    f"diag uniform [{r},{c}]: got {got[r][c]:.2f}, expected 6.0"
                )

    def test_gradients_flow_to_all_weights(self):
        conv = make_conv("diag")
        set_weights(conv, WEIGHTS_DIAG)
        conv(to_tensor(GRID_LARGE)).sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).all(), (
            "All diag group weights should receive non-zero gradients"
        )


class TestSymOracle:
    def _run(self, grid, weights):
        expected = oracle(grid, weights, _SYM_NEIGHBORS)
        conv = make_conv("sym")
        set_weights(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_small_grid(self):
        expected, got = self._run(GRID_SMALL, WEIGHTS_SYM)
        for r in range(len(GRID_SMALL)):
            for c in range(len(GRID_SMALL[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"sym small [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_large_grid(self):
        expected, got = self._run(GRID_LARGE, WEIGHTS_SYM)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"sym large [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_weight_count(self):
        assert make_conv("sym").shared_weights.shape == (1, 1, 4)

    def test_gradients_flow_to_all_weights(self):
        conv = make_conv("sym")
        set_weights(conv, WEIGHTS_SYM)
        conv(to_tensor(GRID_LARGE)).sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).all(), (
            "All sym group weights should receive non-zero gradients"
        )


class TestParityCorrectness:
    """Verify that both parity maps are materialized simultaneously and used correctly."""

    def test_kernel1_and_kernel1_odd_differ_for_n2(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="diag")
        conv._materialize_shared_kernels()
        k1_even = conv.kernel1.detach().clone()
        k1_odd = conv.kernel1_odd.detach().clone()
        assert not torch.allclose(k1_even, k1_odd), (
            "kernel1 and kernel1_odd should differ for n=2 diag"
        )

    def test_ring_k2_has_odd_variant(self):
        # ring k=2 maps differ per parity — odd variant must be created
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="ring")
        conv._materialize_shared_kernels()
        assert hasattr(conv, "kernel1_odd"), "ring k=2 should create kernel1_odd"

    def test_forward_produces_finite_output_even_width(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="diag")
        out = conv(torch.randn(1, 1, 21, 16))
        assert out.isfinite().all() and out.shape == (1, 1, 21, 16)

    def test_forward_produces_finite_output_odd_width(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="diag")
        out = conv(torch.randn(1, 1, 21, 17))
        assert out.isfinite().all() and out.shape == (1, 1, 21, 17)


class TestNoShareOracle:
    """
    Tests for the standard (non-shared) Conv2d using the independent oracle.

    The oracle uses _NOSHARE_OFFSETS: a table mapping each sub-kernel cell
    (sub_i, row, col) to the input offset it reads, per output column parity.
    This is entirely independent of the library's forward pass implementation.
    """

    def _run(self, grid, weights):
        expected = oracle_noshare(grid, weights)
        conv = make_conv_noshare()
        set_weights_noshare(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_small_grid(self):
        expected, got = self._run(GRID_SMALL, WEIGHTS_NOSHARE)
        for r in range(len(GRID_SMALL)):
            for c in range(len(GRID_SMALL[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"noshare small [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_large_grid(self):
        expected, got = self._run(GRID_LARGE, WEIGHTS_NOSHARE)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"noshare large [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
                )

    def test_center_only(self):
        # With only the center cell active, output == input scaled by center weight
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE}
        weights[(0, 1, 0)] = 5.0  # center cell only
        expected, got = self._run(GRID_LARGE, weights)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - 5.0 * GRID_LARGE[r][c]) < 1e-3, (
                    f"center-only [{r},{c}]: got {got[r][c]:.2f}, "
                    f"expected {5.0 * GRID_LARGE[r][c]:.2f}"
                )

    def test_parity_routing_sub1(self):
        # Activate sub1[r=0,c=0] only. For even-col output it reads (-1,-1),
        # for odd-col output it reads (0,-1). These are different input pixels
        # so a wrong parity routing produces a different number.
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE}
        weights[(1, 0, 0)] = 1.0
        expected, got = self._run(GRID_LARGE, weights)
        for r in range(len(GRID_LARGE)):
            for c in range(len(GRID_LARGE[0])):
                assert abs(got[r][c] - expected[r][c]) < 1e-3, (
                    f"parity sub1[r=0,c=0] [{r},{c}]: got {got[r][c]:.2f}, "
                    f"expected {expected[r][c]:.2f}"
                )

    def test_gradients_flow(self):
        conv = make_conv_noshare()
        set_weights_noshare(conv, WEIGHTS_NOSHARE)
        conv(to_tensor(GRID_LARGE)).sum().backward()
        for sub_i, n_r, n_c in [(0, 3, 1), (1, 2, 2)]:
            k = getattr(conv, f"kernel{sub_i}")
            assert k.grad is not None
            assert (k.grad.abs() > 0).all(), (
                f"kernel{sub_i} should have non-zero gradients everywhere"
            )


# ---------------------------------------------------------------------------
# kernel_size=2 tests
# ---------------------------------------------------------------------------


def _check_grid(expected, got, label, tol=1e-2):
    for r in range(len(expected)):
        for c in range(len(expected[0])):
            assert abs(got[r][c] - expected[r][c]) < tol, (
                f"{label} [{r},{c}]: got {got[r][c]:.2f}, expected {expected[r][c]:.2f}"
            )


class TestRingOracleK2:
    def _run(self, grid, weights):
        expected = oracle_k2(grid, weights, _RING2_NEIGHBORS)
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="ring")
        set_weights(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_grid(self):
        e, g = self._run(GRID_K2, WEIGHTS_RING2)
        _check_grid(e, g, "ring k=2")

    def test_weight_count(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="ring")
        assert conv.shared_weights.shape == (1, 1, 3)

    def test_gradients_flow(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="ring")
        set_weights(conv, WEIGHTS_RING2)
        conv(to_tensor(GRID_K2)).sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).all()


class TestDiagOracleK2:
    def _run(self, grid, weights):
        expected = oracle_k2(grid, weights, _DIAG2_NEIGHBORS)
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="diag")
        set_weights(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_grid(self):
        e, g = self._run(GRID_K2, WEIGHTS_DIAG2)
        _check_grid(e, g, "diag k=2")

    def test_weight_count(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="diag")
        assert conv.shared_weights.shape == (1, 1, 10)

    def test_gradients_flow(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="diag")
        set_weights(conv, WEIGHTS_DIAG2)
        conv(to_tensor(GRID_K2)).sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).all()


class TestSymOracleK2:
    def _run(self, grid, weights):
        expected = oracle_k2(grid, weights, _SYM2_NEIGHBORS)
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="sym")
        set_weights(conv, weights)
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_grid(self):
        e, g = self._run(GRID_K2, WEIGHTS_SYM2)
        _check_grid(e, g, "sym k=2")

    def test_weight_count(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="sym")
        assert conv.shared_weights.shape == (1, 1, 10)

    def test_gradients_flow(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False, share_neighbors="sym")
        set_weights(conv, WEIGHTS_SYM2)
        conv(to_tensor(GRID_K2)).sum().backward()
        assert conv.shared_weights.grad is not None
        assert (conv.shared_weights.grad.abs() > 0).all()


class TestNoShareOracleK2:
    def _run(self, grid, weights):
        expected = oracle_noshare_k2(grid, weights)
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False)
        with torch.no_grad():
            for (si, r, c), w in weights.items():
                getattr(conv, f"kernel{si}")[0, 0, r, c] = w
        got = conv(to_tensor(grid))[0, 0].detach().tolist()
        return expected, got

    def test_grid(self):
        e, g = self._run(GRID_K2, WEIGHTS_NOSHARE2)
        _check_grid(e, g, "noshare k=2")

    def test_center_only(self):
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE2}
        weights[(0, 2, 0)] = 5.0  # center cell (sub0 row=2)
        e, g = self._run(GRID_K2, weights)
        for r in range(len(GRID_K2)):
            for c in range(len(GRID_K2[0])):
                assert abs(g[r][c] - 5.0 * GRID_K2[r][c]) < 1e-3

    def test_parity_routing_sub1(self):
        # sub1[r=1,c=0]: even→(-1,-1), odd→(0,-1) — different input pixels
        weights = {k: 0.0 for k in WEIGHTS_NOSHARE2}
        weights[(1, 1, 0)] = 1.0
        e, g = self._run(GRID_K2, weights)
        _check_grid(e, g, "noshare k=2 parity sub1[1,0]")

    def test_gradients_flow(self):
        conv = ph.Conv2d(1, 1, kernel_size=2, stride=1, bias=False)
        with torch.no_grad():
            for (si, r, c), w in WEIGHTS_NOSHARE2.items():
                getattr(conv, f"kernel{si}")[0, 0, r, c] = w
        conv(to_tensor(GRID_K2)).sum().backward()
        for si in range(3):
            k = getattr(conv, f"kernel{si}")
            assert k.grad is not None
            assert (k.grad.abs() > 0).all(), f"kernel{si} should have non-zero gradients"
