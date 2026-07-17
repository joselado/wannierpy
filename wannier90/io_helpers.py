"""Readers for the standard Wannier90 overlap/projection/eigenvalue files
(``.mmn``, ``.amn``, ``.eig``), a ``.win`` writer, and a small
reciprocal-lattice helper.

The ``.mmn``/``.amn``/``.eig`` files are what a DFT code's
``pw2wannier90``-style interface produces; this module does not generate
them, only parses them into the numpy arrays :func:`wannier90.api.run`
expects (or you can build those arrays yourself in memory and skip this
module entirely -- nothing in :mod:`wannier90.api` requires them to have
come from a file). The parsing mirrors
``test-suite/library-mode-test/test_library.F90`` in the upstream Wannier90
source tree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def reciprocal_lattice(real_lattice: np.ndarray) -> np.ndarray:
    """Reciprocal lattice vectors (rows), in 2*pi/[real_lattice units].

    ``real_lattice[i]`` is the i-th direct lattice vector (row convention,
    matching ``real_lattice_loc``/``recip_lattice_loc`` in wannier_lib.F90).
    """
    a1, a2, a3 = real_lattice
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    return np.array([b1, b2, b3], dtype=np.float64)


def read_eig(path, num_bands: int, num_kpts: int) -> np.ndarray:
    """Read a ``.eig``/``EIG`` file into an (num_bands, num_kpts) array."""
    values = [float(x) for x in open(path).read().split()]
    if len(values) != num_bands * num_kpts:
        raise ValueError(
            f"{path}: expected {num_bands * num_kpts} eigenvalues "
            f"({num_bands} bands x {num_kpts} kpts), found {len(values)}"
        )
    return np.array(values, dtype=np.float64).reshape(num_kpts, num_bands).T


def read_amn(path, num_bands: int, num_kpts: int, num_wann: int) -> np.ndarray:
    """Read a ``.amn`` file into an (num_bands, num_wann, num_kpts) array."""
    A = np.zeros((num_bands, num_wann, num_kpts), dtype=np.complex128)
    with open(path) as f:
        f.readline()  # comment line
        nb, nkp, nw = map(int, f.readline().split())
        if (nb, nkp, nw) != (num_bands, num_kpts, num_wann):
            raise ValueError(
                f"{path}: header ({nb},{nkp},{nw}) does not match expected "
                f"({num_bands},{num_kpts},{num_wann})"
            )
        for line in f:
            if not line.strip():
                continue
            m, n, k, re_, im_ = line.split()
            A[int(m) - 1, int(n) - 1, int(k) - 1] = complex(float(re_), float(im_))
    return A


def read_mmn(
    path,
    num_bands: int,
    num_kpts: int,
    nntot: int,
    nnlist: np.ndarray,
    nncell: np.ndarray,
) -> np.ndarray:
    """Read a ``.mmn`` file into an (num_bands, num_bands, nntot, num_kpts) array.

    ``nnlist``/``nncell`` are the k-point neighbour tables returned by
    :func:`wannier90.api.setup` -- the overlap for each (k, b) pair in the
    ``.mmn`` file is matched against them to find its neighbour index,
    exactly as ``wannier_lib.F90``'s reference caller does.
    """
    M = np.zeros((num_bands, num_bands, nntot, num_kpts), dtype=np.complex128)
    with open(path) as f:
        f.readline()  # comment line
        nb, nkp, nn = map(int, f.readline().split())
        if (nb, nkp, nn) != (num_bands, num_kpts, nntot):
            raise ValueError(
                f"{path}: header ({nb},{nkp},{nn}) does not match expected "
                f"({num_bands},{num_kpts},{nntot})"
            )
        for _ in range(num_kpts * nntot):
            nkp1, nkp2, nnl, nnm, nnn = map(int, f.readline().split())
            block = np.empty((num_bands, num_bands), dtype=np.complex128)
            for n in range(num_bands):
                for m in range(num_bands):
                    re_, im_ = map(float, f.readline().split())
                    block[m, n] = complex(re_, im_)
            nn_idx = None
            for inn in range(nntot):
                if nkp2 == nnlist[nkp1 - 1, inn] and (nnl, nnm, nnn) == tuple(
                    int(x) for x in nncell[:, nkp1 - 1, inn]
                ):
                    nn_idx = inn
                    break
            if nn_idx is None:
                raise ValueError(
                    f"{path}: no matching neighbour for k-point {nkp1} -> {nkp2} "
                    f"cell ({nnl},{nnm},{nnn}); nnlist/nncell must come from "
                    f"the matching wannier90.setup() call"
                )
            M[:, :, nn_idx, nkp1 - 1] = block
    return M


def format_exclude_bands(bands) -> str:
    """Format band indices as wannier90's ``exclude_bands`` range syntax
    (e.g. ``"1-5,10"``). ``bands`` is an iterable of 1-indexed band numbers;
    a string is passed through unchanged (already-formatted wannier90
    syntax)."""
    if isinstance(bands, str):
        return bands
    bands = sorted({int(b) for b in bands})
    if not bands:
        return ""
    ranges = []
    start = prev = bands[0]
    for b in bands[1:]:
        if b == prev + 1:
            prev = b
            continue
        ranges.append((start, prev))
        start = prev = b
    ranges.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def _format_win_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_win(path, keywords=None, exclude_bands=None, projections=None) -> None:
    """Write a ``.win`` file from Python values, for calculations that
    don't want to hand-author one on disk.

    In library mode wannier90 reads (and then *ignores*) ``mp_grid``,
    ``num_bands``, and the ``unit_cell_cart``/``atoms_frac``/``kpoints``
    blocks from ``.win`` -- those come from :func:`wannier90.api.setup`'s
    own arguments instead (src/parameters.F90 prints "Ignoring <mp_grid> in
    input file" etc. and always prefers the argument value). So none of
    that needs to go in here. ``num_wann`` is the one keyword wannier90
    requires unconditionally, library mode or not -- put it in ``keywords``.

    Parameters
    ----------
    keywords : dict, optional
        Plain ``key = value`` lines (``num_wann``, ``num_iter``,
        ``dis_win_max``, ``conv_tol``, ...). ``bool`` values become
        ``true``/``false``; everything else is stringified as-is.
    exclude_bands : optional
        Passed through :func:`format_exclude_bands`.
    projections : list[str], optional
        Raw lines for the ``begin projections`` / ``end projections``
        block, in wannier90's own syntax (e.g. ``"f=0.25,0.25,0.25 : s"``).
    """
    lines = [f"{key} = {_format_win_value(value)}" for key, value in (keywords or {}).items()]
    if exclude_bands is not None:
        lines.append(f"exclude_bands : {format_exclude_bands(exclude_bands)}")
    if projections:
        lines.append("")
        lines.append("begin projections")
        lines.extend(projections)
        lines.append("end projections")
    Path(path).write_text("\n".join(lines) + "\n")
