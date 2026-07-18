#!/usr/bin/env python3
"""Build the optional Fortran backend (``wannier90._wannier90``) and drop it
into the installed (or in-place) ``wannier90/`` package directory.

The PyPI release of this package (``pip install wannierpy``) is pure Python
-- it never runs this. Use this script if you specifically want
``backend="fortran"``, which requires:

* a local checkout of this repo (this script isn't shipped in the PyPI
  sdist/wheel -- get it from git),
* a gfortran + LAPACK/BLAS toolchain (e.g. on Debian/Ubuntu:
  ``apt install gfortran libblas-dev liblapack-dev``),
* the Wannier90 3.1.0 source tree, found in this order: the
  ``WANNIER90_SRC`` environment variable, ``vendor/wannier90-3.1.0/`` next
  to this repo, or a ``wannier90-3.1.0/`` sibling directory next to it.

Usage::

    pip install -e .                        # editable install of the pure-Python package first
    WANNIER90_SRC=/path/to/wannier90-3.1.0 python scripts/build_fortran_extension.py

This is the same build ``setup.py`` ran automatically before the package
switched to a pure-Python-by-default PyPI release -- see git history for
that version if you're looking for the previous all-in-one behaviour.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
VENDORED_SRC = HERE / "vendor" / "wannier90-3.1.0"


def _find_wannier90_root() -> Path:
    env_root = os.environ.get("WANNIER90_SRC")
    if env_root:
        return Path(env_root).resolve()
    if VENDORED_SRC.exists():
        return VENDORED_SRC
    sibling = HERE.parent / "wannier90-3.1.0"
    if sibling.exists():
        return sibling
    raise RuntimeError(
        "Could not find the Wannier90 3.1.0 source tree. Set the WANNIER90_SRC "
        f"environment variable, or vendor it at {VENDORED_SRC}."
    )


def _ensure_pic_make_inc(root: Path) -> None:
    """Write a make.inc with -fPIC (needed to link into a shared extension)
    and -DEXIT_FLAG (so Fortran-side fatal errors exit(1) instead of an
    unspecified STOP code -- see wannier90/api.py's module docstring) if one
    isn't already there. Never overwrites an existing make.inc someone may
    have hand-tuned (e.g. for a non-default BLAS/LAPACK)."""
    make_inc = root / "make.inc"
    if make_inc.exists():
        return
    template = (root / "config" / "make.inc.gfort.dynlib").read_text()
    template = template.replace("FCOPTS = -O3 -fPIC", "FCOPTS = -O3 -fPIC -DEXIT_FLAG")
    template = template.replace("LDOPTS = -fPIC", "LDOPTS = -fPIC -DEXIT_FLAG")
    make_inc.write_text(template)


def _build_libwannier(root: Path) -> Path:
    _ensure_pic_make_inc(root)
    subprocess.run(["make", "lib", "COMMS=serial"], cwd=root, check=True)
    lib = root / "libwannier.a"
    if not lib.exists():
        raise RuntimeError(f"make lib did not produce {lib}")
    return lib


def _build_extension(libwannier: Path, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    # f2py's -c mode only understands -l/-L flags for linking against
    # libwannier (a bare path to the .a, passed positionally, is silently
    # dropped -- it isn't a source file type f2py recognizes, and doesn't
    # end up in the meson.build it generates internally). And -L on the
    # f2py command line lands *after* -l in the flags it hands to the
    # linker, which GNU ld won't resolve (it doesn't search a -L directory
    # for a -l that came before it) -- so this uses LIBRARY_PATH, which gcc
    # folds into its own early -L set regardless of -l ordering.
    env = dict(os.environ, LIBRARY_PATH=str(libwannier.parent))
    subprocess.run(
        [
            sys.executable, "-m", "numpy.f2py", "-c",
            str(HERE / "wannier90.pyf"),
            "-lwannier", "-llapack", "-lblas",
        ],
        cwd=build_dir, check=True, env=env,
    )
    matches = list(build_dir.glob("_wannier90*.so"))
    if not matches:
        raise RuntimeError(f"f2py did not produce a _wannier90*.so in {build_dir}")
    return matches[0]


def main() -> None:
    import shutil
    import tempfile

    import wannier90  # fail fast with a clear error if the package itself isn't installed

    root = _find_wannier90_root()
    print(f"Using Wannier90 source at {root}")
    libwannier = _build_libwannier(root)
    with tempfile.TemporaryDirectory(prefix="f2py_wannier90_") as tmp:
        so_path = _build_extension(libwannier, Path(tmp))
        target_dir = Path(wannier90.__file__).resolve().parent
        shutil.copy(so_path, target_dir / so_path.name)
        print(f"Installed {so_path.name} into {target_dir}")


if __name__ == "__main__":
    main()
