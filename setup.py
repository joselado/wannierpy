"""Custom build: compile libwannier.a from the (vendored or sibling) Wannier90
3.1.0 source tree, then build the f2py extension against it and drop the
result into the wannier90 package directory.

f2py's own ``-c`` mode already drives an internal meson build for the
extension itself; hand-writing a parallel top-level meson.build to redo the
same work (and re-derive the libwannier.a compile step, which is really just
"run the existing upstream Makefile") would duplicate that machinery for no
benefit, so this just automates the exact commands used (and verified) while
developing this package.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

HERE = Path(__file__).resolve().parent
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


class BuildWannier90(_build_py):
    def run(self):
        super().run()
        root = _find_wannier90_root()
        libwannier = _build_libwannier(root)
        temp_dir = Path(self.build_lib).resolve().parent / "temp_f2py_wannier90"
        so_path = _build_extension(libwannier, temp_dir)
        target_dir = Path(self.build_lib) / "wannier90"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(so_path, target_dir / so_path.name)


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildWannier90})
