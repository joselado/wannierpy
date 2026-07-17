# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python bindings for [Wannier90](http://www.wannier.org)'s "library mode"
API -- the `wannier_setup`/`wannier_run` Fortran subroutines in
`src/wannier_lib.F90`, wrapped via a hand-written `f2py` signature file
(`wannier90.pyf`) into a compiled extension. This is a separate project
(its own git repo) from the `wannier90-3.1.0/` sibling directory it wraps;
that directory is the upstream Fortran source and is documented by its own
`CLAUDE.md`, not this one.

## Build / install / test

```bash
pip install ".[test]"        # builds libwannier.a + the f2py extension, see below
python -m pytest tests/ -v
```

Requires a Fortran compiler (gfortran) and LAPACK/BLAS dev headers. The
build (`setup.py`'s `BuildWannier90` command) looks for the Wannier90 3.1.0
source in order: the `WANNIER90_SRC` env var, `vendor/wannier90-3.1.0/`
inside this repo, or a `wannier90-3.1.0/` sibling directory (the layout used
in this dev checkout -- there's no vendored/pinned copy or git submodule
yet). It writes a `-fPIC -DEXIT_FLAG` `make.inc` there if none exists, runs
`make lib COMMS=serial`, then `f2py -c wannier90.pyf -lwannier -llapack
-lblas` with `LIBRARY_PATH` pointed at the built `libwannier.a` (see "f2py
gotcha" below), and copies the resulting `_wannier90*.so` into
`wannier90/`.

To iterate on the extension alone without a full `pip install`, that same
`f2py -c` command works standalone from this directory once `libwannier.a`
exists (`cd ../wannier90-3.1.0 && make lib COMMS=serial`).

**Verifying an install actually works** requires testing from outside this
source tree (e.g. a fresh venv + `cd /tmp && pytest .../tests/test_gaas.py`)
-- running pytest from here picks up the local `wannier90/` directory via
cwd on `sys.path` regardless of what's actually installed in site-packages,
which will hide a broken build.

`tests/test_gaas.py` is the only test file: it reproduces the GaAs
reference case from `../wannier90-3.1.0/test-suite/library-mode-test/`
against `ref/results_ref.dat`, in three ways (`in_process` vs subprocess
execution, and file-based vs fully-in-memory `.win` construction) -- it's
both the correctness check and the executable spec for the whole API.

## Architecture

Three layers, in `wannier90/`:

- **`_wannier90*.so`** -- the raw f2py extension. Its two functions,
  `wannier_setup`/`wannier_run`, mirror the Fortran subroutines' argument
  lists exactly as declared in `wannier90.pyf`, with redundant dimension
  arguments (`num_kpts_loc`, `num_bands_loc`, ...) hidden and inferred from
  array shapes. Never call this directly -- go through `api.py`.
- **`api.py`** -- the public `setup()`/`run()` functions, `SetupResult`/
  `RunResult` dataclasses, and the subprocess-isolation machinery
  (`_call`). This is where almost all the non-obvious behavior lives; read
  its module docstring before changing anything here.
- **`io_helpers.py`** -- optional convenience: parsers for the standard
  `.mmn`/`.amn`/`.eig` files a DFT interface produces, plus `write_win`
  (materializes a `.win` from Python data) and `reciprocal_lattice`. Nothing
  in `api.py` requires going through this module -- `M_matrix`/`A_matrix`/
  `eigenvalues` are just numpy arrays, however you build them.

### Why `setup()`/`run()` aren't a thin pass-through (read before touching `api.py` or `wannier90.pyf`)

Three non-obvious constraints, discovered empirically (none are documented
upstream), shape almost everything in `api.py`:

1. **Fatal Fortran errors kill the process, not raise an exception.**
   `io_error` (`src/io.F90`) calls `STOP`/`exit(1)` on any internal error.
   Linked in-process, that takes the whole Python interpreter down with it.
   `api.py` therefore runs each calculation in a `spawn`-ed subprocess by
   default (`in_process=True` opts out) and turns a dead worker into a
   normal `WannierError`.

2. **`wannier_run` silently requires `wannier_setup` to have run in the
   *same process* immediately before it** -- despite `wannier_run`'s own
   arguments looking self-sufficient, it depends on a module-global flag
   (`library_param_read_first_pass` in `src/parameters.F90`) that only
   `wannier_setup` initializes correctly. Skipping this reliably crashes
   with `SIGFPE` mid-disentanglement. This is why `run()` *requires* the
   `SetupResult` from its matching `setup()` call: in subprocess mode, it
   replays the exact original `wannier_setup` call (via
   `SetupResult._setup_args`) inside the same worker before calling
   `wannier_run`. If you add new arguments to `setup()`, make sure they
   still end up captured in `_setup_args`.

3. **`multiprocessing.Queue` deadlocks if you `join()` before draining
   it.** `wannier_run`'s output (U matrices etc.) can exceed the pipe
   buffer; `_call`'s polling `queue.get(timeout=...)` loop (checking
   `proc.is_alive()` between attempts) exists specifically to avoid both
   that deadlock and hanging forever when a worker dies without producing
   output. Don't "simplify" this back to `join()` then `get()`.

### f2py gotcha (relevant if you touch `wannier90.pyf` or the build)

`f2py -c`'s meson backend silently drops a bare `.a` path passed
positionally -- it's not a source type it recognizes, and it doesn't end up
in the meson.build it generates. Linking `libwannier.a` requires
`LIBRARY_PATH` (so gcc's own early `-L` set picks it up) plus `-lwannier`
on the command line, not a path argument -- `setup.py`'s `_build_extension`
depends on this; don't refactor it to pass the `.a` path directly.

`wannier90.pyf` is hand-written rather than auto-cracked from
`wannier_lib.F90` because some `intent(out)` arrays are dimensioned by
`num_nnmax`, a `parameter` in `w90_parameters` rather than a dummy
argument -- `crackfortran` can't resolve that across module boundaries. Its
value (12) is hard-coded in the `.pyf`; if the vendored Wannier90 version
ever changes, re-check `src/parameters.F90`'s `num_nnmax` definition.

### Library mode only reads a subset of `.win`

In library mode, wannier90 reads and then *ignores* `mp_grid`, `num_bands`,
and the `unit_cell_cart`/`atoms_frac`/`kpoints` blocks in `.win` -- those
come from `setup()`'s own arguments instead (confirmed from
`src/parameters.F90`'s "Ignoring `<mp_grid>` in input file" branches, and
covered by `tests/test_gaas.py::test_gaas_fully_in_memory`). Only
`num_wann` is unconditionally required. `write_win`/`setup()`'s
`win_keywords` intentionally don't support those redundant blocks.
