# wannier90py

Python bindings for [Wannier90](http://www.wannier.org)'s "library mode" API
-- the `wannier_setup`/`wannier_run` Fortran subroutines in
`src/wannier_lib.F90` that let a host program compute maximally-localised
Wannier functions in-process, without shelling out to `wannier90.x`. This
package calls that library directly through a compiled extension (built with
`f2py`), passing numpy arrays in and getting numpy arrays out.

It does **not** compute overlaps/projections/eigenvalues itself -- those
(`.mmn`/`.amn`/`.eig` files) still come from a DFT code's
`pw2wannier90`-style interface. `wannier90.io_helpers` parses those standard
Wannier90 file formats into the arrays `wannier90.run()` expects -- or build
those arrays yourself and skip files entirely, see "Everything through
memory" below.

## Installing

Requires a Fortran compiler (gfortran) and LAPACK/BLAS development headers
(e.g. `apt install gfortran libblas-dev liblapack-dev` on Debian/Ubuntu).

```bash
pip install .
```

The build automatically compiles `libwannier.a` from the Wannier90 3.1.0
source and links it into the extension -- see "How the build works" below
for where it expects to find that source.

## Usage

```python
import numpy as np
import wannier90
from wannier90 import io_helpers

# 1. wannier_setup: needs <cwd>/<seedname>.win on disk for algorithmic
#    parameters (projections, exclude_bands, disentanglement window, ...)
#    -- those aren't passed as arguments here, only the structural data is.
setup_result = wannier90.setup(
    "gaas", mp_grid, kpt_latt, real_lattice, num_bands_tot,
    atom_symbols, atoms_cart, gamma_only=False, spinors=False, cwd="run_dir",
)

# 2. Parse the DFT interface's overlap/projection/eigenvalue files. nnlist/
#    nncell (from setup_result) are needed to interpret the .mmn file.
M_matrix = io_helpers.read_mmn("run_dir/gaas.mmn", setup_result.num_bands,
                                num_kpts, setup_result.nntot,
                                setup_result.nnlist, setup_result.nncell)
A_matrix = io_helpers.read_amn("run_dir/gaas.amn", setup_result.num_bands,
                                num_kpts, setup_result.num_wann)
eigenvalues = io_helpers.read_eig("run_dir/gaas.eig", setup_result.num_bands, num_kpts)

# 3. wannier_run: must be passed the SetupResult from step 1 (see "Process
#    isolation" below for why).
run_result = wannier90.run(
    "gaas", setup_result, mp_grid, kpt_latt, real_lattice,
    atom_symbols, atoms_cart, M_matrix, A_matrix, eigenvalues,
    gamma_only=False, cwd="run_dir",
)
print(run_result.wann_centres, run_result.wann_spreads)
```

See `tests/test_gaas.py` for a complete runnable example (the GaAs case
shipped with upstream Wannier90).

## Everything through memory

Nothing in this API requires touching disk. `M_matrix`/`A_matrix`/
`eigenvalues` were always just numpy arrays -- `io_helpers.read_mmn`/
`read_amn`/`read_eig` are one way to build them (from files a DFT interface
wrote), not the only way; build them yourself (e.g. from overlaps you
computed directly in Python) and pass them to `run()` exactly the same way.

The one thing that historically *had* to be a file was `.win`: wannier90's
`param_read` only reads from disk, no library-mode argument carries its
content. `setup()` now writes it for you from Python data, so you never
have to hand-author one:

```python
setup_result = wannier90.setup(
    "gaas", mp_grid, kpt_latt, real_lattice, num_bands_tot,
    atom_symbols, atoms_cart, gamma_only=False, spinors=False,
    win_keywords={
        "num_wann": 8, "num_iter": 1000, "conv_tol": 1e-10,
        "dis_win_max": 24.0, "dis_froz_max": 14.0, "dis_num_iter": 1200,
    },
    exclude_bands=range(1, 6),  # or a pre-formatted string, e.g. "1-5"
    projections=["f=0.25,0.25,0.25 : s", "f=0.25,0.25,0.25 : p"],
    # cwd left unset: a scratch directory is created for you, and returned
    # as setup_result.cwd -- run() below defaults to it automatically.
)
```

`num_wann` is the one keyword wannier90 always requires; `mp_grid`,
`num_bands`, and the cell/atoms/k-points blocks are deliberately not
supported here because library mode reads and *ignores* them regardless
(see `write_win`'s docstring) -- those come from `setup()`'s own arguments,
which is exactly what "library mode" means. If you pass `win_keywords`/
`exclude_bands`/`projections`, they overwrite any `.win` already at that
path; leave all three unset to fall back to an existing hand-authored file,
as in the first example. `tests/test_gaas.py::test_gaas_fully_in_memory`
reproduces the same GaAs case as the file-based test this way, with no
`.win` and no explicit `cwd` at all.

## Process isolation (read this before setting `in_process=True`)

By default, every `wannier90.setup()`/`wannier90.run()` call runs in a
fresh subprocess. This isn't just a safety nicety -- two real constraints in
the underlying Fortran library make it necessary:

1. **Fatal errors call `STOP`, not an exception.** Wannier90's error handler
   (`io_error` in `src/io.F90`) calls Fortran `STOP` (or `exit(1)`, since
   the build sets `-DEXIT_FLAG`) on any internal error -- bad input,
   mismatched shapes, a singular matrix, a missing `.win` file. Linked
   in-process, that kills the whole Python interpreter with no exception
   raised. Isolating each call in a subprocess turns that into a normal
   `wannier90.WannierError`, with the tail of `<seedname>.wout` attached for
   context.

2. **`wannier_run` silently depends on `wannier_setup` having run in the
   *same process* just before it.** This was found empirically while
   building this package (it isn't documented upstream): despite
   `wannier_run` taking what looks like a self-sufficient set of arguments,
   it relies on a module-global flag (`library_param_read_first_pass` in
   `src/parameters.F90`) that only `wannier_setup` initializes correctly.
   Calling `wannier_run` alone in a fresh process reliably crashes with
   `SIGFPE` partway through disentanglement. Because of this,
   `wannier90.run()` requires the `SetupResult` from its matching `setup()`
   call and, in subprocess mode, replays that exact `wannier_setup` call
   (silently, output discarded) in the worker before calling `wannier_run`.

Pass `in_process=True` to skip the subprocess and call the extension
directly -- only do this if you have your own process isolation (e.g. one
calculation per worker in a larger batch job already), and always call
`setup()` then `run()` in that same process in order, matching the only
usage pattern Wannier90 upstream actually tests
(`test-suite/library-mode-test/test_library.F90`).

Because of constraint 2, and because `wannier_setup`/`wannier_run` write
into module-global Fortran state, don't call `setup()`/`run()` more than
once (i.e. more than one calculation) in the same process, in-process mode
or not.

## Serial only

Library mode is documented upstream as serial-only: even a `libwannier`
built with `COMMS=mpi` must be called from a single MPI rank. This package
always builds the serial variant and has no `mpi4py` integration.

## How the build works

`setup.py` (1) compiles `libwannier.a` from the Wannier90 3.1.0 source via
its own upstream `Makefile` (`make lib COMMS=serial`, with a `make.inc`
copied from `config/make.inc.gfort.dynlib` and `-DEXIT_FLAG` added if one
doesn't already exist), then (2) builds the extension via `f2py -c` against
`wannier90.pyf` -- a hand-written f2py signature file (`wannier_setup`/
`wannier_run` weren't auto-wrapped from source because they size some
`intent(out)` arrays using `num_nnmax`, a `parameter` in the `w90_parameters`
module rather than a dummy argument, which `crackfortran` can't resolve
across module boundaries -- see the comment at the top of `wannier90.pyf`).

The build looks for the Wannier90 3.1.0 source, in order: the
`WANNIER90_SRC` environment variable, `vendor/wannier90-3.1.0/` inside this
package, or a `wannier90-3.1.0/` sibling directory next to it (the layout
used while developing this package, against a local Wannier90 checkout --
not yet a pinned vendored copy or git submodule, since this isn't a git
repo). The `.pyf` encodes version-specific facts (argument order,
`num_nnmax`'s value), so pin the Wannier90 source version you build against.

## License

Wannier90 is GPLv2, with no linking exception. This package links
`libwannier.a` into a compiled extension, making the combined work GPLv2 as
well.
