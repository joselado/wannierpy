"""High-level Python API around the Wannier90 "library mode" subroutines
(``wannier_setup``/``wannier_run`` in wannier90-3.1.0/src/wannier_lib.F90).

By default :func:`setup`/:func:`run` run in a freshly ``spawn``-ed worker
subprocess (see ``_worker.py``). This is not an incidental safety net --
it's load-bearing:

* wannier90's Fortran error handler (``io_error``) calls ``STOP``/``exit(1)``
  on fatal errors (bad input, singular matrices, ...). Linked in-process,
  that kills the whole Python interpreter with no exception raised.
* ``wannier_setup``/``wannier_run`` write into module-global Fortran state
  and are only ever exercised once per process by the upstream reference
  caller (test-suite/library-mode-test/test_library.F90) -- calling the pair
  twice in one process is not a supported usage pattern.
* Library mode is documented as serial-only; a subprocess-per-call also
  makes that trivially true by construction.

There's a second, undocumented constraint discovered empirically while
building this wrapper: ``wannier_run`` silently depends on ``wannier_setup``
having already run *in the same process* -- it does not just size arrays,
it also primes a module-global flag (``library_param_read_first_pass`` in
src/parameters.F90) that controls whether ``num_bands`` gets decremented by
the excluded-band count during ``wannier_run``'s own internal ``param_read``
call. Calling ``wannier_run`` alone in a fresh process reliably crashes with
SIGFPE partway through disentanglement. So the subprocess boundary here is
per *(setup, run) pair*, not per call: :func:`run`'s subprocess replays the
exact original :func:`setup` call first (outputs discarded) before calling
``wannier_run``, using the call captured on the ``SetupResult`` you pass in.

Pass ``in_process=True`` to skip the subprocess and call the extension
directly, if you have your own isolation (e.g. you already run one
calculation per worker in a larger job) and want to avoid the overhead --
in that mode just call :func:`setup` then :func:`run` normally in the same
process, matching the upstream pattern directly (no replay needed).
"""
from __future__ import annotations

import multiprocessing
import os
import queue as _queue
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .io_helpers import reciprocal_lattice, write_win  # reciprocal_lattice re-exported for convenience

__all__ = [
    "setup",
    "run",
    "reciprocal_lattice",
    "SetupResult",
    "RunResult",
    "WannierError",
]

_ATOM_SYMBOL_LEN = 20  # must match character*20 in wannier90.pyf


class WannierError(RuntimeError):
    """Raised when the underlying wannier90 call fails or aborts."""


@dataclass
class SetupResult:
    nntot: int
    nnlist: np.ndarray
    nncell: np.ndarray
    num_bands: int
    num_wann: int
    proj_site: np.ndarray
    proj_l: np.ndarray
    proj_m: np.ndarray
    proj_radial: np.ndarray
    proj_z: np.ndarray
    proj_x: np.ndarray
    proj_zona: np.ndarray
    exclude_bands: np.ndarray
    proj_s: np.ndarray
    proj_s_qaxis: np.ndarray
    cwd: str = field(default=None)
    """The directory the calculation ran in (auto-created under the system
    temp dir if you didn't pass ``cwd=`` to :func:`setup`). :func:`run`
    defaults to this if you don't override it -- pass the same ``seedname``
    and this is where ``<seedname>.win``/``.wout`` live."""
    _setup_args: tuple = field(default=None, repr=False)
    """The exact positional arguments passed to ``wannier_setup``. Used to
    replay the call inside :func:`run`'s worker subprocess; not part of the
    public result -- don't rely on its contents."""


@dataclass
class RunResult:
    U_matrix: np.ndarray
    U_matrix_opt: np.ndarray
    lwindow: np.ndarray
    wann_centres: np.ndarray
    wann_spreads: np.ndarray
    spread_total: float
    spread_invariant: float
    spread_tilde: float


def _pad_symbols(atom_symbols) -> np.ndarray:
    arr = np.array([s.encode() if isinstance(s, str) else bytes(s) for s in atom_symbols],
                    dtype=f"S{_ATOM_SYMBOL_LEN}")
    return arr


def _read_wout_tail(cwd: str, seedname: str, n_lines: int = 20) -> str:
    path = os.path.join(cwd, f"{seedname}.wout")
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return f"(could not read {path})"


def _call(func_name: str, args: tuple, cwd: str, seedname: str, in_process: bool, setup_args=None):
    """Invoke ``_wannier90.<func_name>(*args)``. If ``setup_args`` is given
    and we're not running in-process, ``wannier_setup(*setup_args)`` is
    replayed in the same worker process immediately before ``func_name`` --
    see the module docstring for why that's required for ``wannier_run``."""
    if in_process:
        from . import _wannier90

        prev_cwd = os.getcwd()
        os.chdir(cwd)
        try:
            return getattr(_wannier90, func_name)(*args)
        finally:
            os.chdir(prev_cwd)

    from ._worker import run_in_worker

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=run_in_worker, args=(queue, func_name, args, cwd, setup_args))
    proc.start()

    # Drain the queue *before* join(): wannier_run's output (U matrices etc.)
    # can exceed the OS pipe buffer, and Queue.put() blocks in a feeder
    # thread until the reader catches up. join()-then-get() deadlocks on
    # that; poll get() instead, falling back to "the process died without
    # producing output" (a Fortran STOP/SIGFPE never reaches our queue.put).
    result = None
    got_result = False
    while proc.is_alive() or not queue.empty():
        try:
            result = queue.get(timeout=0.2)
            got_result = True
            break
        except _queue.Empty:
            continue
    proc.join()

    if not got_result:
        if proc.exitcode != 0:
            raise WannierError(
                f"wannier90 '{func_name}' aborted (subprocess exit code {proc.exitcode}). "
                f"Last lines of {seedname}.wout:\n{_read_wout_tail(cwd, seedname)}"
            )
        raise WannierError(f"wannier90 '{func_name}' worker exited with no result")

    status, payload = result
    if status == "error":
        raise WannierError(f"wannier90 '{func_name}' raised: {payload}")
    return payload


def setup(
    seedname: str,
    mp_grid,
    kpt_latt,
    real_lattice,
    num_bands_tot: int,
    atom_symbols,
    atoms_cart,
    *,
    win_keywords: dict | None = None,
    exclude_bands=None,
    projections=None,
    recip_lattice=None,
    gamma_only: bool = False,
    spinors: bool = False,
    cwd: str | None = None,
    in_process: bool = False,
) -> SetupResult:
    """Call ``wannier_setup``.

    wannier90 still needs a ``<cwd>/<seedname>.win`` on disk for the
    algorithmic parameters that have no argument here -- ``num_wann``,
    disentanglement/convergence settings, ``exclude_bands``, ``projections``
    (cell/atoms/k-points/``mp_grid``/``num_bands`` in ``.win`` are read but
    *ignored* in library mode; those come from this function's own
    arguments instead, see :func:`wannier90.io_helpers.write_win`). You have
    two ways to provide it:

    * Pass ``win_keywords``/``exclude_bands``/``projections`` here and this
      writes ``.win`` for you (via :func:`wannier90.io_helpers.write_win`)
      -- nothing needs to touch disk yourself, so long as ``num_wann`` is in
      ``win_keywords``. This overwrites any existing ``.win`` at that path.
    * Or write ``<cwd>/<seedname>.win`` yourself beforehand and leave all
      three of those as ``None``.

    If ``cwd`` is ``None`` (the default), a fresh temporary directory is
    created for you (not cleaned up automatically -- it holds ``.wout``,
    useful for debugging; remove it yourself if you don't need it). The
    resolved directory is returned as ``SetupResult.cwd``, and :func:`run`
    uses that by default, so the common all-in-memory case needs no
    explicit ``cwd`` at either call site.

    ``real_lattice``/``atoms_cart`` are in Angstrom; ``kpt_latt`` is in
    fractional (reciprocal-lattice) coordinates, shape ``(3, num_kpts)``.
    """
    if cwd is None:
        cwd = tempfile.mkdtemp(prefix="wannier90_")
    else:
        os.makedirs(cwd, exist_ok=True)

    win_path = os.path.join(cwd, f"{seedname}.win")
    if win_keywords is not None or exclude_bands is not None or projections is not None:
        write_win(win_path, keywords=win_keywords, exclude_bands=exclude_bands, projections=projections)
    elif not os.path.exists(win_path):
        raise WannierError(
            f"No {win_path} and no win_keywords/exclude_bands/projections given -- "
            "either pass those (at minimum win_keywords={'num_wann': ...}) or write "
            f"{seedname}.win into cwd yourself before calling setup()."
        )

    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    if recip_lattice is None:
        recip_lattice = reciprocal_lattice(real_lattice)
    recip_lattice = np.asarray(recip_lattice, dtype=np.float64)
    kpt_latt = np.asarray(kpt_latt, dtype=np.float64)
    atoms_cart = np.asarray(atoms_cart, dtype=np.float64)
    mp_grid = np.asarray(mp_grid, dtype=np.int32)
    symbols = _pad_symbols(atom_symbols)

    args = (
        seedname, mp_grid, real_lattice, recip_lattice, kpt_latt,
        int(num_bands_tot), symbols, atoms_cart, bool(gamma_only), bool(spinors),
    )
    out = _call("wannier_setup", args, cwd, seedname, in_process)
    (nntot, nnlist, nncell, num_bands, num_wann, proj_site, proj_l, proj_m,
     proj_radial, proj_z, proj_x, proj_zona, exclude_bands_out, proj_s, proj_s_qaxis) = out
    return SetupResult(
        nntot=int(nntot), nnlist=nnlist, nncell=nncell,
        num_bands=int(num_bands), num_wann=int(num_wann),
        proj_site=proj_site, proj_l=proj_l, proj_m=proj_m, proj_radial=proj_radial,
        proj_z=proj_z, proj_x=proj_x, proj_zona=proj_zona, exclude_bands=exclude_bands_out,
        proj_s=proj_s, proj_s_qaxis=proj_s_qaxis, cwd=cwd, _setup_args=args,
    )


def run(
    seedname: str,
    setup_result: SetupResult,
    mp_grid,
    kpt_latt,
    real_lattice,
    atom_symbols,
    atoms_cart,
    M_matrix,
    A_matrix,
    eigenvalues,
    *,
    recip_lattice=None,
    gamma_only: bool = False,
    cwd: str | None = None,
    in_process: bool = False,
) -> RunResult:
    """Call ``wannier_run``.

    ``setup_result`` must be the :class:`SetupResult` from the matching
    :func:`setup` call (its ``nnlist``/``nncell`` are what you used to build
    ``M_matrix`` via :func:`wannier90.io_helpers.read_mmn`, and -- in
    subprocess mode -- it's also what lets this call correctly replay
    ``wannier_setup`` in the worker before calling ``wannier_run``; see the
    module docstring). ``cwd`` defaults to ``setup_result.cwd``, so the
    ``.win`` that :func:`setup` wrote (or found) is still there.
    """
    if cwd is None:
        cwd = setup_result.cwd
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    if recip_lattice is None:
        recip_lattice = reciprocal_lattice(real_lattice)
    recip_lattice = np.asarray(recip_lattice, dtype=np.float64)
    kpt_latt = np.asarray(kpt_latt, dtype=np.float64)
    atoms_cart = np.asarray(atoms_cart, dtype=np.float64)
    mp_grid = np.asarray(mp_grid, dtype=np.int32)
    symbols = _pad_symbols(atom_symbols)
    M_matrix = np.asarray(M_matrix, dtype=np.complex128)
    A_matrix = np.asarray(A_matrix, dtype=np.complex128)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

    args = (
        seedname, mp_grid, real_lattice, recip_lattice, kpt_latt,
        symbols, atoms_cart, bool(gamma_only), M_matrix, A_matrix, eigenvalues,
    )
    out = _call("wannier_run", args, cwd, seedname, in_process, setup_args=setup_result._setup_args)
    U_matrix, U_matrix_opt, lwindow, wann_centres, wann_spreads, spread = out
    return RunResult(
        U_matrix=U_matrix, U_matrix_opt=U_matrix_opt, lwindow=lwindow.astype(bool),
        wann_centres=wann_centres, wann_spreads=wann_spreads,
        spread_total=float(spread[0]), spread_invariant=float(spread[1]),
        spread_tilde=float(spread[2]),
    )
