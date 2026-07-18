# Examples

Four self-contained scripts, one per dimensionality, each defining a
hard-coded tight-binding Bloch Hamiltonian and computing the `M_matrix`/
`A_matrix`/`eigenvalues` overlap data `wannier90.run()` needs directly from
it (no `.mmn`/`.amn`/`.eig` files, no DFT interface) -- see `_tb_utils.py`
for how, and why.

| Script | System |
|---|---|
| `0d_molecule.py` | 4-site open linear chain (no periodicity) |
| `1d_ssh_chain.py` | Su-Schrieffer-Heeger dimerized chain (2 orbitals/cell) |
| `2d_square_lattice.py` | Two-orbital checkerboard-like square lattice |
| `3d_cubic_lattice.py` | Two-orbital CsCl-like simple cubic lattice |

Run any of them directly:

```bash
python examples/1d_ssh_chain.py
```

(no install needed -- each script adds the repo root to `sys.path` itself
if `wannierpy` isn't already installed).

## What they show, and why the converged spread is exactly zero

Every example keeps `num_wann == num_bands` (every tight-binding orbital
becomes one Wannier function, no disentanglement). For a *complete,
untruncated* discrete tight-binding manifold like these, the maximally
localized Wannier functions are provably **exactly** the original
tight-binding sites, with **exactly zero** spread -- a real mathematical
fact (Omega_I is a gauge invariant of the raw overlap data alone, and a
full eigenbasis is always unitary), not a limitation of this package. The
1D/2D/3D examples still demonstrate real work: they seed the calculation
with deliberately imperfect (differently-sized Gaussian) trial orbitals,
print the spread *before* any CG minimisation, and let `wannier90.run()`
minimise it down to that exact answer -- see `_tb_utils.py`'s module
docstring for the full derivation, including why naive choices (trial
orbitals equal to the exact eigenbasis, or any other *fixed* trial matrix)
trivially -- and uninformatively -- give zero spread from the very first
iteration, with no minimisation happening at all.

`0d_molecule.py` is simpler (no trial-orbital tricks needed) but has its
own caveat: a single k-point has no Brillouin-zone variation for the
Wannier centre/spread formulas to extract *position* information from, so
while it correctly exercises the full pipeline, the reported centres don't
resolve individual site positions -- see its module docstring.

## Adapting these

To Wannierize your own hard-coded Hamiltonian: write `hamiltonian_k(k_frac)
-> Hermitian ndarray` in the "periodic gauge" (hoppings enter only via
`exp(i*2*pi*k.R)` for integer lattice vectors `R`, no sub-cell position
phases), then call `_tb_utils.build_overlaps` with it. If you want
genuinely non-trivial spreads (not the "complete manifold" zero-spread
case above), use real disentanglement: make your Hamiltonian bigger than
`num_wann` bands, with a genuinely entangled (not simply weakly/linearly
perturbed) coupling between the bands you keep and the ones you don't --
weak or perturbative coupling to a well-separated extra band, or a single
k-point, both still admit an exact zero-spread answer for the reasons
above.
