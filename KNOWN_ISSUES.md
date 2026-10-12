# Known issues

Defects that a review found and that no change has fixed yet. Each entry names its kind and its
evidence.

### K2 · The residual magnitude of a DEIM column is reduced twice

- **Location:** `src/algorithms/deim.jl`, `_deim_indices`.
- **Evidence:** `_deim_indices` reads each column's residual with `maximum(abs, r)` for
  validation and with `last(findmax(abs, r))` for selection. Taking both the magnitude and
  index from one `findmax(abs, r)` saves a residual scan without changing the selection rule.
- **Kind:** defect.
- **Found:** 2026-10-12.

### K3 · The element type of the DEIM residual buffer is not exercised

- **Location:** `src/algorithms/deim.jl`, `_deim_indices`.
- **Evidence:** replacing `r = Vector{float(eltype(Ψ))}(undef, N)` with a `Vector{Float64}`
  buffer leaves the Float32 and Float64 selection tests passing. The broadcast computes in
  its operands' element type before assignment, so widening the stored Float32 residual
  values to Float64 preserves their magnitudes and selected indices.
- **Kind:** missing test.
- **Found:** 2026-10-12.

### K4 · No test separates the two symmetrisation stages of `L̃`

- **Location:** `src/algorithms/symplectic_deim.jl`, `SymplecticDEIM`.
- **Evidence:** the tests pass with `Ls = L` in place of `Ls = (L + transpose(L)) / 2`
  while retaining the final symmetrisation of `L̃`. The identity
  `sym(AᵀLA) = Aᵀ·sym(L)·A` holds in exact arithmetic, but the two constructions can differ
  by product round-off. For a pure-skew `L = K − Kᵀ`, the initial symmetrisation gives
  exactly zero, whereas symmetrising only the product can retain nonzero symmetric
  round-off. The tests reject removal of both symmetrisation stages, but do not cover this
  pure-skew cancellation case.
- **Kind:** missing test.
- **Found:** 2026-10-12.

### K5 · The `Matrix{T}` conversion of `L̃` is not exercised

- **Location:** `src/algorithms/symplectic_deim.jl`, `SymplecticDEIM`.
- **Evidence:** `L̃ = Matrix{T}(A' * Ls * A)` copies the product when it already has type
  `Matrix{T}`. Assigning the product directly to the `L̃::Matrix{T}` field converts it when
  needed and otherwise retains it without a copy. The tests pass without the explicit
  `Matrix{T}` wrapper.
- **Kind:** dead code.
- **Found:** 2026-10-12.
