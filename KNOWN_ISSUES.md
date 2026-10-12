# Known issues

Defects that a review found and that no change has fixed yet. Each entry names its kind and its
evidence.

### K2 · The residual magnitude of a DEIM column is reduced twice

- **Location:** `src/algorithms/deim.jl`, `_deim_indices`.
- **Evidence:** the residual `r` of a column is read by `maximum(abs, r)` for the `0 < ρ < Inf`
  test and read again by `last(findmax(abs, r))` for the pick. Binding both as
  `(ρ, k) = findmax(abs, r)` would save one `O(N)` pass, but every pick and every guard of the
  round-1 mutant lists names one of those two lines by its own text, and the plan's Q5 prescribes
  both by name (`ρ = max|r|`, `select with last(findmax(abs, r))`). The critics' round-1 lists
  and the `deim-all.toml` list of this fix round run against the two lines as they stand.
- **Kind:** dead code.
- **Found:** 2026-10-12.

### K3 · The element type of the DEIM residual buffer is not exercised

- **Location:** `src/algorithms/deim.jl`, `_deim_indices`.
- **Evidence:** `r = Vector{float(eltype(Ψ))}(undef, N)` is not separated by any test from a
  `Vector{Float64}` buffer: the broadcast into `r` computes in the element type of its operands
  and widens exactly. The mutant `r = Vector{Float64}(undef, N)` (D3 of the round-1 lists)
  SURVIVED both on `b980779` and on this branch. The probe `equivalent-mutants.jl` of this fix
  round shows `Float64.(rw) == rn`, the same `maximum(abs, ·)` and the same pick for both
  precisions at three scales.
- **Kind:** missing test.
- **Found:** 2026-10-12.

### K4 · No test separates the two symmetrisation stages of `L̃`

- **Location:** `src/algorithms/symplectic_deim.jl`, `SymplecticDEIM`.
- **Evidence:** no test separates the two stages, so dropping the first symmetrisation (`Ls = L`)
  while the product is symmetrised once more leaves the mutant (S1 of the round-1 lists, B2 of the
  fix round) SURVIVED. The tests are what is missing, not the effect: the identity
  `sym(AᵀLA) = Aᵀ·sym(L)·A` holds in exact arithmetic, and in floating point the mutant reproduces
  `L̃` only to the round-off of the product. On the pure skew `L = K − Kᵀ`, where `sym(L)` is
  exactly zero, the retained construction gives `L̃ = 0` exactly while the mutant gives a symmetric
  round-off, and `cancellation-equivalence.jl` of the verify-2 round measures
  `‖a − b‖∞ = 1.0835201e-6` in Float32 and `8.881784197001252e-16` in Float64 against that zero.
  The probe `equivalent-mutants.jl` of the fix round covers 36 other matrices, where the two
  constructions are both exactly symmetric and `‖a − b‖∞ ≤ 4·eps(T)·‖b‖∞`. Removing both stages is
  caught, by the plan's skew-`L` test.
- **Kind:** missing test.
- **Found:** 2026-10-12.

### K5 · The `Matrix{T}` conversion of `L̃` is not exercised

- **Location:** `src/algorithms/symplectic_deim.jl`, `SymplecticDEIM`.
- **Evidence:** `L̃ = Matrix{T}(A' * Ls * A)` is not separated by any test from
  `L̃ = A' * Ls * A`, which the field type `L̃::Matrix{T}` converts on construction: in every
  combination the field admits, the product is already a matrix of `T`, so `Matrix{T}(x)` copies
  a matrix of that element type and the field's `convert` would return `x` itself. The mutant E4
  of the round-1 lists SURVIVED. The conversion is therefore redundant with the field type, and
  the plan's Q1 formula does not name it.
- **Kind:** dead code.
- **Found:** 2026-10-12.
