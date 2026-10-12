# Known issues

Defects that a review found and that no change has fixed yet. Each entry names its kind and its
evidence.

## KI-1 · The DEIM result has no `@test`

- **Kind:** missing test (pre-existing).
- **Where:** `test/algorithms/deim.jl`, last line.
- **Claim:** the DEIM approximation error is checked with `@assert avg_deim_error < 5e-5`, not
  with `@test`, and no test checks the interpolation matrix `Π`. The mutant
  `j = argmax(r)` → `j = argmin(r)` in `src/algorithms/deim.jl` survives the test suite.
- **Evidence:** replace `j = argmax(r)` with `j = argmin(r)` in `src/algorithms/deim.jl`, then run

  ```
  julia --project=test -e 'using Pkg; Pkg.instantiate()'
  julia --project=test test/algorithms/deim.jl
  ```

  from the repository root. The file runs to the end with no failure and no error.
- **Fix:** make the check a `@test`, and compare `Π` with the known DEIM indices.

### K1 · A non-finite entry in the snapshot gives a full-size basis with a non-finite spectrum

- **Location:** `src/reducedbasis.jl:158`, the fallback of the energy rule in `_basis_size`.
- **Evidence:** with `S[3, 4] = Inf` on the `40 × 60` rank-10 Float64 snapshot of
  `test/helpers/spectra.jl`, `ReducedBasis(PODSVD(), S, NamedTuple(), ps; tolerance = 1e-3)`
  returns `k = 40 = kmax` with `isequal(Λ[1:3], [NaN, NaN, NaN])` instead of raising, and
  `ReducedBasisMethods._basis_size([4.0, NaN, -0.5], nothing, 0.5)` returns `3 = length(Λ)`: a
  non-finite energy makes `findfirst` find nothing, and the fallback then keeps the whole
  admissible basis. The `ReducedBasis` constructor has no guard for a non-finite snapshot.
- **Kind:** defect.
- **Found:** 2026-10-12.
