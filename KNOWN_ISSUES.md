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
