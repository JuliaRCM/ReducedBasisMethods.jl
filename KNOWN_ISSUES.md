# Known issues

Defects that a review found and that no change has fixed yet. Each entry names its kind and its
evidence.

## KI-1 · The DEIM result has no `@test`

- **Kind:** missing test (pre-existing).
- **Where:** `test/algorithms/deim.jl`, last line.
- **Claim:** the DEIM approximation error is checked with `@assert avg_deim_error < 5e-5`, not
  with `@test`, and no test checks the interpolation matrix `Π`. The mutant
  `j = argmax(r)` → `j = argmin(r)` in `src/algorithms/deim.jl` survives the test suite.
- **Evidence:** `mutate.jl <worktree> src/algorithms/deim.jl '        j = argmax(r)'
  '        j = argmin(r)' algorithms/deim.jl` gives SURVIVED.
- **Fix:** make the check a `@test`, and compare `Π` with the known DEIM indices.

## KI-2 · The skeleton test has no mutant that only its assertion catches

- **Kind:** not verified.
- **Where:** `test/integration/skeleton.jl`.
- **Claim:** one mutant, a `Printf` entry added to `[deps]`, was caught by the Pkg manifest
  error, not by `keys(project["deps"]) ⊆ permitted`. A second mutant, a `Random` entry, fails
  that assertion. The evidence is weak, not absent.

## KI-3 · The CHANGELOG entry for the test migration names the wrong dependencies

- **Kind:** docs.
- **Where:** `CHANGELOG.md`, `[Unreleased]`, the entry "Test infrastructure reorganized".
- **Claim:** the entry says that Aqua, GeometricBrackets, LinearAlgebra, Random,
  ReducedComplexityModeling, SafeTestsets, TOML and Test move from `[extras]` and `[targets]`.
  Only Aqua, TOML and Test were in `[extras]`. GeometricBrackets, LinearAlgebra and
  ReducedComplexityModeling stay in `[deps]`, and Random and SafeTestsets are new in
  `test/Project.toml`.
- **Evidence:** `git show 221852b:Project.toml` lists only Aqua, TOML and Test in `[extras]`, and
  GeometricBrackets, LinearAlgebra and ReducedComplexityModeling in `[deps]`, where
  `Project.toml` keeps them.

## KI-4 · An older CHANGELOG entry contradicts the test migration

- **Kind:** docs.
- **Where:** `CHANGELOG.md`, `[Unreleased]`, `### Changed`, the entry "`[deps]` is now generic
  infrastructure only".
- **Claim:** it says "The test target gains `Aqua` and `TOML`" and gives new `[compat]` bounds
  for `TOML`, `Test` and `Aqua`. `Project.toml` has no `[targets]` and none of those three bounds:
  the test dependencies are in `test/Project.toml`, which gives a test-only dependency no bound.
  The section contradicts itself.
- **Evidence:** `git show 221852b:Project.toml` has `[targets]` and the three bounds;
  `Project.toml` has neither.
