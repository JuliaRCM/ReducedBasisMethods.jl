# Release Notes

All notable changes to ReducedBasisMethods.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually
changed, so that a compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries. 2 versions were
released before it, the most recent `v0.1.0`, and neither is written up here: the record of
that history is `git log` and the tags. It is named as a gap rather than reconstructed,
because a changelog assembled after the fact loses exactly the reasoning that makes it worth
keeping. The `[Unreleased]` target below is provisional — confirm it when the first entry is
written.

## [Unreleased] — targeting 0.2.0

### New Features

- **`reference/` is under version control.** It holds the reference implementation of *Symplectic
  model reduction methods for the Vlasov equation* (Tyranowski & Kraus,
  [arXiv:1910.06026](https://arxiv.org/abs/1910.06026),
  [doi:10.1002/ctpp.202200046](https://doi.org/10.1002/ctpp.202200046)). Nine files: one generator
  of the full-model reference data (`Vlasov_Analytic.jl`), three computing the bases — POD, PSD
  cotangent lift, PSD complex SVD — and five solving the reduced ODE and PODE equations that use
  them.

  It had **never been committed, on any branch**, and the directory was not gitignored but simply
  never added. It existed only as untracked files in one working tree, so nothing outside that
  machine had it.

  Committed **byte for byte as found**, deliberately: it is the baseline the rewrite is diffed
  against, so it is worth more unmodified than tidied. The consequence for anyone staging these
  files is that **the `pre-commit` formatter stage will reject them** — all nine fail
  `JuliaFormatter --check` against this repository's `sciml` style. They are already
  NFC-normalised, so that stage passes.

  What it is **not**: these files do not solve the Vlasov–Poisson system. They integrate
  `q̇ = p`, `ṗ = −β²q` against its closed-form solution, with `E(x) = β²x` in place of a Poisson
  solve. The paper's three bases were validated on that linear model, while `scripts/` carries the
  self-consistent model and implements only cotangent-lift EVD.

  They do not run as they stand. They target a GeometricIntegrators v1.x API — `set_config`,
  `getTableauERK4`, `create_hdf5`, `SSolutionODE` — and read and write **21 distinct hard-coded
  absolute paths**, 42 occurrences in all, under one author's former scratch directory. Each of
  the three projection scripts reads the same six β-run data files and writes one basis file; each
  reduced script reads two and writes two; the generator writes one.

### Bug Fixes

- **`ReducedTensor` now follows the `AbstractArray` convention for out-of-bounds dimensions.** Dimensions
  beyond the first three now return `1` for `size(rt, i)` and `Base.OneTo(1)` for `axes(rt, i)`,
  aligning with standard Julia array behavior. Previously these raised `BoundsError`. This fixes
  code patterns that work with generic `AbstractArray` interfaces.

### Changed

- **`[deps]` is now generic infrastructure only.** Removed from `[deps]`, with their `[compat]`
  entries where present: `ParticleMethods`, `PoissonSolvers`, `Distances`, `LaTeXStrings`,
  `LinearMaps`, `OffsetArrays`, `Optimisers`, `Parameters`, `Plots`, `Random`,
  `RecursiveArrayTools`, `TypedTables`, `Zygote`. `GeometricBrackets` and
  `MultiIndexArrays` join `[deps]`, at `0.1.1` each, for `ReducedTensor`: they provide the
  tensor operations and index utilities it uses. `IterativeSolvers` leaves the test
  dependencies. New `[compat]` bounds: `LinearAlgebra` and `Statistics` at `1`. The package now
  resolves and loads on Julia 1.11 and later.

- `scripts/bump_on_tail_2_projections.jl` is now Unicode NFC-normalised. It stored `Ã` as `A` plus
  a combining tilde on three lines, inherited from macOS rather than chosen. Nothing about what the
  script computes changes — Julia's parser normalises identifiers to NFC either way, and the base
  and normalised files parse to identical expression trees — but a `grep` pattern or an editor
  search typed in NFC now matches it, where before it silently matched nothing. No file under
  `src/` or `test/` was affected.

  The `X̃` and `x̃` in the same file still carry a combining tilde, and correctly so: a tilde over
  `X` or `x` has no precomposed codepoint, so NFC leaves them decomposed and the file is
  nonetheless fully normalised.

- **The test suite checks the package structure.** `test/integration/skeleton.jl` verifies that
  `[deps]` contains only packages from an explicit allowlist of generic infrastructure.
  `Aqua.test_all` with every check enabled runs from `test/quality/aqua.jl`. These checks catch
  unused dependencies and exported names that have no definition. A new `ReducedTensor` testset
  compares the tensor's stencil-based indexing against the dense double projection over all index
  pairs. The orphaned test files `poisson_test.jl` and `bracket_operators_test.jl` are removed.
  `trainingset_tests.jl` is removed with `TrainingSet`. `test/runtests.jl` no longer includes
  missing files.

- **The test dependencies are in `test/Project.toml`, and the test files mirror `src/`.** Aqua,
  TOML and Test move from `[extras]` and `[targets]` of `Project.toml`, which keeps no test
  dependency and no `[compat]` bound for them. Random and SafeTestsets are new test dependencies.
  GeometricBrackets, LinearAlgebra and ReducedComplexityModeling stay in `[deps]`, and
  `test/Project.toml` lists them with the same bounds. `deim_test.jl` becomes
  `algorithms/deim.jl`; `reducedbasis_tests.jl` splits into `reducedbasis.jl` and
  `reduced_tensor.jl`; `skeleton_tests.jl` becomes `integration/skeleton.jl`, and its Aqua check
  moves to `quality/aqua.jl`. Each file has its own `using`, and a file that draws random numbers
  fixes the seed. `runtests.jl` runs each file in its own `@safetestset`, in the `core` group,
  which the test arguments select. Nothing under `src/` changes.

### Breaking Changes

- **The grid-based and particle-based code left this package.** `src/gridbased/` and
  `src/particles/` are gone, and with them every name they exported. A caller who reached
  `PoissonTensor`, `PoissonOperator` or `Arakawa` now wants `GeometricBrackets`;
  `PotentialReducedTensor` and `VelocityReducedMatrix` are in `VlasovMethods`;
  `_apply_Δₓ!`, `_apply_Δₓ₄!` and `_apply_Rₓ!` are in `PoissonSolvers`; `_apply_∫dv!` is in
  `VlasovMethods`; and `multiindex`, `linearindex` and `_stencil_indices` are in
  `MultiIndexArrays`. `_apply_P_ϕ!` and `_apply_P_h!` went to `GeometricBrackets` with the
  bracket they apply. Code changes attended the split. `_apply_Δₓ₄!` in PoissonSolvers uses a
  different stencil than the removed code: (1,−16,30,−16,1)/12h² (fourth-order) replaces
  (5,−32,54,−32,5)/12h² (second-order), so callers see different numerical results. `PoissonTensor`
  and `PoissonOperator` in `GeometricBrackets` were rewritten: `Base.materialize` became
  `Base.Array`, and `@assert` bounds checks became `BoundsError`.

  `ReducedTensor` **stays**, now in `src/reduced_tensor.jl`. Its `PT <: PoissonTensor{DT}`
  bound is what makes this package depend on `GeometricBrackets`: relaxing the bound without an
  interface would break `getindex`, which reaches through to `_stencil_indices`.

  `ReducedElectricField`, `DEIMElectricField`, `Snapshots`, `IntegratorParameters`,
  `ReducedIntegratorCache` and `reduced_integrate_vp` were moved to `VlasovMethods/src/particles/`,
  which VlasovMethods does not include, so no package defines these names.

- **Vlasov and training code left this package.** `TrainingSet` is removed, together with
  `src/trainingset.jl`. The exports of `read_sampling_parameters`, `IntegratorCache` and
  `integrate_vp`, which had no definitions, are removed. `ReducedBasis` loses its fields
  `initconds`, `integrator` and `poisson`, and the three matching positional constructor arguments;
  the constructor `ReducedBasis(::CotangentLiftEVD, ::TrainingSet)` is removed. Its HDF5 round trip
  no longer writes those fields. The Vlasov HDF5 routines `save_tests`, `save_testing_parameters`
  and `h5save(fpath, ::IntegratorParameters, ::PoissonSolverPBSplines, ...)` are gone.

- **Minimum Julia is now 1.11.** The floor is raised from the declared 1.7 (never tested and
  would not resolve against current dependency versions) to 1.11. `GeometricBrackets`, added to
  this release's dependencies, declares `julia = "1.11"`. CI derives its lower matrix entry from
  this field, so every declared floor is tested.
