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

## [Unreleased] — targeting 0.3.0

### New Features

- **`reduced_ode` and `reduced_pode` emit the reduced equation forms as `GeometricEquations`
  problems.** `reduced_ode(b, ∇H!, timespan, timestep, z₀; parameters)` is the ordinary problem of
  the reduced equation the basis carries: `ż = J_k Aᵀ ∇H(Az)` for the symplectic bases
  `CotangentLiftSVD`, `CotangentLiftEVD` and `ComplexSVD`, where `A = Matrix(b)`, and the Galerkin
  form `ż = Uᵀ J_N ∇H(Uz)` for the POD bases `PODSVD` and `PODEVD`, where `U = Φ`. A POD basis
  forms no canonical symplectic matrix in its reduced coordinates, so its reduced dimension `k`
  may be odd.
  `reduced_pode(b, ∇H!, timespan, timestep, z₀; parameters)` is the partitioned problem of the
  symplectic bases, whose two fields are the first and the last `k` rows of the same form; a POD
  basis has no `q` and `p` halves to split, and no method. A `SymplecticDEIM` emits `ż = J_2k ∇H̃(z)`
  through `hamiltonian_gradient!` by the same two functions. `∇H!(g, t, x, params)` writes
  `∇H(x)` into `g`, in the argument order of a vector field, and `parameters` reaches it unchanged.
  The state is in reduced coordinates — `project(b, x₀)` gives it — and its element type must be
  the basis's. Integration stays the caller's: these functions emit a problem and no integrator is
  a dependency. A vector field call allocates 0 B for basis states, and for the `ForwardDiff.Dual`
  states Newton's `JacobianAutodiff` passes after the first call for that element type, which fills
  a per-type pair of work buffers; whatever `∇H!` allocates is the caller's.

- **`GeometricEquations` joins `[deps]`, at `"0.21"`.** The two emitters above build
  `ODEProblem` and `PODEProblem`, and import `ODEProblem`, `PODEProblem` and `NullParameters` by
  name rather than through the package's whole export list. `GeometricIntegrators` and
  `GeometricIntegratorsBase` do not join `[deps]`: an integrator is the caller's, and the test
  dependencies supply one.

- **`spectrumplot` plots a reduced basis's spectrum, and `CairoMakie` joins `[weakdeps]` at
  `"0.15"`.** `spectrumplot(σ::AbstractVector{<:Real}; title, resolution)` draws a spectrum against
  its index on a log-scaled axis and returns a `Makie.FigureAxisPlot`, whose `plot` field holds the
  series, and `save(path, spectrumplot(σ))` writes it to a file. It is the singular-value or
  eigenvalue spectrum a `ReducedBasis` stores in `b.Λ`, so it draws the values as they are given,
  never squared and never rooted: an eigenvalue spectrum from a Gram matrix `MᵀM` and a
  singular-value spectrum of `M` differ by a square, and reading one as the other flatters the
  decay by a factor of two. The spectrum is drawn in `Float64`, the precision the plot stores, and
  is lifted onto the floor `eps(Float64) * maximum(σ)` first: an exact zero of a rank-deficient
  snapshot, and the slightly negative eigenvalue the Gram route returns, belong on the axis rather
  than below it. An empty spectrum, one holding a `NaN` or an `Inf`, and one holding no positive
  value at all — which leaves a log-scaled axis no scale to draw on — each throw an
  `ArgumentError`. The method lives in the package extension
  `ReducedBasisMethodsCairoMakieExt`: `src/diagnostics.jl` holds the stub, so
  `using ReducedBasisMethods` needs no plotting stack at all and `spectrumplot` raises the usual
  `MethodError` until `CairoMakie` is loaded.

  `Aqua.test_all` now runs inside a `@testset` of its own, as check 4 asks, so each check carries
  its own passes and failures in the suite's summary; `stale_deps`, `undefined_exports` and
  `deps_compat` — the last of which now also holds `CairoMakie`'s `[compat]` entry — are among them.

  The docs build sets `doctest = true`, `get_DEIM_interpolation_matrix` carries the package's
  first `jldoctest`, and `Weave` and `Plots` leave both `docs/make.jl` and `docs/Project.toml`: no
  page used either. The build also greps its own output for a physical word and fails on a match,
  so a name that reaches the site alone is red in the Documentation workflow rather than
  published.

## [0.2.0] — 2026-10-12

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

- **`ReducedBasis` is built from a snapshot by one constructor, with five algorithms.**
  `ReducedBasis(algorithm, S, parameters, paramspace; k = nothing, tolerance = nothing)`
  decomposes the snapshot matrix `S` itself. `S` must have element type `Float32` or `Float64`:
  any other element type, complex and integer included, is a `MethodError` at the constructor, on
  every algorithm. The algorithm tags are `PODSVD`, `PODEVD`, `CotangentLiftSVD`,
  `CotangentLiftEVD` and `ComplexSVD`. Exactly one of `k` (the basis size) and `tolerance` (in
  `[0, 1)`, which chooses the size from the spectrum) must be given. Before any decomposition, an
  `ArgumentError` is raised for both or neither, for a `k` outside `1:kmax`, where
  `kmax = min(N, m)` for the POD and complex algorithms and `min(N, 2m)` for the cotangent lift,
  for a `tolerance` outside `[0, 1)`, for a snapshot with a zero dimension, for an odd row count
  under a PSD algorithm, which reduces the stacked `[q; p]`, and for a snapshot with a `NaN` or an
  infinite entry. The struct holds `algorithm`, `parameters`, `paramspace`, `Λ`, `k`, `Φ` and `Ψ`.
  `Λ` is the full spectrum, sorted descending with negative entries kept. `Ψ` is the imaginary
  part for `ComplexSVD` and `nothing` otherwise.

- **`Matrix` assembles the basis, and `lift` and two error measures are exported.**
  `Matrix(b)` returns `Φ` for the POD routes, `diag(Φ, Φ)` for the cotangent lift and
  `[Φ −Ψ; Ψ Φ]` for the complex SVD. `lift` maps reduced coordinates back to the full space.
  `project` applied to a `ReducedBasis` is a new method of `ReducedComplexityModeling.project`;
  this package does not export a `project` of its own. `projection_error(b, X)` is the absolute
  Frobenius norm of `X - lift(b, project(b, X))`, and `reduction_error(b, X, Z)` is the same norm
  of `X - lift(b, Z)` for reduced coordinates `Z`.

- **`scripts/reference_spectra.jl` regenerates the paper's reference spectra.**
  It rebuilds the reference data from the closed form in `reference/Vlasov_Analytic.jl`, and
  prints the singular values and the energy-selected basis size of the three SVD routes.
- **`SymplecticDEIM` and `hamiltonian_gradient!`.** Structure-preserving DEIM for a Hamiltonian
  `H(x) = ½xᵀLx + Σᵢ hᵢ(x)`, after Peng and Mohseni,
  [doi:10.1137/140978922](https://doi.org/10.1137/140978922). DEIM is applied to the densities
  `hᵢ`, not to the gradient `∇H`, and the reduced gradient is then obtained by differentiating the
  reduced Hamiltonian exactly. It is therefore the gradient of a scalar, and its Jacobian is
  symmetric by construction, which is what makes the reduced Hamiltonian an exact invariant of the
  reduced system. The plain DEIM reduction of the same paper, eq. (5.8), gives neither: it uses
  `W = AᵀΨ(ΠᵀΨ)⁻¹`, which does not guarantee a symmetric reduced-gradient Jacobian, and the
  authors state that its reduced system is "not necessarily Hamiltonian". Both the call of the
  object and `hamiltonian_gradient!` read 0 B where their callbacks allocate nothing: the density
  callbacks receive a lazy view of `A_{Sⱼ}z`, so whatever a callback allocates is the caller's.
- **`ReducedTensor` wraps any rank-3 `GridTensor`, of any width its grid holds.** Its sum reads
  the tensor's own stencil, so it no longer needs the width-1 `Arakawa` the `PoissonTensor` bound
  stood for, and `getindex` allocates nothing after the first call rather than a `Vector` per call.

### Bug Fixes

- **`ReducedTensor` now follows the `AbstractArray` convention for out-of-bounds dimensions.** Dimensions
  beyond the first three now return `1` for `size(rt, i)` and `Base.OneTo(1)` for `axes(rt, i)`,
  aligning with standard Julia array behavior. Previously these raised `BoundsError`. This fixes
  code patterns that work with generic `AbstractArray` interfaces.

- **The DEIM selection follows the magnitude of the residual, and its result is no longer
  silently `Float64`.** `get_DEIM_interpolation_matrix` took the row of the largest *value* of the
  residual rather than of the largest magnitude. Where the residual has a positive entry, the row
  of the largest value can differ from the row of the largest magnitude: with normalised columns,
  `Ψ = [1 0; 0 -2; 0 1]` selected rows `[1, 3]` instead of `[1, 2]`. Where no entry is positive, the
  largest value is the smallest magnitude, and the selection could take a row already taken, so the
  interpolation matrix `Π` held the same row twice: `Ψ = [1 0; 0 -1; 0 -1]` normalised gave rows
  `[1, 1]`. The selection now takes the largest magnitude, and where two are equal the
  first of them. `Π` takes the element type of its input, `float(eltype(Ψ))`, where it was always
  `Float64`, so a `Float32` basis stays in `Float32`. This closes KI-1: the Example 1 error is a
  `@test` rather than an `@assert`, and `Π` is compared against the rows a hand-computed case
  selects.

### Changed

- CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of `Julia min`, and a test
  job saves the Julia cache only when it succeeds.

- **`get_DEIM_interpolation_matrix` rejects a basis it cannot interpolate.** It throws an
  `ArgumentError` where `k > N`, and where a column's largest residual magnitude is not both
  positive and finite — `0` for a zero column or one exactly dependent on the ones before it,
  the first column included, and `NaN` or `Inf` for a `NaN` or an `Inf` in the basis. Before,
  the argument was never validated: a `k > N` basis reached the solve. At `k = N + 1` it returned
  silently with a repeated row (a random `3 × 4` basis gave rows `[2, 3, 1, 1]`), and at `k ≥ N + 2`
  it threw a `SingularException` (a random `3 × 5` basis). A zero, dependent, `NaN` or `Inf` column was
  accepted with no diagnostic of its own. A zero first column was returned as a unit vector
  when `k = 1`, and reached that same singular solve when `k ≥ 2`.

- **`[deps]` is now generic infrastructure only.** Removed from `[deps]`, with their `[compat]`
  entries where present: `ParticleMethods`, `PoissonSolvers`, `Distances`, `LaTeXStrings`,
  `LinearMaps`, `OffsetArrays`, `Optimisers`, `Parameters`, `Plots`, `Random`,
  `RecursiveArrayTools`, `TypedTables`, `Zygote`. `MultiIndexArrays` joins `[deps]`, at `0.1.2`,
  for `ReducedTensor`: it provides the grid tensor and the index utilities it uses. `IterativeSolvers`
  leaves the test dependencies. New `[compat]` bounds: `LinearAlgebra` and `Statistics` at `1`.
  The package now resolves and loads on Julia 1.11 and later.

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
  GeometricBrackets becomes a test-only dependency, and `LinearAlgebra` and
  ReducedComplexityModeling stay in `[deps]`, and `test/Project.toml` lists them without a
  bound of its own. `deim_test.jl` becomes
  `algorithms/deim.jl`; `reducedbasis_tests.jl` splits into `reducedbasis.jl` and
  `reduced_tensor.jl`; `skeleton_tests.jl` becomes `integration/skeleton.jl`, and its Aqua check
  moves to `quality/aqua.jl`. Each file has its own `using`, and a file that draws random numbers
  fixes the seed. `runtests.jl` runs each file in its own `@safetestset`, in the `core` group,
  which the test arguments select. Nothing under `src/` changes.

- **`test/Project.toml` no longer repeats the root's bounds.** Its `[compat]` entries for
  LinearAlgebra and ReducedComplexityModeling are removed. The test environment
  contains the package, so the resolver applies the root `Project.toml` bounds to every shared
  dependency; a test bound can only narrow them, and tests that run on narrower bounds than the
  package claims hide a failing version. The rule is: `test/Project.toml` and `docs/Project.toml`
  carry no `[compat]` entry for a dependency of the root. The test-only bounds for Aqua,
  ForwardDiff, GeometricBrackets and GeometricIntegrators stay.
  `docs/Project.toml` had no shared entry and is unchanged.

- **The SVD routes skip the right singular vectors, and the EVD routes are rewritten.**
  `PODSVD`, `CotangentLiftSVD` and `ComplexSVD` call LAPACK's `gesvd!` with `jobu = 'S'` and
  `jobvt = 'N'`, so the right singular vectors, which the basis does not use, are never formed.
  `PODEVD` and `CotangentLiftEVD` take the eigendecomposition of the Gram matrix and build the
  basis from a Householder QR, so the leading columns keep the nested spans: the first `k` columns
  of a larger basis span the same space as a basis of size `k`, for each leading block that keeps
  full rank. At rank loss the QR completes an orthonormal basis, so the projector can include
  directions outside the snapshot's column space.

- **`LazyArrays` is no longer a dependency.**
  It leaves `[deps]` and `[compat]`. It was used only by the `ApplyArray` construction in the
  cotangent-lift EVD route, which the rewrite removes, so a caller that needs `LazyArrays` now
  adds it to its own environment.

- **`ReducedComplexityModeling` is required at `0.1.1`.** The `0.1` bound admitted 0.1.0, which
  needs GeometricEquations 0.10 and GeometricSolutions 0.3. At the lower bounds the resolver
  selects 0.1.1 instead, so the CI Downgrade job failed its lower-bound check.

### Breaking Changes

- **`ReducedTensor` is built on `GridTensor`, not on `PoissonTensor`.** A caller passes any rank-3
  `MultiIndexArrays.GridTensor`; a `PoissonTensor`, a dense array, or a `GridTensor` of another
  rank is a `MethodError`. `GeometricBrackets` leaves `[deps]` and `[compat]`, and
  `MultiIndexArrays` takes the bound `0.1.2`. The constructor throws a `DimensionMismatch` or an
  `ArgumentError` where it asserted, and an index off the array throws a `BoundsError` through
  `checkbounds`, not an `AssertionError`. A tensor of stencil width `w` on a grid with a direction
  shorter than `2w + 1` nodes is an `ArgumentError`: its stencil names a node twice.

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

  `ReducedTensor` **stays**, now in `src/reduced_tensor.jl`, and is re-parameterised on
  `MultiIndexArrays`' `GridTensor`: its bound is `GT <: GridTensor{DT, 3}`, and its sum follows the
  tensor's own locality contract rather than the width-1 `Arakawa` in a `PoissonTensor`.

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
  would not resolve against current dependency versions) to 1.11. `GeometricBrackets`, the test
  environment's bracket, declares `julia = "1.11"`. CI derives its lower matrix entry from
  this field, so every declared floor is tested.

- **The 0.1.0 reduced-basis algorithms and their fields are removed.**
  `UnspecifiedAlgorithm` goes, with the particle-shaped and field-shaped fields `Λₚ`, `kₚ`, `Ψₚ`,
  `Λₑ`, `kₑ`, `Ψₑ` and `Πₑ`. A basis is no longer split into a particle part and a field part:
  `Λ` replaces `Λₚ` and `Λₑ` as the full spectrum. Code that built a basis calls the new
  constructor on the snapshot and selects the algorithm by its tag.

- **The 0.1.0 basis functions are deleted, with their exports.**
  `get_PODBasis_cotangentLiftEVD`, `get_PODBasis_EVD`, `get_Ψ` and `sorteigen` are removed, and
  `src/eigen.jl` with them. The two EVD functions are now the `CotangentLiftEVD` and `PODEVD`
  tags. `Ψ` is a field of the basis, set for `ComplexSVD`, so `get_Ψ` has no counterpart. `POD`
  is not an algorithm tag: `ReducedComplexityModeling` exports `POD`, and this package exports no
  second `POD` or `project`, so a session that loads both sees one name each. The POD routes are
  `PODSVD` and `PODEVD`.

- **The HDF5 layout changed, so a file written by 0.1.0 no longer loads.**
  The keys are now `algorithm` (the tag name), `k`, `Λ`, `Φ`, `Ψ` (for `ComplexSVD` only),
  `parameters` and `parameterspace`. The 0.1.0 keys `kp`, `ke`, `Λp`, `Ψp`, `Λe`, `Ψe` and `Πe`
  are gone. The new layout holds no snapshot, so a basis saved by 0.1.0 is recomputed from its
  snapshot with the new constructor.
