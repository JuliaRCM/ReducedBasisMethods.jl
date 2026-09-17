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

### Changed

- **Every dependency now carries a `[compat]` bound.** `Distances`, `Optimisers`, `Statistics` and
  `Zygote` were in `[deps]` with no entry, so the resolver was free to install any version of them,
  including one whose interface this package does not use. They are now bounded at `0.10`, `0.4`,
  `1` and `0.7`. `Parameters` gains `0.13` alongside `0.12`. The five bounds come from the five open
  CompatHelper pull requests (#19, #26, #32, #33, #34), combined here into one change so that the
  resolver sees them together rather than one at a time.

- `scripts/bump_on_tail_2_projections.jl` is now Unicode NFC-normalised. It stored `Ã` as `A` plus
  a combining tilde on three lines, inherited from macOS rather than chosen. Nothing about what the
  script computes changes — Julia's parser normalises identifiers to NFC either way, and the base
  and normalised files parse to identical expression trees — but a `grep` pattern or an editor
  search typed in NFC now matches it, where before it silently matched nothing. No file under
  `src/` or `test/` was affected.

  The `X̃` and `x̃` in the same file still carry a combining tilde, and correctly so: a tilde over
  `X` or `x` has no precomposed codepoint, so NFC leaves them decomposed and the file is
  nonetheless fully normalised.

### Breaking Changes

- **The grid-based and particle-based code left this package.** `src/gridbased/` and
  `src/particles/` are gone, and with them every name they exported. A caller who reached
  `PoissonTensor`, `PoissonOperator` or `Arakawa` now wants `PoissonBrackets`;
  `PotentialReducedTensor` and `VelocityReducedMatrix` are in `VlasovMethods`;
  `_apply_Δₓ!`, `_apply_Δₓ₄!` and `_apply_Rₓ!` are in `PoissonSolvers`; `_apply_∫dv!` is in
  `VlasovMethods`; and `multiindex`, `linearindex` and `_stencil_indices` are in
  `MultiIndexArrays`. `_apply_P_ϕ!` and `_apply_P_h!` went to `PoissonBrackets` with the
  bracket they apply. Nothing was rewritten on the way — **every body is byte-identical to
  what stood here**, so the split reviews as a move.

  `ReducedTensor` **stays**, now in `src/reduced_tensor.jl`. Its `PT <: PoissonTensor{DT}`
  bound is what makes this package depend on `PoissonBrackets`: relaxing the bound without an
  interface would break `getindex`, which reaches through to `_stencil_indices`.

  `ReducedElectricField`, `DEIMElectricField`, `Snapshots`, `IntegratorParameters`,
  `ReducedIntegratorCache` and `reduced_integrate_vp` moved to `VlasovMethods/src/particles/`
  as files, but are **not yet reachable from there** — see *Open Issues*.

- **Minimum Julia is now 1.10**, raised from the declared 1.7. 1.10 is the LTS and the floor across
  the whole tree; 1.7 was declared but never tested and would not resolve against the current
  dependency versions. CI now derives its lower matrix entry from this field, so a declared floor
  that nobody tests is no longer possible.

## Open Issues

- **Two test files were orphaned by the split.** `test/runtests.jl` still includes
  `poisson_test.jl` and `bracket_operators_test.jl`, but the functions they exercise left this
  package: `_apply_∫dv!` is now in `VlasovMethods`, and `_apply_P_h!` and `_apply_P_ϕ!` are in
  `PoissonBrackets`. Both files were left in place rather than deleted or rewritten, because
  the tests themselves are worth keeping and belong with the code they test. Moving them is a
  later task. The suite cannot run in any case while the package does not resolve.
  Recorded 2026-09-17.

- **The package does not resolve.** `[compat] PoissonSolvers = "0.1, 0.2, 0.3"` is stale: the
  released `PoissonSolvers` is 0.5, and `VlasovMethods` requires `0.4, 0.5`, so the two cannot
  be satisfied together and `Pkg.resolve` reports *Unsatisfiable requirements*. This predates
  the split and is not fixed by it. `PoissonBrackets` is a second obstacle: it is unregistered,
  so it cannot be resolved from General at all, and it declares `julia = "1.11"`, above this
  package's declared 1.10 floor. Recorded 2026-09-17.

- **The package does not load.** Recorded 2026-08-31, when the cause was a `using VlasovMethods`
  with no matching `[deps]` entry. The split removed that line, and the cause is now the
  unsatisfiable resolve above; past it, `src/trainingset.jl`, `src/reducedbasis.jl` and
  `src/h5routines.jl` still name `Snapshots` and `IntegratorParameters` in type position, and
  both definitions left for `VlasovMethods`. Restoring the load needs those two reachable
  again, not a change here.
