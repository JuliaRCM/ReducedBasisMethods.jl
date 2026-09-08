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

### Bug Fixes

### Changed

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

- **Minimum Julia is now 1.10**, raised from the declared 1.7. 1.10 is the LTS and the floor across
  the whole tree; 1.7 was declared but never tested and would not resolve against the current
  dependency versions. CI now derives its lower matrix entry from this field, so a declared floor
  that nobody tests is no longer possible.

## Open Issues

- **The package does not load.** `src/ReducedBasisMethods.jl:12` does `using VlasovMethods`, but
  `VlasovMethods` appears in neither `[deps]` nor `Manifest.toml`, so loading fails immediately with
  `ArgumentError: Package VlasovMethods not found in current path`. Adding the dependency is not on
  its own enough: `VlasovMethods` does not currently load either, so the failure would chain.
  Recorded 2026-08-31.
