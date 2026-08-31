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
