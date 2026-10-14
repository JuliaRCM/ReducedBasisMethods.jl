using ReducedBasisMethods
using Documenter: DocMeta, doctest
using Test

# The doctests of this package, in its docstrings and in the manual under `docs/src`, as the
# `Doctests` job of `CI.yml` runs them: `manual = false`, because the manual pages carry no
# `jldoctest` of their own and their `@meta` block is evaluated in `Main`.
#
# Documenter evaluates a page's `@meta` block in `Main`, and a `@safetestset` file runs in a module
# of its own, so `ReducedBasisMethods` is imported into `Main` first.
@eval Main import ReducedBasisMethods

DocMeta.setdocmeta!(ReducedBasisMethods, :DocTestSetup,
    :(using ReducedBasisMethods); recursive = true)

doctest(ReducedBasisMethods; manual = false)
