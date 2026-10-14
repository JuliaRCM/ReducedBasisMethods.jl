using ReducedBasisMethods
using Documenter: DocMeta, doctest
using Test

# Run the package's docstring doctests, matching the CI Doctests job.
# `manual = false` excludes manual doctests; `docs/make.jl` checks them during the build.
# Documenter evaluates page metadata in Main, so import the package there as well.
@eval Main import ReducedBasisMethods

DocMeta.setdocmeta!(ReducedBasisMethods, :DocTestSetup,
    :(using ReducedBasisMethods); recursive = true)

doctest(ReducedBasisMethods; manual = false)
