using ReducedBasisMethods
using Documenter: DocMeta, doctest

# Run the package's docstring doctests. `manual = false` excludes the manual's doctests, which
# `docs/make.jl` runs during the build.
# Documenter evaluates page metadata in Main, so import the package there as well.
@eval Main import ReducedBasisMethods

DocMeta.setdocmeta!(ReducedBasisMethods, :DocTestSetup,
    :(using ReducedBasisMethods); recursive = true)

doctest(ReducedBasisMethods; manual = false)
