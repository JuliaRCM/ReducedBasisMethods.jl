using ReducedBasisMethods
using Test
using TOML

@testset "Generic-only skeleton" begin
    project = TOML.parsefile(joinpath(pkgdir(ReducedBasisMethods), "Project.toml"))

    # [deps] holds only generic infrastructure: no model package, no integrator
    permitted = ("GeometricBrackets", "HDF5", "LazyArrays", "LinearAlgebra",
        "MultiIndexArrays", "ReducedComplexityModeling", "Statistics")
    @test keys(project["deps"]) ⊆ permitted
end
