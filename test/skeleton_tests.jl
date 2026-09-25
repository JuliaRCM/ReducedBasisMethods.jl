using Aqua
using TOML

@testset "Generic-only skeleton" begin
    project = TOML.parsefile(joinpath(pkgdir(ReducedBasisMethods), "Project.toml"))

    # [deps] holds only generic infrastructure: no model package, no integrator
    permitted = ("GeometricBrackets", "GeometricEquations", "HDF5", "LazyArrays",
        "LinearAlgebra", "MultiIndexArrays", "ReducedComplexityModeling", "Statistics")
    @test keys(project["deps"]) ⊆ permitted

    # the integrator is a test-only dependency
    @test haskey(project["extras"], "GeometricIntegratorsBase")
    @test "GeometricIntegratorsBase" ∈ project["targets"]["test"]

    Aqua.test_stale_deps(ReducedBasisMethods)
    Aqua.test_undefined_exports(ReducedBasisMethods)
end
