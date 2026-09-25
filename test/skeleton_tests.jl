using Aqua
using TOML

@testset "Generic-only skeleton" begin
    project = TOML.parsefile(joinpath(pkgdir(ReducedBasisMethods), "Project.toml"))

    # [deps] holds only generic infrastructure: no model package, no integrator
    permitted = ("GeometricBrackets", "HDF5", "LazyArrays", "LinearAlgebra",
        "MultiIndexArrays", "ReducedComplexityModeling", "Statistics")
    @test keys(project["deps"]) ⊆ permitted

    Aqua.test_all(ReducedBasisMethods)
end
