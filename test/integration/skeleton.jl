using ReducedBasisMethods
using Test
using TOML

@testset "Generic-only skeleton" begin
    project = TOML.parsefile(joinpath(pkgdir(ReducedBasisMethods), "Project.toml"))

    # [deps] holds only generic infrastructure: no model package, no integrator
    permitted = ("GeometricEquations", "HDF5", "LinearAlgebra",
        "MultiIndexArrays", "ReducedComplexityModeling")
    @test keys(project["deps"]) ⊆ permitted

    # the growth-rate fit of a Vlasov run is the model's, not this package's
    @test !isdefined(ReducedBasisMethods, :get_regression_αβ)
    @test !isdefined(ReducedBasisMethods, :find_maxima)
end
