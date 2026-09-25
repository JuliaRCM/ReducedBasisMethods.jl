
h5file = "temp.h5"

using ReducedComplexityModeling: Parameter, ParameterSpace

@testset "ReducedBasis" begin
    np = 100
    nr = 23
    ne = 11

    μ = Parameter(:μ, 0.0, 1.0, 3)
    ν = Parameter(:ν, 1.0, 1.0, 1)
    σ = Parameter(:σ, 0.0, 4.0, 2)

    pspace = ParameterSpace(μ, ν, σ)

    parameters = NamedTuple()

    rb1 = ReducedBasis(
        CotangentLiftEVD(), parameters, pspace,
        rand(nr), rand(np, nr),
        rand(ne), rand(np, ne),
        rand(np, ne))

    @test rb1.parameters == parameters
    @test rb1.paramspace == pspace

    h5save(h5file, rb1; mode = "w")
    @test isfile(h5file)

    rb2 = h5load(ReducedBasis, h5file)

    @test rb1.parameters == rb2.parameters
    @test rb1.paramspace == rb2.paramspace

    @test rb1.Λₚ == rb2.Λₚ
    @test rb1.kₚ == rb2.kₚ
    @test rb1.Ψₚ == rb2.Ψₚ
    @test rb1.Λₑ == rb2.Λₑ
    @test rb1.kₑ == rb2.kₑ
    @test rb1.Ψₑ == rb2.Ψₑ
    @test rb1.Πₑ == rb2.Πₑ

    rm(h5file)
end
