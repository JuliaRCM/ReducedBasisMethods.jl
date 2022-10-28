
h5file  = "temp.h5"

@testset "ReducedBasis" begin

    using ParticleMethods

    dt = 1e-1
    nd = 2
    np = 100
    nh = 10
    nt = 10
    nr = 23
    ne = 11
    p = 3
    L = 2π

    x = rand(nd, np)
    v = rand(nd, np)
    w = rand(1, np)

    particles = ParticleList(x,v,w)
    poisson = PoissonSolverPBSplines(p, nh, L)

    μ = Parameter(:μ, 0.0, 1.0, 3)
    ν = Parameter(:ν, 1.0, 1.0, 1)
    σ = Parameter(:σ, 0.0, 4.0, 2)

    pspace = ParameterSpace(μ, ν, σ)

    parameters = NamedTuple()

    integrator = IntegratorParameters(dt, nt, nt+1, nh, np, length(pspace))

    rb1 = ReducedBasis(CotangentLiftEVD(), parameters, pspace, particles, integrator, poisson,
                       rand(nr), rand(np,nr),
                       rand(ne), rand(np,ne),
                       rand(np,ne))

    @test rb1.parameters == parameters
    @test rb1.paramspace == pspace
    @test rb1.initconds == particles
    @test rb1.integrator == integrator
    @test rb1.poisson == poisson
    
    
    h5save(h5file, rb1; mode="w")
    @test isfile(h5file)

    rb2 = h5load(ReducedBasis, h5file)

    @test rb1.parameters == rb2.parameters
    @test rb1.paramspace == rb2.paramspace
    @test rb1.initconds == rb2.initconds
    @test rb1.integrator == rb2.integrator

    @test rb1.Λₚ == rb2.Λₚ
    @test rb1.kₚ == rb2.kₚ
    @test rb1.Ψₚ == rb2.Ψₚ
    @test rb1.Λₑ == rb2.Λₑ
    @test rb1.kₑ == rb2.kₑ
    @test rb1.Ψₑ == rb2.Ψₑ
    @test rb1.Πₑ == rb2.Πₑ

    rm(h5file)

end
