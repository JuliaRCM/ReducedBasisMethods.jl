
h5file  = "temp.h5"

@testset "TrainingSet" begin

    using ParticleMethods
    using PoissonSolvers

    dt = 1e-1
    nd = 2
    np = 100
    nh = 10
    nt = 10
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

    ts1 = TrainingSet(particles, poisson, nd, nt+1, parameters, pspace, integrator)

    for p in eachindex(pspace)
        # copy solution
        ts1.snapshots.X[:,:,:,p] .= rand(nd, np, nt+1)
        ts1.snapshots.V[:,:,:,p] .= rand(nd, np, nt+1)
        ts1.snapshots.A[:,:,:,p] .= rand(nd, np, nt+1)
        ts1.snapshots.Φ[:,:,:,p] .= rand(nd, nh, nt+1)

        # copy diagnostics
        ts1.snapshots.W[:,p] .= rand(nt+1)
        ts1.snapshots.K[:,p] .= rand(nt+1)
        ts1.snapshots.M[:,p] .= rand(nt+1)
    end    

    @test ts1.parameters == parameters
    @test ts1.paramspace == pspace
    @test ts1.initconds == particles
    @test ts1.integrator == integrator
    @test ts1.poisson == poisson
    
    
    h5save(h5file, ts1; mode="w")
    @test isfile(h5file)

    ts2 = h5load(TrainingSet, h5file)

    @test ts1.parameters == ts2.parameters
    @test ts1.paramspace == ts2.paramspace
    @test ts1.initconds.list == ts2.initconds.list
    @test ts1.snapshots == ts2.snapshots
    @test ts1.integrator == ts2.integrator

    rm(h5file)

end
