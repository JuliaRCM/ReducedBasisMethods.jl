
h5file  = "temp.h5"

@testset "TrainingSet" begin

    nd = 2
    np = 10
    nh = 5
    ns = 11

    μ = Parameter(:μ, 0.0, 1.0, 3)
    ν = Parameter(:ν, 1.0, 1.0, 1)
    σ = Parameter(:σ, 0.0, 4.0, 2)

    ps = ParameterSpace(μ, ν, σ)

    X = rand(nd, np, ns, length(ps))
    V = rand(nd, np, ns, length(ps))
    A = rand(nd, np, ns, length(ps))
    Φ = rand(nd, nh, ns, length(ps))
    W = rand(ns, length(ps))
    K = rand(ns, length(ps))
    M = rand(ns, length(ps))


    ts1 = TrainingSet(ps, X, V, A, Φ, W, K, M)

    @test ts1.paramspace == ps
    
    @test ts1.X == X
    @test ts1.V == V
    @test ts1.A == A
    @test ts1.Φ == Φ
    @test ts1.W == W
    @test ts1.K == K
    @test ts1.M == M


    
    h5save(ts1, h5file; mode="w")
    @test isfile(h5file)

    ts2 = h5load(TrainingSet, h5file)
    rm(h5file)
    @test ts1 == ts2

end
