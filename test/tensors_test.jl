@testset "Tensor Tests" begin

    # Build a small Poisson Tensor, and check if P(ϕ) == P̃(ϕ̄)

    Random.seed!(123)
    n₁ = 8
    n₂ = 12
    n = n₁*n₂
    x = range(0,1,length=n₁)
    v = range(-1,1,length=n₂)
    h₁ = x[2]-x[1]
    h₂ = v[2]-v[1]

    m = 8
    m₁ = 3

    ϕ̃ = rand(m₁)
    f̃ = rand(m)
    g̃ = rand(m)

    Ψf = rand(n,m)
    Ψϕ = rand(n₁,m₁)

    ϕ = Ψϕ * ϕ̃
    f = Ψf * f̃
    g = Ψf * g̃
    h = vec([ _ϕ + _v^2/2 for _ϕ in ϕ, _v in v ])

    P = PoissonTensor( Float64, n₁, n₂, Arakawa(n₁, n₂, h₁, h₂) )
    P̃ = ReducedTensor( P, Ψf, Ψf )
    Ph = PoissonOperator(P, h)

    P̃₁ = PotentialReducedTensor(P, Ψf, Ψf, Ψϕ)
    P̃₂ = VelocityReducedMatrix(P, Ψf, Ψf, v)

    gPₕf = 0
    for i in 1:n, j in 1:n, k in 1:n
        gPₕf += P[i,j,k] * g[i] * f[j] * h[k]
    end

    # test PoissonOperator
    @test abs( gPₕf - dot(g,Ph,f) ) / abs(dot(g,Ph,f)) < 1e-14

    # test ReducedTensor
    g̃P̃ₕf̃ = 0
    for i in 1:m, j in 1:m, k in 1:n
        g̃P̃ₕf̃ += P̃[i,j,k] * g̃[i] * f̃[j] * h[k]
    end
    @test abs( gPₕf - g̃P̃ₕf̃ ) / abs(gPₕf) < 1e-14

    # test fully reduced PotentialReducedTensor and VelocityReducedMatrix
    g̃P̃ϕf̃ = dot(g̃, P̃₂, f̃)
    for i in 1:m, j in 1:m, k in 1:m₁
        g̃P̃ϕf̃ += P̃₁[i,j,k] * g̃[i] *f̃[j] * ϕ̃[k] 
    end
    @test abs( g̃P̃ϕf̃ - g̃P̃ₕf̃ ) / abs(g̃P̃ϕf̃) < 1e-14

end