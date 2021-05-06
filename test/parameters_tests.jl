
samples = [0.0, 1.0, 2.0, 3.0, 4.0]
grid    = [0.0 0.5 1.0 0.0 0.5 1.0;
           1.0 1.0 1.0 1.0 1.0 1.0;
           0.0 0.0 0.0 4.0 4.0 4.0]

@testset "Parameter" begin

    @test_throws AssertionError Parameter(:μ, 1.0, 4.0, samples)
    @test_throws AssertionError Parameter(:μ, 0.0, 3.0, samples)
    @test_throws AssertionError Parameter(:μ, 1.0, 0.0, 5)
    @test_throws AssertionError Parameter(:μ, 1.0, 0.0)

    p1 = Parameter(:μ, 0.0, 4.0, samples)
    p2 = Parameter(:μ, 0.0, 4.0, samples)
    p3 = Parameter(:μ, 0.0, 4.0, 5)
    p4 = Parameter(:μ, 0.0, 4.0)
    p5 = Parameter(:μ, 0.0, 4.0, Vector{Float64}())

    @test p1 == p2
    @test p1 == p3
    @test p1 != p4
    @test p4 == p5

    @test hash(p1) == hash(p2)
    @test hash(p1) == hash(p3)
    @test hash(p1) != hash(p4)
    @test hash(p4) == hash(p5)

    @test length(p1) == length(samples)
    @test length(p2) == length(samples)
    @test length(p3) == length(samples)
    @test length(p4) == 0
    @test length(p5) == 0

    @test size(p1) == size(samples)
    @test size(p2) == size(samples)
    @test size(p3) == size(samples)
    @test size(p4) == (0,)
    @test size(p5) == (0,)

    p1 = Parameter(:μ, 0.0, 1.0, 3)
    p2 = Parameter(:ν, 1.0, 1.0, 1)
    p3 = Parameter(:σ, 0.0, 4.0, 2)

    @test NamedTuple(p1, p2, p3) == NamedTuple{(:μ, :ν, :σ)}((p1, p2, p3))

    @test parameter_grid(p1, p2, p3) == grid

end

@testset "ParameterSpace" begin

    μ = Parameter(:μ, 0.0, 1.0, 3)
    ν = Parameter(:ν, 1.0, 1.0, 1)
    σ = Parameter(:σ, 0.0, 4.0, 2)

    p1 = ParameterSpace(NamedTuple(μ, ν, σ), parameter_grid(μ, ν, σ))
    p2 = ParameterSpace(μ, ν, σ)

    @test p1 == p2

    @test p1[:] == p2[:]

    @test p1[:μ] == p1[1] == p2[1] == p2[:μ]
    @test p1[:ν] == p1[2] == p2[2] == p1[:ν]
    @test p1[:σ] == p1[3] == p2[3] == p1[:σ]

    @test p1[:μ,:] == p1[1,:] == p2[1,:] == p2[:μ,:]
    @test p1[:ν,:] == p1[2,:] == p2[2,:] == p1[:ν,:]
    @test p1[:σ,:] == p1[3,:] == p2[3,:] == p1[:σ,:]

    for j in axes(grid,2)
        @test p1[:,j] == p2[:,j]

        @test p1[:μ,j] == p1[1,j] == p2[1,j] == p2[:μ,j]
        @test p1[:ν,j] == p1[2,j] == p2[2,j] == p1[:ν,j]
        @test p1[:σ,j] == p1[3,j] == p2[3,j] == p1[:σ,j]
    end
    
end
