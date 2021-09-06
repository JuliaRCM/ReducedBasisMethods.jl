
samples = [0.0, 1.0, 2.0, 3.0, 4.0]
grid    = [ 0.0  1.0  0.0
            0.5  1.0  0.0
            1.0  1.0  0.0
            0.0  1.0  4.0
            0.5  1.0  4.0
            1.0  1.0  4.0 ]

@testset "Parameter" begin

    @test_throws AssertionError Parameter(:μ, 1.0, 4.0, samples)
    @test_throws AssertionError Parameter(:μ, 0.0, 3.0, samples)
    @test_throws AssertionError Parameter(:μ, 1.0, 0.0, 5)
    @test_throws AssertionError Parameter(:μ, 1.0, 0.0)

    p1 = Parameter(:μ, 0.0, 4.0, samples)
    p2 = Parameter(:μ, 0.0, 4.0, samples)
    p3 = Parameter(:μ, 0.0, 4.0, 5)
    p4 = Parameter(:μ, 0.0, 4.0)
    p5 = Parameter(:μ, 0.0, 4.0, nothing)

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

    @test hassamples(p1) == true
    @test hassamples(p2) == true
    @test hassamples(p3) == true
    @test hassamples(p4) == false
    @test hassamples(p5) == false

    @test collect(p1) == samples
    @test collect(p2) == samples
    @test collect(p3) == samples
    @test collect(p4) === nothing
    @test collect(p5) === nothing

    @test minimum(p1) == 0.0
    @test minimum(p2) == 0.0
    @test minimum(p3) == 0.0
    @test minimum(p4) == 0.0
    @test minimum(p5) == 0.0
    
    @test maximum(p1) == 4.0
    @test maximum(p2) == 4.0
    @test maximum(p3) == 4.0
    @test maximum(p4) == 4.0
    @test maximum(p5) == 4.0

    p1 = Parameter(:μ, 0.0, 1.0, 3)
    p2 = Parameter(:ν, 1.0, 1.0, 1)
    p3 = Parameter(:σ, 0.0, 4.0, 2)

    @test NamedTuple(p1, p2, p3) == NamedTuple{(:μ, :ν, :σ)}((p1, p2, p3))

end

@testset "ParameterSpace" begin

    μ = Parameter(:μ, 0.0, 1.0, 3)
    ν = Parameter(:ν, 1.0, 1.0, 1)
    σ = Parameter(:σ, 0.0, 4.0, 2)

    pkeys = (:μ, :ν, :σ)
    params = (μ, ν, σ)

    p1 = ParameterSpace(NamedTuple(params...), sample(CartesianParameterSampler(), params...))
    p2 = ParameterSpace(NamedTuple(params...))
    p3 = ParameterSpace(params...)

    @test p1 == p2 == p3
    @test p1[:] == p2[:] == p3[:] == grid
    @test p1[:,:] == p2[:,:] == p3[:,:] == grid

    for i in axes(grid,1)
        @test p1(i) == p2(i) == p3(i) == NamedTuple{pkeys}(Tuple(grid[i,:]))
        @test p1[i] == p2[i] == p3[i] == grid[i,:]
        @test p1[i,:] == p2[i,:] == p3[i,:] == grid[i,:]
    end

    for j in axes(grid,2)
        @test p1[:,j] == p2[:,j] == p3[:,j] == grid[:,j]
        @test p1[:,pkeys[j]] == p2[:,pkeys[j]] == p3[:,pkeys[j]] == grid[:,j]
        @test p1[pkeys[j]] == p2[pkeys[j]] == p3[pkeys[j]] == grid[:,j]
    end

    for i in axes(grid,1)
        for j in axes(grid,2)
            @test p1[i,j] == p2[i,j] == p3[i,j] == grid[i,j]
            @test p1[i,pkeys[j]] == p2[i,pkeys[j]] == p3[i,pkeys[j]] == grid[i,j]
        end
    end

end
