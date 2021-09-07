
grid    = [ 0.0  1.0  0.0
            0.5  1.0  0.0
            1.0  1.0  0.0
            0.0  1.0  4.0
            0.5  1.0  4.0
            1.0  1.0  4.0 ]

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
