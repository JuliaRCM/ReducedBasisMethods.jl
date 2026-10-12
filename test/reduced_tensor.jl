using GeometricBrackets: Arakawa, PoissonTensor
using MultiIndexArrays
using MultiIndexArrays: GridTensor, MultiIndexLazyArray
using Random
using ReducedBasisMethods
using Test

# The width-`w` periodic neighbourhood test of `J` around `I`, written from the wrapped offsets by
# hand and never through `stencil_indices`, so that a defect in the stencil cannot reach the
# reference and the code under test together.
function in_neighbourhood(I::CartesianIndex, J::CartesianIndex, w::Int, g::Tuple)
    all(mod.(Tuple(J - I) .+ w, g) .<= 2w)
end

# The width-`w` `GridTensor` whose nonzero entries lie in the width-`w` neighbourhood of the first
# index and nowhere else: the contract of the locality interface, on a table that is not Arakawa's.
function neighbourhood_tensor(rng::AbstractRNG, ::Type{DT}, g::Tuple, w::Int) where {DT}
    N = g[1] * g[2]
    L = LinearIndices(g)
    C = rand(rng, DT, N, N, N) .- DT(0.5)
    return GridTensor(
        MultiIndexLazyArray(DT,
            (I, J, K) -> in_neighbourhood(I, J, w, g) && in_neighbourhood(I, K, w, g) ?
                         C[L[I], L[J], L[K]] : zero(DT), g, g, g),
        w)
end

# The bracket case. The closure is needed because `MultiIndexLazyArray` refuses a callable struct.
function arakawa_tensor(::Type{DT}, g::Tuple) where {DT}
    a = Arakawa(g[1], g[2], DT(0.7), DT(1.3))
    return GridTensor(MultiIndexLazyArray(DT, (I, J, K) -> a(I, J, K), g, g, g), 1)
end

# The reference: the double projection over every index pair of the materialised tensor.
function dense_projection(A::AbstractArray{DT, 3}, Pi, Pj) where {DT}
    [sum(A[m, n, k] * Pi[m, i] * Pj[n, j] for m in axes(A, 1), n in axes(A, 2))
     for i in axes(Pi, 2), j in axes(Pj, 2), k in axes(A, 3)]
end

# The same sum, with every entry outside the contract's neighbourhoods replaced by zero.
function contract_projection(A::AbstractArray{DT, 3}, Pi, Pj, w::Int, g::Tuple) where {DT}
    L = CartesianIndices(g)
    masked = [in_neighbourhood(L[m], L[n], w, g) && in_neighbourhood(L[m], L[k], w, g) ?
              A[m, n, k] : zero(DT) for m in axes(A, 1), n in axes(A, 2), k in axes(A, 3)]
    return dense_projection(masked, Pi, Pj)
end

bound(::Type{DT}, R) where {DT} = sqrt(eps(DT)) * maximum(abs, R)

# The allocations of the second call of `f(a, b)`, behind a barrier that specialises on every
# argument, so that only the call itself is measured.
allocations(f::F, a::A, b::B) where {F, A, B} = (f(a, b); @allocated f(a, b))

@testset "ReducedTensor" begin
    @testset "the reduced operator is the hand projection" begin
        for DT in (Float32, Float64)
            local nx, nv = 5, 6
            local N = nx * nv
            local Pi = rand(Xoshiro(2), DT, N, 3)
            local Pj = rand(Xoshiro(3), DT, N, 2)

            # the bracket case runs on 5 × 6 only: `Arakawa` needs nx, nv ≥ 3
            local gt = arakawa_tensor(DT, (nx, nv))
            local rt = ReducedTensor(gt, Pi, Pj)
            local R = dense_projection(Array(gt), Pi, Pj)

            @test size(rt) == (3, 2, N)
            @test maximum(abs, R) > 0

            # the coefficients are Arakawa's integer sign table over `12 hx hv`, whose largest
            # entry is 1: the spacing the closure chose is pinned here and a wrong one fails,
            # where the projection comparison alone cannot see it
            @test maximum(abs, Array(gt)) ≈ inv(DT(0.7)) * inv(DT(1.3)) / 12
            @test maximum(abs, rt .- R) ≤ bound(DT, R)

            # the second, non-bracket tensor: one caller does not demonstrate generality
            for w in (1, 2)
                local g = (nx, nv)
                local n = g[1] * g[2]
                local Qi = rand(Xoshiro(2), DT, n, 3)
                local Qj = rand(Xoshiro(3), DT, n, 2)
                local gt2 = neighbourhood_tensor(Xoshiro(1), DT, g, w)
                local rt2 = ReducedTensor(gt2, Qi, Qj)
                local R2 = dense_projection(Array(gt2), Qi, Qj)

                @test size(rt2) == (3, 2, n)
                @test maximum(abs, R2) > 0
                @test maximum(abs, rt2 .- R2) ≤ bound(DT, R2)
            end
        end
    end

    @testset "the entries the width-1 stencil of k alone would drop exist" begin
        # Nonzero entries with `K` outside the width-1 neighbourhood of `J` distinguish the
        # nested neighbourhoods from restricting both summation indices to the stencil of `k`.
        for DT in (Float32, Float64)
            local g = (5, 6)
            local A = Array(neighbourhood_tensor(Xoshiro(1), DT, g, 1))
            local L = CartesianIndices(g)
            local displaced = count(!iszero(A[i, j, k]) &&
                                        !in_neighbourhood(L[j], L[k], 1, g)
            for i in axes(A, 1), j in axes(A, 2), k in axes(A, 3))
            @test displaced > 0
        end
    end

    @testset "the stencil branch restricts the sum, on a contract the tensor breaks" begin
        # the all-ones tensor breaks the locality contract on purpose, so that a `getindex` that
        # always sums the whole axis is caught instead of passing on both sides
        for DT in (Float32, Float64)
            local g = (5, 6)
            local N = g[1] * g[2]
            local Pi = rand(Xoshiro(2), DT, N, 3)
            local Pj = rand(Xoshiro(3), DT, N, 2)
            local ones_gt = GridTensor(
                MultiIndexLazyArray(
                    DT, (I, J, K) -> one(DT), g, g, g), 1)
            local rt = ReducedTensor(ones_gt, Pi, Pj)
            local A = Array(ones_gt)
            local restricted = contract_projection(A, Pi, Pj, 1, g)
            local dense = dense_projection(A, Pi, Pj)

            # the test cannot be vacuous: the restricted reference differs from the dense sum
            @test maximum(abs, restricted .- dense) > sqrt(eps(DT)) * maximum(abs, dense)
            @test maximum(abs, rt .- restricted) ≤ bound(DT, restricted)
        end
    end

    @testset "the constructor refuses what it cannot project" begin
        for DT in (Float32, Float64)
            local N = 30
            local Pi = rand(Xoshiro(2), DT, N, 3)
            local Pj = rand(Xoshiro(3), DT, N, 2)
            local gt = arakawa_tensor(DT, (5, 6))

            local short = rand(Xoshiro(5), DT, N - 1, 2)
            @test_throws DimensionMismatch ReducedTensor(gt, rand(Xoshiro(4), DT, N + 1, 3), Pj)
            @test_throws DimensionMismatch ReducedTensor(gt, Pi, short)

            # an integer projection is refused by the same element-type rule
            @test_throws ArgumentError ReducedTensor(gt, ones(Int, N, 3), Pj)

            # a direction shorter than `2w + 1` makes the stencil name a node twice; the 4 × 5
            # grid is long enough for width 1, so the width is read from the tensor
            for (g, w) in (((2, 5), 1), ((4, 5), 2))
                local n = g[1] * g[2]
                local gt2 = neighbourhood_tensor(Xoshiro(1), DT, g, w)
                @test_throws ArgumentError ReducedTensor(
                    gt2, rand(Xoshiro(2), DT, n, 3), rand(Xoshiro(3), DT, n, 2))
            end
            local gt3 = neighbourhood_tensor(Xoshiro(1), DT, (4, 5), 1)
            @test ReducedTensor(gt3, rand(Xoshiro(2), DT, 20, 3), rand(Xoshiro(3), DT, 20, 2)) isa
                  ReducedTensor
        end

        # the element type of the projections is the tensor's, not a promoted one
        local gt32 = arakawa_tensor(Float32, (5, 6))
        local Pi32 = rand(Xoshiro(2), Float32, 30, 3)
        local Pj32 = rand(Xoshiro(3), Float32, 30, 2)
        @test_throws ArgumentError ReducedTensor(gt32, Float64.(Pi32), Pj32)
        @test_throws ArgumentError ReducedTensor(gt32, Pi32, Float64.(Pj32))

        # only a rank-3 `GridTensor` is a `ReducedTensor`
        local Pi = rand(Xoshiro(2), Float64, 30, 3)
        local Pj = rand(Xoshiro(3), Float64, 30, 2)
        local a = Arakawa(5, 6, 0.7, 1.3)
        @test_throws MethodError ReducedTensor(PoissonTensor(Float64, 5, 6, a), Pi, Pj)
        @test_throws MethodError ReducedTensor(zeros(Float64, 30, 30, 30), Pi, Pj)
        @test_throws MethodError ReducedTensor(
            GridTensor(
                MultiIndexLazyArray(Float64, (I, J) -> one(Float64),
                    (5, 6), (5, 6)), 1), Pi, Pj)
        @test_throws MethodError ReducedTensor(
            GridTensor(
                MultiIndexLazyArray(Float64, (I, J, K, L) -> one(Float64),
                    (3, 4), (3, 4), (3, 4), (3, 4)),
                1), Pi, Pj)
    end

    @testset "an index off the grid throws the reduced tensor's own BoundsError" begin
        for DT in (Float32, Float64)
            local N = 30
            local gt = arakawa_tensor(DT, (5, 6))
            local rt = ReducedTensor(gt, rand(Xoshiro(2), DT, N, 3), rand(Xoshiro(3), DT, N, 2))

            for (i, j, k) in ((0, 1, 1), (size(rt, 1) + 1, 1, 1), (1, 1, size(rt, 3) + 1))
                @test_throws BoundsError(rt, (i, j, k)) rt[i, j, k]
            end
        end
    end

    @testset "the entries, their type and their index forms" begin
        for DT in (Float32, Float64)
            local N = 30
            local g = (5, 6)
            local gt = arakawa_tensor(DT, g)
            local Pi = rand(Xoshiro(2), DT, N, 3)
            local Pj = rand(Xoshiro(3), DT, N, 2)
            local rt = ReducedTensor(gt, Pi, Pj)

            # the accumulator stays in `DT`, which `eltype(Array(rt))` cannot see
            @test @inferred(rt[2, 1, 7]) isa DT

            # another integer type and a linear index go through `AbstractArray`'s fallback
            @test rt[Int32(2), Int32(1), Int32(7)] == rt[2, 1, 7]
            @test rt[7] == Array(rt)[7]

            # any `AbstractMatrix` of `DT` is a projection
            local wide = rand(Xoshiro(6), DT, N, 4)
            @test ReducedTensor(gt, view(wide, :, 1:3), Pj)[2, 1, 7] ==
                  ReducedTensor(gt, wide[:, 1:3], Pj)[2, 1, 7]
            local Q = rand(Xoshiro(7), DT, 2, N)
            @test ReducedTensor(gt, Pi, adjoint(Q))[2, 1, 7] ==
                  ReducedTensor(gt, Pi, copy(adjoint(Q)))[2, 1, 7]
        end
    end

    @testset "the entries allocate nothing" begin
        # `getindex` is called on every right-hand side evaluation downstream; the `> 0` control
        # shows that the barrier measures
        for DT in (Float32, Float64), w in (1, 2)

            local g = (5, 6)
            local n = g[1] * g[2]
            local gt = neighbourhood_tensor(Xoshiro(1), DT, g, w)
            local rt = ReducedTensor(gt, rand(Xoshiro(2), DT, n, 3), rand(Xoshiro(3), DT, n, 2))
            local out = Array{DT}(undef, size(rt))
            @test allocations((r, o) -> r[2, 1, 7], rt, out) == 0
            @test allocations((r, o) -> copyto!(o, r), rt, out) == 0
            @test allocations((r, o) -> collect(r), rt, out) > 0
        end
    end

    @testset "the module reaches MultiIndexArrays only through its public names" begin
        @test !isdefined(ReducedBasisMethods, :MultiIndexArrays)
        local private = filter(
            n -> isdefined(ReducedBasisMethods, n) &&
                 Base.binding_module(ReducedBasisMethods, n) === MultiIndexArrays,
            setdiff(names(MultiIndexArrays; all = true), names(MultiIndexArrays)))
        @test isempty(private)
    end
end
