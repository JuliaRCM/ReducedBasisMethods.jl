using LinearAlgebra
using Random
using ReducedBasisMethods
using Test

include("../helpers/spectra.jl")

Random.seed!(29)

"""
The tolerance check 5 fixes, `√(r₄·r₅)` with `r_k` the tail energy ratio of `σᵢ = 2⁻ⁱ`,
`i = 1:10`. It sits between `1 - c₄/E ≈ 3.9e-3` and `1 - c₅/E ≈ 9.8e-4`, so the energy rule keeps
five columns and no other number does.
"""
const TOLERANCE_45 = 1.9520e-3

@testset "SVD routes" begin
    for T in (Float32, Float64)
        for (N, m) in ((200, 30), (40, 60))
            @testset "$T, N = $N, m = $m" begin
                # Check 1: POD by SVD. The snapshot has rank 3, so its singular values are `σ`
                # followed by round-off, and every one of them is compared with the absolute Weyl
                # bound `max(N, m)·eps(T)·σ₁`, never with a relative one: `rtol = √eps(T)` accepts
                # a deviation a hundred Weyl bounds wide in Float64.
                σ = T[0.5, 0.25, 0.125]
                S = low_rank_matrix(T, N, m, σ)
                Ssaved = copy(S)

                b = ReducedBasis(PODSVD(), S, NamedTuple(), testparamspace(); k = 3)
                @test S == Ssaved
                @test length(b.Λ) == min(N, m)
                @test b.k == 3
                σref = [σ; zeros(T, length(b.Λ) - length(σ))]
                @test maximum(abs.(sqrt.(b.Λ) .- σref)) ≤ max(N, m) * eps(T) * σ[1]

                # Check 5, the SVD half: the spectrum σᵢ = 2⁻ⁱ, i = 1:10, and the energy rule
                # keeps k = 5 for both SVD routes.
                σ10 = T.(2.0 .^ (-(1:10)))
                S10 = low_rank_matrix(T, N, m, σ10)

                b5 = ReducedBasis(PODSVD(), S10, NamedTuple(), testparamspace();
                    tolerance = T(TOLERANCE_45))
                @test b5.k == 5
                @test b5.k ≤ length(b5.Λ)
                @test size(b5.Φ) == (N, 5)

                # a fixed k returns exactly k columns
                for k in (5, 20)
                    bk = ReducedBasis(PODSVD(), S10, NamedTuple(), testparamspace(); k = k)
                    @test bk.k == k
                    @test size(bk.Φ) == (N, k)
                end

                # A Float32 snapshot gives a Float32 basis: no promotion to Float64.
                @test eltype(b5.Λ) === T
                @test eltype(b5.Φ) === T
                @test b5.Ψ === nothing
            end
        end
    end

    # The complex route's spectrum is σ² of the complex snapshot, and Ψ is its imaginary part.
    for T in (Float32, Float64)
        σ10 = T.(2.0 .^ (-(1:10)))
        C = complex_low_rank_matrix(T, 40, 60, σ10)
        S = complex_snapshot(C)
        Ssaved = copy(S)
        b = ReducedBasis(ComplexSVD(), S, NamedTuple(), testparamspace(); k = 5)
        @test ReducedBasis(ComplexSVD(), S, NamedTuple(), testparamspace();
            tolerance = T(TOLERANCE_45)).k == 5
        @test S == Ssaved

        @test length(b.Λ) == min(40, 60)
        @test sqrt.(b.Λ[1:5]) ≈ σ10[1:5] rtol = √eps(T)
        @test size(b.Φ) == (40, 5)
        @test size(b.Ψ) == (40, 5)
        @test eltype(b.Λ) === T
        @test eltype(b.Φ) === T
        @test eltype(b.Ψ) === T
        @test Matrix(b)[1:40, 1:5] == b.Φ
        @test Matrix(b)[1:40, 6:10] == -b.Ψ
        @test Matrix(b)[41:80, 1:5] == b.Ψ
        @test Matrix(b)[41:80, 6:10] == b.Φ
    end
end
