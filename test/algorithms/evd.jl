using LinearAlgebra
using Random
using ReducedBasisMethods
using Test

include("../helpers/spectra.jl")

Random.seed!(31)

@testset "EVD routes" begin
    for T in (Float32, Float64)
        for (N, m) in ((200, 30), (40, 60))
            @testset "$T, N = $N, m = $m" begin
                # The Gram spectrum is sorted by value, never by magnitude, and it is cut to the
                # basis limit `min(N, 2m)`: past the rank of the snapshot the remaining
                # eigenvalues are round-off.
                S = cotangent_snapshot(low_rank_matrix(T, N, 2m, T[0.5, 0.25]))
                b = ReducedBasis(
                    CotangentLiftEVD(), S, NamedTuple(), testparamspace(); k = 2)
                @test issorted(b.Λ; rev = true)
                @test length(b.Λ) == min(N, 2m)
                @test b.k == 2

                # Check 5, the EVD half: the same energy rule and the same k as the SVD routes.
                σ10 = T.(2.0 .^ (-(1:10)))
                S10 = cotangent_snapshot(low_rank_matrix(T, N, 2m, σ10))
                b5 = ReducedBasis(CotangentLiftEVD(), S10, NamedTuple(), testparamspace();
                    tolerance = T(1.9520e-3))
                @test b5.k == 5

                # The POD EVD route cuts the Gram spectrum to `min(N, m)`, and its leading
                # singular values are recovered to `√eps` where the squared condition number
                # allows it — the first five, the ones the gap precondition of check 3 admits.
                SP = low_rank_matrix(T, N, m, σ10)
                bp = ReducedBasis(PODEVD(), SP, NamedTuple(), testparamspace(); k = 5)
                @test length(bp.Λ) == min(N, m)
                @test bp.k == 5
                @test sqrt.(bp.Λ[1:5]) ≈ σ10[1:5] rtol = √eps(T)

                # A Float32 snapshot gives a Float32 basis.
                @test eltype(bp.Λ) === T
                @test eltype(bp.Φ) === T
                @test bp.Ψ === nothing
            end
        end

        # The rank-2 EVD case, N ≫ m: the Gram matrix is not cut, so its round-off eigenvalues
        # of both signs are all in the spectrum a caller plots.
        S = cotangent_snapshot(low_rank_matrix(T, 200, 60, T[0.5, 0.25]))
        b = ReducedBasis(CotangentLiftEVD(), S, NamedTuple(), testparamspace(); k = 2)
        @test length(b.Λ) == 60
        @test any(<(0), b.Λ)
        @test issorted(b.Λ; rev = true)
    end

    @testset "basis size" begin
        # The clamp: a negative eigenvalue carries no energy. Without `max.(Λ, 0)` the total
        # energy is 4.5 rather than 5, and the rule keeps one column instead of two.
        @test ReducedBasisMethods._basis_size([4.0, 1.0, -0.5], nothing, 0.15) == 2

        # tolerance 0 keeps the first index that reaches the full energy, and never runs past the
        # end of the spectrum.
        @test ReducedBasisMethods._basis_size([4.0, 1.0, -0.5], nothing, 0.0) == 2
        @test ReducedBasisMethods._basis_size(Float32[4, 0], nothing, 0.0f0) == 1

        # k of any `Integer` is converted to `Int`.
        @test ReducedBasisMethods._basis_size([1.0, 2.0, 3.0], Int32(2), nothing) === 2
        @test ReducedBasisMethods._basis_size([1.0, 2.0, 3.0], UInt8(3), nothing) === 3

        # The energy rule is relative: scaling the snapshot scales the spectrum, and leaves k.
        S = low_rank_matrix(Float64, 60, 20, 2.0 .^ (-(1:6)))
        ks = [ReducedBasis(
                  PODSVD(), s .* S, NamedTuple(), testparamspace(); tolerance = 1e-3).k
              for s in (1.0, 3.7e8, 3.7e-8)]
        @test all(==(ks[1]), ks)
    end
end
