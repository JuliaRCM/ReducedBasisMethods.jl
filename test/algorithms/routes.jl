using LinearAlgebra
using Random
using ReducedBasisMethods
using Test

include("../helpers/spectra.jl")

Random.seed!(37)

@testset "Cross-route checks" begin
    for T in (Float32, Float64)
        for (N, m) in ((200, 30), (40, 60))
            @testset "$T, N = $N, m = $m" begin
                σ10 = T.(2.0 .^ (-(1:10)))

                # Check 3: the two routes of a structure agree on the projector `ΦΦᵀ`, and so
                # does the reference `svd(·).U[:, 1:k]`. Columns are never compared, because the
                # sign of a singular vector is not fixed. The comparison is made only at a `k`
                # whose spectral gap carries the Gram route — the Gram matrix squares the
                # condition number — and the gap is asserted as the precondition.
                for k in (5, 6)
                    SP = low_rank_matrix(T, N, m, σ10)
                    bS = ReducedBasis(PODSVD(), SP, NamedTuple(), testparamspace(); k = k)
                    bE = ReducedBasis(PODEVD(), SP, NamedTuple(), testparamspace(); k = k)
                    Ur = svd(SP).U[:, 1:k]

                    @test (bS.Λ[k] - bS.Λ[k + 1]) / bS.Λ[1] ≥ √eps(T)
                    @test opnorm(bS.Φ * bS.Φ' - bE.Φ * bE.Φ') ≤ √eps(T)
                    @test opnorm(bS.Φ * bS.Φ' - Ur * Ur') ≤ √eps(T)

                    M = low_rank_matrix(T, N, 2m, σ10)
                    S = cotangent_snapshot(M)
                    cS = ReducedBasis(
                        CotangentLiftSVD(), S, NamedTuple(), testparamspace();
                        k = k)
                    cE = ReducedBasis(
                        CotangentLiftEVD(), S, NamedTuple(), testparamspace();
                        k = k)
                    Ur = svd(M).U[:, 1:k]

                    @test (cS.Λ[k] - cS.Λ[k + 1]) / cS.Λ[1] ≥ √eps(T)
                    @test opnorm(cS.Φ * cS.Φ' - cE.Φ * cE.Φ') ≤ √eps(T)
                    @test opnorm(cS.Φ * cS.Φ' - Ur * Ur') ≤ √eps(T)
                end

                # Check 5: the two routes give the same `k` for the same tolerance.
                SP = low_rank_matrix(T, N, m, σ10)
                M = low_rank_matrix(T, N, 2m, σ10)
                S = cotangent_snapshot(M)
                ks = [ReducedBasis(alg, X, NamedTuple(), testparamspace();
                          tolerance = T(1.9520e-3)).k
                      for (alg, X) in ((PODSVD(), SP), (PODEVD(), SP),
                    (CotangentLiftSVD(), S), (CotangentLiftEVD(), S))]
                @test all(==(5), ks)
            end
        end
    end
end
