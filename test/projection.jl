using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedComplexityModeling: project
using Test

include("helpers/spectra.jl")

Random.seed!(41)

@testset "Symplectic identities" begin
    for T in (Float32, Float64)
        for (N, m) in ((200, 30), (40, 60))
            @testset "$T, N = $N, m = $m" begin
                # Rank-2 data like the paper's: `[q; p]` built from two N-vectors.
                S = cotangent_snapshot(low_rank_matrix(T, N, 2m, T[0.5, 0.25]))
                JN = symplectic_form(T, N)
                Id = Matrix{T}(I, 2N, 2N)

                for k in (5, 20)
                    Jk = symplectic_form(T, k)
                    for alg in (CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
                        @testset "$(nameof(typeof(alg))), k = $k" begin
                            b = ReducedBasis(alg, S, NamedTuple(), testparamspace(); k = k)
                            A = Matrix(b)

                            # Check 2 and check 4: `A⁺A = I`, `AᵀJ_N A = J_k`, and the symplectic
                            # inverse the blockwise `project` applies, all to `N·eps(T)` in the
                            # 2-norm. `J` is dense and built here, never formed in the package.
                            @test opnorm(A' * A - I) ≤ N * eps(T)
                            @test opnorm(A' * JN * A - Jk) ≤ N * eps(T)
                            @test opnorm(project(b, Id) - Jk' * A' * JN) ≤ N * eps(T)

                            if alg isa ComplexSVD
                                @test b.Ψ !== nothing
                                @test size(b.Ψ) == (N, k)
                            else
                                @test b.Ψ === nothing
                            end
                        end
                    end
                end

                # The assembled cotangent-lift basis is `diag(Φ, Φ)`.
                b = ReducedBasis(
                    CotangentLiftEVD(), S, NamedTuple(), testparamspace(); k = 5)
                A = Matrix(b)
                @test A[1:N, 1:5] == b.Φ
                @test A[(N + 1):(2N), 6:10] == b.Φ
                @test iszero(A[1:N, 6:10])
                @test iszero(A[(N + 1):(2N), 1:5])
            end
        end
    end
end

@testset "Reference projectors" begin
    for T in (Float32, Float64)
        σ10 = T.(2.0 .^ (-(1:10)))
        k = 5

        # Check 4: the complex route agrees with the reference `svd(Q + iP)` through `AAᵀ`.
        # A phase of a column rotates `(Φⱼ, Ψⱼ)`, and a basis built from `conj(U)` spans
        # `X - iV`; only the projector tells them apart.
        for (N, m) in ((200, 30), (40, 60))
            C = complex_low_rank_matrix(T, N, m, σ10)
            S = complex_snapshot(C)
            b = ReducedBasis(ComplexSVD(), S, NamedTuple(), testparamspace(); k = k)

            U = svd(C).U[:, 1:k]
            Ar = [real(U) -imag(U); imag(U) real(U)]
            @test opnorm(Matrix(b) * Matrix(b)' - Ar * Ar') ≤ √eps(T)
        end
    end
end

@testset "Round trip" begin
    for T in (Float32, Float64)
        σ10 = T.(2.0 .^ (-(1:10)))
        k = 5
        tail(σ) = sum(abs2, σ[(k + 1):end])

        for (N, m) in ((200, 30), (40, 60))
            @testset "$T, N = $N, m = $m" begin
                SP = low_rank_matrix(T, N, m, σ10)
                for alg in (PODSVD(), PODEVD())
                    b = ReducedBasis(alg, SP, NamedTuple(), testparamspace(); k = k)
                    @test abs(projection_error(b, SP)^2 - tail(σ10)) ≤ √eps(T) * tail(σ10)
                end

                M = low_rank_matrix(T, N, 2m, σ10)
                S = cotangent_snapshot(M)
                for alg in (CotangentLiftSVD(), CotangentLiftEVD())
                    b = ReducedBasis(alg, S, NamedTuple(), testparamspace(); k = k)
                    @test abs(projection_error(b, S)^2 - tail(σ10)) ≤ √eps(T) * tail(σ10)
                end

                C = complex_low_rank_matrix(T, N, m, σ10)
                SC = complex_snapshot(C)
                b = ReducedBasis(ComplexSVD(), SC, NamedTuple(), testparamspace(); k = k)
                @test abs(projection_error(b, SC)^2 - tail(svd(C).S)) ≤
                      √eps(T) * tail(svd(C).S)
            end
        end
    end
end

@testset "Projection and reduction error" begin
    for T in (Float32, Float64)
        σ10 = T.(2.0 .^ (-(1:10)))
        S = low_rank_matrix(T, 40, 60, σ10)
        k = 5
        b = ReducedBasis(PODSVD(), S, NamedTuple(), testparamspace(); k = k)

        # `projection_error` is the residual norm, and not some other measure of the basis.
        @test projection_error(b, S) == norm(S - lift(b, project(b, S)))

        # `reduction_error` lifts the reduced trajectory: it must not project `X` again.
        Z₀ = rand(T, k, 2)
        X = lift(b, Z₀)
        Z = copy(Z₀)
        Z[1, 1] += 0.37
        @test reduction_error(b, X, Z) ≈ 0.37
        @test reduction_error(b, X, Z₀) == 0

        # An `X` with a component outside the basis separates the two: the full-space residual is
        # larger than the reduced-coordinate difference, which a measurement made after `project`
        # would also give.
        W = S[:, 1:2] - lift(b, project(b, S[:, 1:2]))
        Xw = lift(b, Z₀) + W
        @test norm(W) > 0
        @test reduction_error(b, Xw, Z) ≈ hypot(reduction_error(b, X, Z), norm(W))

        # A vector and a matrix take the same route.
        x = S[:, 1]
        @test project(b, x) ≈ project(b, S)[:, 1]
        @test lift(b, project(b, x)) ≈ lift(b, project(b, S))[:, 1]
    end
end

@testset "Inference" begin
    for T in (Float32, Float64)
        σ10 = T.(2.0 .^ (-(1:10)))
        SP = low_rank_matrix(T, 40, 60, σ10)
        C = complex_low_rank_matrix(T, 40, 60, σ10)
        SC = complex_snapshot(C)
        M = low_rank_matrix(T, 40, 60, σ10)
        SL = cotangent_snapshot(M)

        for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
            S = alg isa ComplexSVD ? SC : (alg isa Union{PODSVD, PODEVD} ? SP : SL)
            b = ReducedBasis(alg, S, NamedTuple(), testparamspace(); k = 5)
            r = alg isa Union{PODSVD, PODEVD} ? 5 : 10

            x = rand(T, size(S, 1))
            X = rand(T, size(S, 1), 3)
            z = rand(T, r)
            Z = rand(T, r, 3)

            @test (@inferred project(b, x)) isa Vector{T}
            @test (@inferred project(b, X)) isa Matrix{T}
            @test (@inferred lift(b, z)) isa Vector{T}
            @test (@inferred lift(b, Z)) isa Matrix{T}
            @test (@inferred projection_error(b, X)) isa T
            @test (@inferred reduction_error(b, X, Z)) isa T
        end
    end
end
