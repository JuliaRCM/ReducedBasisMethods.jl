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

# The mixed-input fixture. The no-power-of-2 condition is on the `Float64` input,
# which a `Float32` basis reads: none of its entries is representable in `Float32`
# (`1 + 1e-10` rounds to `1`), so a projection carried through that precision misses the
# bound below. The `Float32` input, read by a `Float64` basis, holds `2` and `4`.
mixed_input(::Type{Float64}, n::Int) = 1 .+ 1e-10 .* (1:n)
mixed_input(::Type{Float32}, n::Int) = Float32(1) .+ Float32(0.1) .* (1:n)

@testset "Mixed element types" begin
    # `project` returns the element type of `b.Φ' * x`, and not the element type of the basis: the
    # coordinates of a `Float32` snapshot read in `Float64` are `Float64`.
    for (BT, XT) in ((Float32, Float64), (Float64, Float32))
        @testset "$(BT) basis, $(XT) input" begin
            N, m, k = 40, 20, 5
            rtol = 2N * eps(Float64)
            σ = BT.(2.0 .^ (-(1:10)))
            SP = low_rank_matrix(BT, N, m, σ)
            SL = cotangent_snapshot(low_rank_matrix(BT, N, 2m, σ))
            SC = complex_snapshot(complex_low_rank_matrix(BT, N, m, σ))

            for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(),
                ComplexSVD())
                S = alg isa ComplexSVD ? SC : (alg isa Union{PODSVD, PODEVD} ? SP : SL)
                b = ReducedBasis(alg, S, NamedTuple(), testparamspace(); k = k)
                v = mixed_input(XT, size(S, 1))
                X = v .* [XT(1) XT(2) XT(3)]

                @testset "$(nameof(typeof(alg)))" begin
                    for x in (X, v)
                        @test eltype(project(b, x)) === Float64
                        @test eltype(lift(b, project(b, x))) === Float64
                        @test (@inferred project(b, x)) ≈ Matrix(b)' * x rtol = rtol
                    end
                end
            end
        end
    end
end

@testset "Mismatched domains" begin
    # The blockwise maps read the full-space and the reduced coordinates in blocks, so a row count
    # that does not match the basis is refused before the read, and not sliced away in silence.
    for alg in (CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        b = ReducedBasis(alg, ones(6, 2), NamedTuple(), testparamspace(); k = 1)  # N = 3, k = 1

        @test size(project(b, ones(6, 1))) == (2, 1)
        @test size(lift(b, ones(2, 1))) == (6, 1)
        for n in (5, 7)
            @test_throws DimensionMismatch project(b, ones(n, 1))
            @test_throws DimensionMismatch project(b, ones(n))
        end
        for n in (1, 3)
            @test_throws DimensionMismatch lift(b, ones(n, 1))
            @test_throws DimensionMismatch lift(b, ones(n))
        end
    end
end

# A concrete barrier: the arguments' types are known, and the call before the measurement warms
# the method.
allocations(f::F, b::B, x::X) where {F, B, X} = (f(b, x); @allocated f(b, x))

@testset "Projection allocation" begin
    # `project` is called once per integration step and its output is `2k × p` whatever the number
    # of rows of the snapshot. A tenfold increase of those rows must not cost more than the output
    # does, or the call copies what it reads. The barrier, the warm call, one BLAS thread and the
    # minimum of ten repeats are what make the count reproducible; the 2 KiB slack is far below one
    # copied `1800 × 3` block.
    threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        for T in (Float32, Float64)
            for alg in (CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
                bytes = Int[]
                for N in (200, 2000)
                    b = ReducedBasis(
                        alg, randn(T, 2N, 20), NamedTuple(), testparamspace(); k = 5)
                    X = randn(T, 2N, 3)
                    push!(bytes, minimum(allocations(project, b, X) for _ in 1:10))
                end
                @test abs(last(bytes) - first(bytes)) ≤ 2048
            end
        end

        # The barrier sees an allocation.
        b = ReducedBasis(PODSVD(), randn(200, 20), NamedTuple(), testparamspace(); k = 5)
        @test allocations((b, x) -> collect(x), b, randn(200, 3)) > 0
    finally
        BLAS.set_num_threads(threads)
    end
end
