using LinearAlgebra
using Random
using ReducedBasisMethods
using Test

include("helpers/spectra.jl")

Random.seed!(43)

@testset "Constructor guards" begin
    ps = testparamspace()
    S = low_rank_matrix(Float64, 40, 20, 2.0 .^ (-(1:5)))

    # `k` is at most `min(N, m)` for POD and the complex SVD, and `min(N, 2m)` for the cotangent
    # lift, because that is how many columns a snapshot of `N` rows admits — the Gram spectrum of
    # the EVD routes is longer than that and is stored whole.
    @test ReducedBasis(PODSVD(), S, NamedTuple(), ps; k = 20).k == 20
    @test size(Matrix(ReducedBasis(PODSVD(), S, NamedTuple(), ps; k = 20).Φ)) == (40, 20)
    @test_throws ArgumentError ReducedBasis(PODSVD(), S, NamedTuple(), ps; k = 21)

    # An out-of-range `k` is an `ArgumentError` for every `Integer` type, the ones whose
    # conversion to `Int` would throw `InexactError` first included: the range is checked in the
    # type `k` arrives in.
    for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        Ssmall = alg isa Union{PODSVD, PODEVD} ? low_rank_matrix(Float64, 4, 2, [1.0]) :
                 cotangent_snapshot(low_rank_matrix(Float64, 4, 2, [1.0]))
        for k in (typemax(UInt), big(typemax(Int)) + 1, -big(typemax(Int)) - 2)
            @test_throws ArgumentError ReducedBasis(alg, Ssmall, NamedTuple(), ps; k = k)
        end
    end

    # k = 0, tolerance outside [0, 1), and both or neither of the two keywords.
    @test_throws ArgumentError ReducedBasis(PODSVD(), S, NamedTuple(), ps; k = 0)
    @test_throws ArgumentError ReducedBasis(PODSVD(), S, NamedTuple(), ps; tolerance = 1)
    @test_throws ArgumentError ReducedBasis(PODSVD(), S, NamedTuple(), ps; tolerance = -0.1)
    @test_throws ArgumentError ReducedBasis(PODSVD(), S, NamedTuple(), ps)
    @test_throws ArgumentError ReducedBasis(
        PODSVD(), S, NamedTuple(), ps; k = 5, tolerance = 0.5)

    # tolerance 0 is accepted, and it never runs past the end of the spectrum.
    b0 = ReducedBasis(PODSVD(), S, NamedTuple(), ps; tolerance = 0.0)
    @test 1 ≤ b0.k ≤ length(b0.Λ)

    # An empty snapshot is refused before anything is decomposed. The `tolerance` form is what
    # separates the guard from the checks that come later: without it the spectrum is empty and
    # the energy rule runs off the end of it, which is a `BoundsError` and not an
    # `ArgumentError`.
    for (alg, S) in ((PODSVD(), zeros(0, 3)), (PODSVD(), zeros(3, 0)),
        (PODEVD(), zeros(0, 3)), (CotangentLiftSVD(), zeros(0, 4)),
        (CotangentLiftSVD(), zeros(4, 0)), (ComplexSVD(), zeros(4, 0)))
        @test_throws ArgumentError ReducedBasis(alg, S, NamedTuple(), ps; k = 1)
        @test_throws ArgumentError ReducedBasis(alg, S, NamedTuple(), ps; tolerance = 0.5)
    end

    # A PSD algorithm reduces the stacked `[q; p]`, which has an even number of rows.
    for alg in (CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        @test_throws ArgumentError ReducedBasis(alg, rand(41, 6), NamedTuple(), ps; k = 1)
        @test_throws ArgumentError ReducedBasis(alg, zeros(41, 0), NamedTuple(), ps; k = 1)
    end
end

@testset "Assembled basis" begin
    ps = testparamspace()
    σ10 = 2.0 .^ (-(1:10))
    M = low_rank_matrix(Float64, 40, 60, σ10)
    S = cotangent_snapshot(M)

    # `Matrix(b)` assembles U for POD, `diag(Φ, Φ)` for the cotangent lift and `[Φ −Ψ; Ψ Φ]` for
    # the complex SVD — the A of T7.4 and of the reduced equation forms.
    for alg in (PODSVD(), PODEVD())
        b = ReducedBasis(
            alg, low_rank_matrix(Float64, 40, 60, σ10), NamedTuple(), ps; k = 5)
        @test Matrix(b) == b.Φ
        @test size(Matrix(b)) == (40, 5)
    end

    b = ReducedBasis(CotangentLiftSVD(), S, NamedTuple(), ps; k = 5)
    @test size(Matrix(b)) == (80, 10)
    @test Matrix(b)[1:40, 1:5] == b.Φ
    @test Matrix(b)[41:80, 6:10] == b.Φ
    @test iszero(Matrix(b)[1:40, 6:10])
    @test iszero(Matrix(b)[41:80, 1:5])

    b = ReducedBasis(
        ComplexSVD(), complex_snapshot(complex_low_rank_matrix(Float64, 40, 60,
            σ10)),
        NamedTuple(), ps; k = 5)
    @test size(Matrix(b)) == (80, 10)
end

@testset "HDF5 round trip" begin
    ps = testparamspace()
    parameters = (α = 1.5, β = [1.0, 2.0])
    σ10 = 2.0 .^ (-(1:10))
    M = low_rank_matrix(Float64, 40, 60, σ10)

    mktempdir() do dir
        for (alg, S) in ((PODSVD(), low_rank_matrix(Float64, 40, 60, σ10)),
            (PODEVD(), low_rank_matrix(Float64, 40, 60, σ10)),
            (CotangentLiftSVD(), cotangent_snapshot(M)),
            (CotangentLiftEVD(), cotangent_snapshot(M)),
            (ComplexSVD(), complex_snapshot(complex_low_rank_matrix(Float64, 40, 60, σ10))))
            fpath = joinpath(dir, "$(nameof(typeof(alg))).h5")

            b1 = ReducedBasis(alg, S, parameters, ps; k = 5)
            h5save(fpath, b1; mode = "w")
            b2 = h5load(ReducedBasis, fpath)

            @test typeof(b2.algorithm) === typeof(b1.algorithm)
            @test b2.k == b1.k
            @test b2.Λ == b1.Λ
            @test b2.Φ == b1.Φ
            @test b2.parameters == b1.parameters
            @test b2.paramspace == b1.paramspace

            z = rand(alg isa Union{PODSVD, PODEVD} ? 5 : 10)
            @test lift(b2, z) == lift(b1, z)

            if alg isa ComplexSVD
                @test b2.Ψ == b1.Ψ
            else
                @test b2.Ψ === nothing
            end
        end
    end

    # A tag name the file format does not know is refused, and `POD` is not one of the five.
    @test_throws ArgumentError ReducedBasisMethods._algorithm("POD")
    @test_throws ArgumentError ReducedBasisMethods._algorithm("UnspecifiedAlgorithm")
end
