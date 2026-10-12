using LinearAlgebra
using Random
using ReducedBasisMethods
using Test

include("helpers/spectra.jl")

Random.seed!(43)

"A `Float64` snapshot whose entries throw on any read."
struct Unreadable <: AbstractMatrix{Float64}
    n::Int
    m::Int
end
Base.size(S::Unreadable) = (S.n, S.m)
Base.getindex(::Unreadable, ::Int, ::Int) = error("the snapshot was read")

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
        kmax = alg isa Union{PODSVD, PODEVD} ? 2 : (alg isa ComplexSVD ? 1 : 2)
        b = ReducedBasis(alg, Ssmall, NamedTuple(), ps; k = kmax)
        @test b.k == kmax
        @test size(b.Φ, 2) == kmax
        @test size(Matrix(b), 2) == (alg isa Union{PODSVD, PODEVD} ? kmax : 2kmax)
        @test_throws ArgumentError ReducedBasis(alg, Ssmall, NamedTuple(), ps; k = kmax + 1)
        @test_throws ArgumentError ReducedBasis(alg, Ssmall, NamedTuple(), ps; k = 0)
        @test_throws ArgumentError ReducedBasis(
            alg, Ssmall, NamedTuple(), ps; tolerance = 1)
        @test_throws ArgumentError ReducedBasis(
            alg, Ssmall, NamedTuple(), ps; tolerance = -0.1)
        @test_throws ArgumentError ReducedBasis(alg, Ssmall, NamedTuple(), ps)
        @test_throws ArgumentError ReducedBasis(
            alg, Ssmall, NamedTuple(), ps; k = 1, tolerance = 0.5)
        b0 = ReducedBasis(alg, Ssmall, NamedTuple(), ps; tolerance = 0.0)
        @test 1 ≤ b0.k ≤ kmax
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
    for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        for Sempty in (zeros(0, 4), zeros(4, 0))
            @test_throws ArgumentError ReducedBasis(alg, Sempty, NamedTuple(), ps; k = 1)
            @test_throws ArgumentError ReducedBasis(
                alg, Sempty, NamedTuple(), ps; tolerance = 0.5)
        end
    end

    # A PSD algorithm reduces the stacked `[q; p]`, which has an even number of rows.
    for alg in (CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        @test_throws ArgumentError ReducedBasis(alg, rand(41, 6), NamedTuple(), ps; k = 1)
        @test_throws ArgumentError ReducedBasis(alg, zeros(41, 0), NamedTuple(), ps; k = 1)
    end

    # Every route takes the same element types, `Float32` and `Float64`: a complex snapshot made
    # the Gram route return a wrong basis, and an integer one passed the EVD routes and failed in
    # LAPACK on the SVD routes.
    for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        for Sbad in (rand(ComplexF64, 8, 4), rand(-3:3, 8, 4), rand(Float16, 8, 4))
            @test_throws MethodError ReducedBasis(alg, Sbad, NamedTuple(), ps; k = 1)
        end
    end

    # A non-finite entry is refused for every tag, at a fixed `k` and under the energy rule, which
    # without the guard kept the whole admissible basis and a `NaN` spectrum.
    for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD()),
        T in (Float32, Float64)

        for bad in (T(NaN), T(Inf), T(-Inf))
            Sbad = rand(T, 8, 4)
            Sbad[3, 2] = bad
            @test_throws ArgumentError ReducedBasis(alg, Sbad, NamedTuple(), ps; k = 1)
            @test_throws ArgumentError ReducedBasis(
                alg, Sbad, NamedTuple(), ps; tolerance = 1e-3)
        end
    end

    # Every keyword guard runs before the snapshot is read: a snapshot whose entries throw on
    # access reaches the `ArgumentError`, and the decomposition, which would read it, never starts.
    for alg in (PODSVD(), PODEVD(), CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
        S8 = Unreadable(8, 4)
        @test_throws ArgumentError ReducedBasis(alg, S8, NamedTuple(), ps; k = 100)
        @test_throws ArgumentError ReducedBasis(alg, S8, NamedTuple(), ps; tolerance = 1)
        @test_throws ArgumentError ReducedBasis(alg, S8, NamedTuple(), ps)
        @test_throws ArgumentError ReducedBasis(
            alg, S8, NamedTuple(), ps; k = 1, tolerance = 0.5)
        @test_throws ErrorException ReducedBasis(alg, S8, NamedTuple(), ps; k = 1)
    end

    # The basis size is the number of columns of `Φ`, whoever builds the basis.
    @test ReducedBasis(PODSVD(), NamedTuple(), ps, rand(3), rand(3, 2), nothing).k == 2
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
