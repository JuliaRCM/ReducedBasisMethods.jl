using CairoMakie
using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedComplexityModeling: Parameter, ParameterSpace
using Test

"""
The single ``Lines`` series of a figure's axis. `CairoMakie` is loaded here, and only here, so this
file tests the extension and not the package's own load path.
"""
function series(fig)
    only(filter(p -> p isa CairoMakie.Makie.Lines,
        only(filter(a -> a isa CairoMakie.Makie.Axis, fig.content)).scene.plots))
end

"`Lines` stores its converted positional argument as one vector of points, so the element type of
those points is the precision the plot holds, and the comparison of check 2 is made in it."
plotted_y(fig) = [q[2] for q in series(fig).positions[]]
plotted_eltype(fig) = eltype(eltype(series(fig).positions[]))

@testset "Diagnostics" begin
    @testset "a full-rank spectrum renders headless, exactly as given ($T)" for T in (Float64, Float32)
        # check 2: σᵢ = 10^(-i) for i = 1:10. `save` writes the PNG through Cairo, so no display
        # is involved and none can be missing.
        σ = T.(10.0 .^ (-(1:10)))

        fig = spectrumplot(σ)
        # the values are drawn as given, never squared and never rooted: read them back in the
        # precision the plot stores, which is where the comparison of `==` is made.
        @test plotted_eltype(fig) == Float64
        @test plotted_y(fig) == convert(Vector{plotted_eltype(fig)}, σ)
        @test plotted_y(fig) == σ

        mktempdir() do dir
            path = joinpath(dir, "spectrum.png")
            save(path, fig)
            bytes = read(path)
            @test bytes[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]
        end
    end

    @testset "a one-element spectrum renders" begin
        @test plotted_y(spectrumplot(Float64[3.0])) == [3.0]
    end

    @testset "an empty, a NaN and an Infinity spectrum are refused" begin
        @test_throws ArgumentError spectrumplot(Float64[])
        @test_throws ArgumentError spectrumplot([1.0, NaN])
        @test_throws ArgumentError spectrumplot([1.0, Inf])
        @test_throws ArgumentError spectrumplot([1.0, -Inf])
    end

    @testset "a rank-deficient spectrum draws its zeros at the floor ($T)" for T in (Float64, Float32)
        # check 3: the EVD route takes the eigenvalues of the Gram matrix XᵀX. A snapshot column
        # that is exactly zero leaves exact zeros among them, and round-off leaves the neighbours
        # of a dependent column slightly below zero. Both are lifted to `eps(T) * maximum(σ)`, so
        # a log-scaled axis draws the whole spectrum and no value is silently dropped.
        Random.seed!(11)
        X0 = hcat(randn(T, 9, 3), zeros(T, 9, 2))     # two dead snapshot columns
        Λ = reverse(eigen(Symmetric(X0' * X0)).values)

        # the precondition of the clause, asserted rather than assumed
        @test count(iszero, Λ) == 2
        floor = eps(T) * maximum(Λ)
        # every positive entry is far above the floor, so the two descriptions of the drawn series
        # agree: `max` and the clamping of the non-positive entries alone
        @test all(>(floor), Λ[Λ .> 0])

        fig = spectrumplot(Λ)
        @test plotted_eltype(fig) == Float64
        @test plotted_y(fig) == convert(Vector{plotted_eltype(fig)}, max.(Λ, floor))
        @test all(>(0), plotted_y(fig))
    end

    @testset "a slightly negative eigenvalue is drawn at the floor" begin
        # the other face of the same round-off: the Gram route keeps the negative eigenvalues of a
        # nearly dependent snapshot, and they must reach the axis rather than fall off it, while the
        # positive entries are drawn exactly as they are given
        Random.seed!(7)
        X = randn(Float64, 12, 4)
        Λ = reverse(eigen(Symmetric(hcat(X, X, X)' * hcat(X, X, X))).values)
        @test any(<(0), Λ)
        @test any(>(0), Λ)

        floor = eps(Float64) * maximum(Λ)
        drawn = plotted_y(spectrumplot(Λ))
        @test drawn[Λ .> 0] == Λ[Λ .> 0]
        @test drawn[Λ .≤ 0] == fill(floor, count(≤(0), Λ))
        @test all(>(0), drawn)
    end

    @testset "the spectrum of a rank-deficient snapshot, as the Gram route stores it" begin
        Random.seed!(7)
        paramspace = ParameterSpace(Parameter(:σ, 0.0, 1.0, 2))
        X = randn(Float64, 12, 4)
        b = ReducedBasis(PODEVD(), hcat(X, X, X), NamedTuple(), paramspace;
            k = 12)

        # the route stores the raw spectrum with its non-positive entries kept
        Λ = b.Λ
        @test any(≤(0), Λ)
        @test issorted(Λ; rev = true)

        floor = eps(Float64) * maximum(Λ)
        drawn = plotted_y(spectrumplot(Λ))
        @test drawn[Λ .> 0] == Λ[Λ .> 0]
        @test drawn[Λ .≤ 0] == fill(floor, count(≤(0), Λ))
    end
end
