using CairoMakie
using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedComplexityModeling: Parameter, ParameterSpace
using Test

"`spectrumplot` with `DISPLAY` unset: the render needs no display, so the test states that instead
of inheriting whatever the runner happens to set."
render(σ) = withenv("DISPLAY" => nothing) do
    spectrumplot(σ)
end

"The PNG bytes that `save` writes for a figure, also with `DISPLAY` unset."
function save_png(fap)
    return withenv("DISPLAY" => nothing) do
        path = joinpath(mktempdir(), "spectrum.png")
        save(path, fap)
        return read(path)
    end
end

"`FigureAxisPlot` keeps its series in its `plot` field, and `Lines` keeps its converted positional
argument as one vector of points: the element type of those points is the precision the plot holds,
and that is where the comparison of check 2 is made."
plotted_y(fap) = [q[2] for q in fap.plot.positions[]]
plotted_eltype(fap) = eltype(eltype(fap.plot.positions[]))

@testset "Diagnostics" begin
    @testset "the plot is a `FigureAxisPlot`" begin
        @test render([1.0, 0.1]) isa CairoMakie.Makie.FigureAxisPlot
    end

    @testset "a full-rank spectrum renders headless, exactly as given ($T)" for T in (Float64, Float32)
        # check 2: σᵢ = 10^(-i) for i = 1:10. `save` writes the PNG through Cairo, so no display
        # is involved and none can be missing.
        σ = T.(10.0 .^ (-(1:10)))

        fap = render(σ)
        # the values are drawn as given, never squared and never rooted: read them back in the
        # precision the plot stores, which is where the comparison of `==` is made.
        @test plotted_eltype(fap) == Float64
        @test plotted_y(fap) == convert(Vector{plotted_eltype(fap)}, σ)
        @test plotted_y(fap) == σ
        # the floor is `eps(Float64) * maximum(σ)`, and the smallest of these values sits far
        # above it, so the floor lifts none of them and the comparison is an exact one
        @test all(>(eps(Float64) * Float64(maximum(σ))), σ)

        @test save_png(fap)[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]
    end

    @testset "a one-element spectrum renders" begin
        @test plotted_y(render(Float64[3.0])) == [3.0]
    end

    @testset "an empty, a NaN, an Infinity and a non-positive spectrum are refused" begin
        @test_throws ArgumentError spectrumplot(Float64[])
        @test_throws ArgumentError spectrumplot([1.0, NaN])
        @test_throws ArgumentError spectrumplot([1.0, Inf])
        @test_throws ArgumentError spectrumplot([1.0, -Inf])
        # no positive value at all, so a log-scaled axis has no scale to draw the spectrum on
        @test_throws ArgumentError spectrumplot(zeros(3))
        @test_throws ArgumentError spectrumplot(Float32[0, 0])
        @test_throws ArgumentError spectrumplot([-1.0])
        @test_throws ArgumentError spectrumplot([0.0, -1e-17])
        # the zero snapshot set, whose eigenvalues the Gram route stores as exact zeros
        paramspace = ParameterSpace(Parameter(:σ, 0.0, 1.0, 2))
        Λ = ReducedBasis(PODEVD(), zeros(12, 4), NamedTuple(), paramspace; k = 1).Λ
        @test all(iszero, Λ)
        @test_throws ArgumentError spectrumplot(Λ)
    end

    @testset "a rank-deficient spectrum draws its zeros at the floor ($T)" for T in (Float64, Float32)
        # check 3: the EVD route takes the eigenvalues of the Gram matrix XᵀX. A snapshot column
        # that is exactly zero leaves exact zeros among them, and round-off leaves the neighbours
        # of a dependent column slightly below zero. Both are lifted onto the floor
        # `eps(Float64) * maximum(σ)`, so a log-scaled axis draws the whole spectrum and no value
        # is silently dropped.
        Random.seed!(11)
        X0 = hcat(randn(T, 9, 3), zeros(T, 9, 2))     # two dead snapshot columns
        Λ = reverse(eigen(Symmetric(X0' * X0)).values)

        # the precondition of the clause, asserted rather than assumed
        @test count(iszero, Λ) == 2

        floor = eps(Float64) * Float64(maximum(Λ))
        fap = render(Λ)
        drawn = plotted_y(fap)
        @test plotted_eltype(fap) == Float64
        @test drawn == max.(Λ, floor)
        @test issorted(drawn; rev = true)
        @test all(>(0), drawn)
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

        # the floor lifts a positive round-off value along with the non-positive ones, and the
        # drawn series stays in the order the spectrum is stored in
        floor = eps(Float64) * Float64(maximum(Λ))
        @test any(x -> 0 < x < floor, Λ)

        drawn = plotted_y(render(Λ))
        @test drawn == max.(Λ, floor)
        @test issorted(drawn; rev = true)
        @test all(>(0), drawn)
    end

    @testset "the same round-off in a Float32 Gram spectrum" begin
        # the floor is taken in the precision the plot stores, so a `Float32` spectrum is drawn in
        # `Float64` and its non-positive eigenvalues are lifted onto that floor as well
        Random.seed!(7)
        X = randn(Float32, 12, 4)
        Λ = reverse(eigen(Symmetric(hcat(X, X, X)' * hcat(X, X, X))).values)
        @test any(<(0), Λ)
        @test any(>(0), Λ)

        drawn = plotted_y(render(Λ))
        @test drawn == max.(Float64.(Λ), eps(Float64) * maximum(Λ))
        @test issorted(drawn; rev = true)
        @test all(>(0), drawn)
    end

    @testset "a spectrum finer than the plot stores is drawn in `Float64`" begin
        # the plot stores `Float64`, and a spectrum finer than that is drawn in it: it is that
        # widening which makes the comparisons above plain `Vector{Float64}` comparisons
        fap = render(big.([1.0, 0.5, 0.25]))
        @test plotted_eltype(fap) == Float64
        @test plotted_y(fap) == [1.0, 0.5, 0.25]
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

        # the same data through the EVD route: the floor lifts its round-off tail, which holds a
        # positive value below the floor as well
        floor = eps(Float64) * Float64(maximum(Λ))
        @test any(x -> 0 < x < floor, Λ)

        drawn = plotted_y(render(Λ))
        @test drawn == max.(Λ, floor)
        @test issorted(drawn; rev = true)
    end
end
