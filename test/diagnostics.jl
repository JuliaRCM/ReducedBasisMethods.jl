using CairoMakie
using ReducedBasisMethods
using ReducedComplexityModeling: Parameter, ParameterSpace
using Test

"The plotted values of a `FigureAxisPlot`, and their element type."
plotted_y(fap) = [q[2] for q in fap.plot.positions[]]
plotted_eltype(fap) = eltype(eltype(fap.plot.positions[]))

@testset "Diagnostics" begin
    @testset "a spectrum is drawn as given and saved headless ($T)" for T in (Float64, Float32)
        σ = T.(10.0 .^ (-(1:10)))
        fap = spectrumplot(σ)
        @test fap isa CairoMakie.Makie.FigureAxisPlot
        @test plotted_eltype(fap) == Float64
        @test plotted_y(fap) == σ

        path = joinpath(mktempdir(), "spectrum.png")
        save(path, fap)
        @test read(path)[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]
    end

    @testset "a one-element and a BigFloat spectrum" begin
        @test plotted_y(spectrumplot([3.0])) == [3.0]
        @test plotted_y(spectrumplot(big.([1.0, 0.5, 0.25]))) == [1.0, 0.5, 0.25]
    end

    @testset "values below the floor are drawn at the floor ($T)" for T in (Float64, Float32)
        # the floor is eps(Float64) * maximum(σ) = eps(Float64)
        σ = T[1, 1e-3, 1e-20, 0, -1e-18]
        @test plotted_y(spectrumplot(σ)) == [1, Float64(T(1e-3)), eps(), eps(), eps()]
    end

    @testset "a spectrum with a NaN, an Inf or no positive value is refused" begin
        @test_throws ArgumentError spectrumplot(Float64[])
        @test_throws ArgumentError spectrumplot([1.0, NaN])
        @test_throws ArgumentError spectrumplot([1.0, Inf])
        @test_throws ArgumentError spectrumplot([1.0, -Inf])
        @test_throws ArgumentError spectrumplot(zeros(3))
        @test_throws ArgumentError spectrumplot(Float32[0, 0])
        @test_throws ArgumentError spectrumplot([0.0, -1e-17])
        # the zero snapshot set, whose eigenvalues the Gram route stores as exact zeros
        paramspace = ParameterSpace(Parameter(:σ, 0.0, 1.0, 2))
        Λ = ReducedBasis(PODEVD(), zeros(12, 4), NamedTuple(), paramspace; k = 1).Λ
        @test all(iszero, Λ)
        @test_throws ArgumentError spectrumplot(Λ)
    end
end
