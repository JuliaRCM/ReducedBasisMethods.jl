module ReducedBasisMethodsCairoMakieExt

using CairoMakie
using ReducedBasisMethods

function ReducedBasisMethods.spectrumplot(σ::AbstractVector{<:Real};
        title = "", resolution = nothing)
    isempty(σ) && throw(ArgumentError("no spectrum to plot"))
    all(isfinite, σ) || throw(ArgumentError(
        "the spectrum holds a NaN or an Inf, which a log-scaled axis cannot draw"))
    any(>(0), σ) || throw(ArgumentError(
        "the spectrum holds no positive value, so a log-scaled axis has no scale to draw it on"))

    # The spectrum is drawn in the precision the plot stores, which is `Float64`: a `Float32`
    # spectrum is widened to it, and that widening is exact.
    #
    # A rank-deficient snapshot set gives a Gram matrix with exact zeros among its eigenvalues, and
    # a nearly dependent one gives round-off eigenvalues slightly below zero. `max` lifts every
    # value at or below the floor onto it, which is what puts the whole spectrum on the axis
    # instead of dropping the round-off tail off it.
    y = max.(convert(Vector{Float64}, σ), eps(Float64) * Float64(maximum(σ)))

    fig = Figure(size = resolution === nothing ? (620, 380) : resolution)
    ax = Axis(fig[1, 1]; xlabel = "i", ylabel = "σᵢ", yscale = log10)
    isempty(title) || Label(fig[0, 1], title; halign = :left, fontsize = 16)
    plt = lines!(ax, 1:length(y), y)
    return CairoMakie.Makie.FigureAxisPlot(fig, ax, plt)
end

end
