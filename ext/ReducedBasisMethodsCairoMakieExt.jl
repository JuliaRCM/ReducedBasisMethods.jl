module ReducedBasisMethodsCairoMakieExt

using CairoMakie
using ReducedBasisMethods

function ReducedBasisMethods.spectrumplot(σ::AbstractVector{<:Real};
        title = "", resolution = nothing)
    isempty(σ) && throw(ArgumentError("no spectrum to plot"))
    all(isfinite, σ) || throw(ArgumentError(
        "the spectrum holds a NaN or an Inf, which a log-scaled axis cannot draw"))

    # The spectrum is drawn in its own precision, at its own floor: a `Float32` spectrum is not
    # widened to `Float64` just to be drawn.
    s = convert(Vector{float(eltype(σ))}, σ)
    # A rank-deficient snapshot set gives a Gram matrix with exact zeros among its eigenvalues, and
    # a nearly dependent one gives round-off eigenvalues slightly below zero. Both are lifted to
    # the floor rather than drawn below it, and every positive value is drawn as it is given.
    floor = eps(eltype(s)) * maximum(s)
    y = [σᵢ ≤ 0 ? floor : σᵢ for σᵢ in s]

    fig = Figure(size = resolution === nothing ? (620, 380) : resolution)
    ax = Axis(fig[1, 1]; xlabel = "i", ylabel = "σᵢ", yscale = log10)
    isempty(title) || Label(fig[0, 1], title; halign = :left, fontsize = 16)
    lines!(ax, 1:length(y), y)
    return fig
end

end
