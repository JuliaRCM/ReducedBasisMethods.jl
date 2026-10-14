module ReducedBasisMethodsMakieExt

using Makie
using ReducedBasisMethods

function ReducedBasisMethods.spectrumplot(σ::AbstractVector{<:Real};
        title = "", ylabel = "", size = (620, 380))
    all(isfinite, σ) || throw(ArgumentError("the spectrum holds a NaN or an Inf"))
    any(>(0), σ) || throw(ArgumentError("the spectrum holds no positive value"))

    # A log-scaled axis cannot draw the exact zeros of a rank-deficient snapshot set or the
    # round-off negatives of a Gram spectrum, so every value below the floor is lifted onto it.
    y = max.(Float64.(σ), eps(Float64) * Float64(maximum(σ)))

    return lines(y; figure = (; size),
        axis = (; title, titlealign = :left, xlabel = "i", ylabel, yscale = log10))
end

end
