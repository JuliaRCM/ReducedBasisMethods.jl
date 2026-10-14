"""
    spectrumplot(σ::AbstractVector{<:Real}; kwargs...)

The singular-value spectrum `σ` of a reduced basis, drawn on a log-scaled axis against its index:
the diagnostic that says how many basis vectors a snapshot set admits, and where the spectrum
runs into round-off.

The values are drawn as they are given, never squared and never rooted: an eigenvalue spectrum
from a Gram matrix `MᵀM` and a singular-value spectrum of `M` differ by a square, and reading
one as the other flatters the decay by a factor of two. A caller that holds the eigenvalues
plots them as they are.

The axis is log-scaled, so the spectrum is lifted onto the floor `eps(Float64) * maximum(σ)`
before it is drawn: a rank-deficient snapshot set leaves exact zeros among the eigenvalues of its
Gram matrix, and a nearly dependent one leaves values slightly below zero, and both belong on the
axis rather than below it. The floor is taken in `Float64`, the precision the plot stores, and the
whole spectrum is drawn in that precision, so a `Float32` spectrum is widened by an exact
widening.

An empty `σ` throws an `ArgumentError`, and so does a spectrum holding a `NaN` or an `Inf` and one
holding no positive value at all, which leaves a log-scaled axis no scale to draw on.

Returns a `Makie.FigureAxisPlot`, whose `plot` field holds the plotted series, and
`save(path, spectrumplot(σ))` writes it to a PNG.

Defined in a package extension: load `CairoMakie` to make it available.

```julia
using ReducedBasisMethods, CairoMakie
fap = spectrumplot(b.Λ)          # b.Λ is the full stored spectrum
save("spectrum.png", fap)
```
"""
function spectrumplot end
