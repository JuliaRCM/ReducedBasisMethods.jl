"""
    spectrumplot(σ::AbstractVector{<:Real}; title = "", ylabel = "", size = (620, 380))

Draw the spectrum `σ` of a reduced basis against its index on a log-scaled axis, in `Float64`.
The values are drawn as given: singular values, or the squared singular values (Gram eigenvalues)
that a `ReducedBasis` stores in `Λ`. `ylabel` names which, and `size` is the figure size.

Every value below `eps(Float64) * maximum(σ)` is drawn at that floor, so the exact zeros and the
round-off negatives of a Gram spectrum stay on the axis. A spectrum that holds a `NaN` or an `Inf`,
or no positive value, throws an `ArgumentError`.

Returns a `Makie.FigureAxisPlot`. Defined in a package extension that loads with `Makie`; `save`
needs a backend such as `CairoMakie`.

```julia
using ReducedBasisMethods, CairoMakie
save("spectrum.png", spectrumplot(b.Λ; ylabel = "λᵢ"))
```
"""
function spectrumplot end
