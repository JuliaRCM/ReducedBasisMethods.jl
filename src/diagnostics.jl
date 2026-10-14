"""
    spectrumplot(σ::AbstractVector{<:Real}; kwargs...)

The singular-value spectrum `σ` of a reduced basis, drawn on a log-scaled axis against its index:
the diagnostic that says how many basis vectors a snapshot set admits, and where the spectrum
runs into round-off.

The values are drawn as they are given, never squared and never rooted: an eigenvalue spectrum
from a Gram matrix `MᵀM` and a singular-value spectrum of `M` differ by a square, and reading
one as the other flatters the decay by a factor of two. A caller that holds the eigenvalues
plots them as they are.

The axis is log-scaled, so a value at or below zero — an exact zero of a rank-deficient spectrum,
or the slightly negative eigenvalue the Gram route can return — is lifted to the floor
`eps(T) * maximum(σ)`, taken in the spectrum's own precision, instead of falling off the axis.
Every positive value is drawn as it is given, never rounded and never lifted: a `Float32` value is
widened to the `Float64` the plot stores, which is exact.

An empty `σ` throws an `ArgumentError`, and so does a spectrum holding a `NaN` or an `Inf`,
which a log-scaled axis cannot draw.

Defined in a package extension: load `CairoMakie` to make it available.

```julia
using ReducedBasisMethods, CairoMakie
fig = spectrumplot(b.Λ)          # b.Λ is the full stored spectrum
```
"""
function spectrumplot end
