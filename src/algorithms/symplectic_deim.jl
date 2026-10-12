"""
    SymplecticDEIM(A, L, U, stencil, h, ∇h)

Structure-preserving DEIM for a Hamiltonian `H(x) = ½xᵀLx + Σᵢ hᵢ(x)`, after Peng and Mohseni,
[doi:10.1137/140978922](https://doi.org/10.1137/140978922).

DEIM is applied to the densities `hᵢ`, not to the gradient `∇H`, and the reduced gradient is
then obtained by differentiating the reduced Hamiltonian exactly. The reduced gradient is
therefore the gradient of a scalar, and its Jacobian is symmetric by construction. In contrast,
the plain DEIM reduction of eq. (5.8) of the same paper uses `W = AᵀΨ(ΠᵀΨ)⁻¹` and does not
guarantee a symmetric reduced-gradient Jacobian or a Hamiltonian reduced system.

# Arguments
- `A`: an `N × r` basis. It need not be symplectic: `H̃` is defined for any `A`.
- `L`: the `N × N` matrix of the quadratic part. Only its symmetric part enters `H̃`, which is
  exact, because `½xᵀLx = ½xᵀ·sym(L)·x`.
- `U`: an `M × m` orthonormal basis of density snapshots, where `M` is the number of densities.
- `stencil(i)`: the indices of `x` that the density `hᵢ` reads, as any 1-based
  `AbstractVector{<:Integer}`.
- `h(xs, i)`: the scalar `hᵢ` at the values `xs` of its stencil. `xs` is a lazy view of
  `A[stencil(i), :] * z`, which allocates nothing.
- `∇h(xs, i)`: the gradient of `hᵢ` with respect to `xs`, as any indexable collection — a
  `Vector` or a `Tuple`.

The element type is `T = float(promote_type(eltype(A), eltype(L), eltype(U)))`. `L̃` and the
weights are stored in `T`, and `A` is stored as passed. The call of the returned object and
[`hamiltonian_gradient!`](@ref) are generic over `eltype(z)`, with no typed internal buffer, so
`ForwardDiff` dual numbers pass through them; inputs mixed between two element types are not
supported.

Throws an `ArgumentError` where `m > M`, and for a column of the density basis `U` whose largest
DEIM residual magnitude `ρ` is not both positive and finite: `ρ == 0` where the column is zero
or exactly dependent on the ones before it, and `ρ` is `NaN` or `Inf` where the column holds a
`NaN` or an `Inf`. [`_deim_indices`](@ref) is the check.
"""
struct SymplecticDEIM{T, AT <: AbstractMatrix, ST, HT, GT}
    A::AT
    L̃::Matrix{T}
    p::Vector{Int}
    w::Vector{T}
    stencil::ST
    h::HT
    ∇h::GT
end

function SymplecticDEIM(
        A::AbstractMatrix, L::AbstractMatrix, U::AbstractMatrix, stencil, h, ∇h)
    T = float(promote_type(eltype(A), eltype(L), eltype(U)))

    # The weights express the sum of the DEIM-interpolated densities using sampled values:
    # 𝟙ᵀU(ΠᵀU)⁻¹Πᵀh = wᵀh[p].
    p = _deim_indices(U)
    w = convert(Vector{T}, (U[p, :])' \ (U' * ones(T, size(U, 1))))

    # `L̃ = Aᵀ·sym(L)·A`, symmetrised once more after the product, so that the Jacobian of the
    # gradient is symmetric to the last bit rather than to the round-off of the product.
    Ls = (L + transpose(L)) / 2
    L̃ = A' * Ls * A
    L̃ .= (L̃ + transpose(L̃)) / 2

    return SymplecticDEIM{T, typeof(A), typeof(stencil), typeof(h), typeof(∇h)}(A, L̃, p, w,
        stencil, h, ∇h)
end

# `_StencilVector(A, S, z)` is the lazy vector `v` with `v[k] = Σ_c A[S[k], c] * z[c]`, which the
# density callbacks receive in place of a materialised `A[S, :] * z`. Nothing is allocated for
# it, and an entry is computed when a callback reads it.
struct _StencilVector{T, AT <: AbstractMatrix, ST <: AbstractVector{<:Integer},
    ZT <: AbstractVector} <: AbstractVector{T}
    A::AT
    S::ST
    z::ZT
end

# The lazy view `v[k] = Σ_c A[S[k], c] * z[c]`, with element type `promote_type(T, eltype(z))`.
@inline function _stencil_vector(A::AT, S::ST, z::ZT, ::Type{T}) where {AT, ST, ZT, T}
    return _StencilVector{promote_type(T, eltype(ZT)), AT, ST, ZT}(A, S, z)
end

Base.size(v::_StencilVector) = (length(v.S),)

@inline function Base.getindex(v::_StencilVector{T}, k::Int) where {T}
    s = zero(T)
    for c in axes(v.A, 2)
        s += v.A[v.S[k], c] * v.z[c]
    end
    return s
end

"""
    (sd::SymplecticDEIM)(z)

The reduced Hamiltonian `H̃(z) = ½zᵀL̃z + Σⱼ wⱼ h_{pⱼ}(A_{Sⱼ}z)`, where `Sⱼ = stencil(pⱼ)` and
`A_{Sⱼ} = A[Sⱼ, :]`. Its gradient is [`hamiltonian_gradient!`](@ref), and `H̃` is an exact
invariant of `ż = J₂ₖ∇H̃(z)`, after Peng and Mohseni,
[doi:10.1137/140978922](https://doi.org/10.1137/140978922).

The call allocates nothing of its own: `h` receives a lazy view of `A_{Sⱼ}z`, so where `stencil`
and `h` allocate nothing the call reads 0 B, and anything a callback allocates is the caller's.
A `NaN` or an `Inf` in `z`, or from `h`, enters `H̃` with no check.
"""
function (sd::SymplecticDEIM{T})(z::AbstractVector) where {T}
    H = dot(z, sd.L̃, z) / 2
    for j in eachindex(sd.p)
        i = sd.p[j]
        H += sd.w[j] * sd.h(_stencil_vector(sd.A, sd.stencil(i), z, T), i)
    end
    return H
end

"""
    hamiltonian_gradient!(g, sd::SymplecticDEIM, z)

Writes `∇H̃(z) = L̃z + Σⱼ wⱼ A_{Sⱼ}ᵀ ∇h_{pⱼ}(A_{Sⱼ}z)` into `g`, after Peng and Mohseni,
[doi:10.1137/140978922](https://doi.org/10.1137/140978922). Its Jacobian,
`L̃ + Σⱼ wⱼ A_{Sⱼ}ᵀ ∇²h_{pⱼ}(A_{Sⱼ}z) A_{Sⱼ}`, is symmetric by construction, so `H̃` of
[`SymplecticDEIM`](@ref) is an exact invariant of `ż = J₂ₖ∇H̃(z)`.

The write allocates nothing of its own: `∇h` receives a lazy view of `A_{Sⱼ}z` and may return
any indexable collection, a `Vector` or a `Tuple`, so where `stencil` and `∇h` allocate nothing
the write reads 0 B, and anything a callback allocates is the caller's.

The write gives no guarantee for an aliased array: `g` and `z` must be distinct, because the
quadratic part is written with `mul!`, which requires its output and its input to differ.

The constructor validates the density basis through [`_deim_indices`](@ref): `m > M`, and a
column whose largest DEIM residual magnitude is zero or non-finite, raise `ArgumentError`.
This write propagates errors from array operations and callbacks. A `NaN` or an `Inf` in `z`,
or from `∇h`, enters `g` with no check.
"""
function hamiltonian_gradient!(
        g::AbstractVector, sd::SymplecticDEIM{T}, z::AbstractVector) where {T}
    mul!(g, sd.L̃, z)
    for j in eachindex(sd.p)
        i = sd.p[j]
        S = sd.stencil(i)
        v = sd.∇h(_stencil_vector(sd.A, S, z, T), i)
        for k in eachindex(S), c in axes(sd.A, 2)

            g[c] += sd.w[j] * sd.A[S[k], c] * v[k]
        end
    end
    return g
end
