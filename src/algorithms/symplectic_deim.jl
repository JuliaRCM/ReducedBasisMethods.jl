"""
    SymplecticDEIM(A, L, U, stencil, h, ∇h)

Structure-preserving DEIM for a Hamiltonian `H(x) = ½xᵀLx + Σᵢ hᵢ(x)`, after Peng and Mohseni,
[doi:10.1137/140978922](https://doi.org/10.1137/140978922).

DEIM is applied to the densities `hᵢ`, not to the gradient `∇H`, and the reduced gradient is
then obtained by differentiating the reduced Hamiltonian exactly. The reduced gradient is
therefore the gradient of a scalar, and its Jacobian is symmetric by construction. The plain
DEIM reduction of eq. (5.8) of the same paper, `W = AᵀΨ(ΠᵀΨ)⁻¹`, gives neither: that `W` is
not symmetric, and the authors state that the reduced system is "not necessarily Hamiltonian".

# Arguments
- `A`: an `N × r` basis. It need not be symplectic: `H̃` is defined for any `A`.
- `L`: the `N × N` matrix of the quadratic part. Only its symmetric part enters `H̃`, which is
  exact, because `½xᵀLx = ½xᵀ·sym(L)·x`.
- `U`: an `M × m` orthonormal basis of density snapshots, where `M` is the number of densities.
- `stencil(i)`: the indices of `x` that the density `hᵢ` reads.
- `h(xs, i)`: the scalar `hᵢ` at the values `xs` of its stencil.
- `∇h(xs, i)`: the gradient of `hᵢ` with respect to `xs`.

The element type is `T = float(promote_type(eltype(A), eltype(L), eltype(U)))`, and the stored
data is in `T`. `sd(z)` and [`hamiltonian_gradient!`](@ref) are generic over `eltype(z)`, so
`ForwardDiff` dual numbers pass through them; inputs mixed between two element types are not
supported.

Throws an `ArgumentError` where `m > M`, and for a column of `U` whose largest DEIM residual
magnitude is not finite and positive; see [`_deim_indices`](@ref).
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

    # The weights make the interpolation of the densities exact on the DEIM rows:
    # 𝟙ᵀU(ΠᵀU)⁻¹Πᵀh = wᵀh[p].
    p = _deim_indices(U)
    w = convert(Vector{T}, (U[p, :])' \ (U' * ones(T, size(U, 1))))

    # `L̃ = Aᵀ·sym(L)·A`, symmetrised once more after the product, so that the Jacobian of the
    # gradient is symmetric to the last bit rather than to round-off of the product.
    Ls = (L + transpose(L)) / 2
    L̃ = Matrix{T}(A' * Ls * A)
    L̃ .= (L̃ + transpose(L̃)) / 2

    return SymplecticDEIM{T, typeof(A), typeof(stencil), typeof(h), typeof(∇h)}(A, L̃, p, w,
        stencil, h, ∇h)
end

"""
    (sd::SymplecticDEIM)(z)

The reduced Hamiltonian `H̃(z) = ½zᵀL̃z + Σⱼ wⱼ h_{pⱼ}(A_{Sⱼ}z)`, where `Sⱼ = stencil(pⱼ)` and
`A_{Sⱼ} = A[Sⱼ, :]`. Its gradient is [`hamiltonian_gradient!`](@ref), and `H̃` is an exact
invariant of `ż = J₂ₖ∇H̃(z)`.
"""
function (sd::SymplecticDEIM)(z::AbstractVector)
    H = transpose(z) * (sd.L̃ * z) / 2
    for j in eachindex(sd.p)
        i = sd.p[j]
        H += sd.w[j] * sd.h(sd.A[sd.stencil(i), :] * z, i)
    end
    return H
end

"""
    hamiltonian_gradient!(g, sd::SymplecticDEIM, z)

Writes `∇H̃(z) = L̃z + Σⱼ wⱼ A_{Sⱼ}ᵀ ∇h_{pⱼ}(A_{Sⱼ}z)` into `g`, after Peng and Mohseni,
[doi:10.1137/140978922](https://doi.org/10.1137/140978922). Its Jacobian,
`L̃ + Σⱼ wⱼ A_{Sⱼ}ᵀ ∇²h_{pⱼ}(A_{Sⱼ}z) A_{Sⱼ}`, is symmetric by construction, so `H̃` of
[`SymplecticDEIM`](@ref) is an exact invariant of `ż = J₂ₖ∇H̃(z)`.

The name is `hamiltonian_gradient!` rather than `gradient!` because `gradient!` clashes with
`ForwardDiff` in a test environment.
"""
function hamiltonian_gradient!(g::AbstractVector, sd::SymplecticDEIM, z::AbstractVector)
    g .= sd.L̃ * z
    for j in eachindex(sd.p)
        i = sd.p[j]
        AS = sd.A[sd.stencil(i), :]
        g .+= sd.w[j] .* (transpose(AS) * sd.∇h(AS * z, i))
    end
    return g
end
