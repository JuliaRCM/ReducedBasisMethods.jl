using LinearAlgebra
using Random
using ReducedComplexityModeling: Parameter, ParameterSpace

"""
Snapshot matrices with a prescribed singular-value spectrum, and the dense symplectic form the
projection identities are stated against.

Every builder draws from the default Xoshiro stream, so each caller fixes a seed first.
"""

"`r` orthonormal columns of length `N`, from the current random stream."
function orthonormal_columns(::Type{T}, N::Int, r::Int) where {T}
    return Matrix(qr(randn(T, N, r)).Q)[:, 1:r]
end

"An `N × m` real matrix `U * Diagonal(σ) * Vᵀ` with orthonormal factors, whose singular
values are `σ` up to floating-point round-off."
function low_rank_matrix(::Type{T}, N::Int, m::Int, σ::AbstractVector) where {T <: Real}
    r = length(σ)
    return orthonormal_columns(T, N, r) * Diagonal(T.(σ)) * orthonormal_columns(T, m, r)'
end

"An `N × m` complex matrix with the singular values `σ`."
function complex_low_rank_matrix(::Type{T}, N::Int, m::Int,
        σ::AbstractVector) where {T <: Real}
    r = length(σ)
    return orthonormal_columns(Complex{T}, N, r) * Diagonal(T.(σ)) *
           orthonormal_columns(Complex{T}, m, r)'
end

"""
A `2N × m` PSD snapshot `[Q; P]` whose cotangent-lift matrix `[Q P]` is `M`, so that
`Q = M[:, 1:m]` and `P = M[:, m+1:2m]`.
"""
function cotangent_snapshot(M::AbstractMatrix)
    _, m2 = size(M)
    m = m2 ÷ 2
    return [M[:, 1:m]; M[:, (m + 1):m2]]
end

"A `2N × m` PSD snapshot `[Q; P]` with `Q + iP = C`."
function complex_snapshot(C::AbstractMatrix)
    return [real(C); imag(C)]
end

"The dense symplectic form `[0 I; -I 0]`, `2n × 2n`."
function symplectic_form(::Type{T}, n::Int) where {T}
    J = zeros(T, 2n, 2n)
    for i in 1:n
        J[i, n + i] = one(T)
        J[n + i, i] = -one(T)
    end
    return J
end

"A one-parameter space, for a basis whose parameters no test reads."
testparamspace() = ParameterSpace(Parameter(:σ, 0.0, 1.0, 2))
