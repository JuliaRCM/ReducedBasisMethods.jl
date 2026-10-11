"""
    _sorted_eigen(G)

The eigen-decomposition of the symmetric matrix `G`, with the eigenvalues sorted in descending
order and the eigenvectors permuted to match. `eigen` returns them ascending, and both the basis
size and the leading columns are counted from the largest, so the order is reversed once here.

The eigenvectors are sorted by their value, never by its magnitude: a Gram matrix of a
rank-deficient snapshot has round-off eigenvalues of both signs, and `|λ|` puts those among the
leading ones.
"""
function _sorted_eigen(G::AbstractMatrix)
    F = eigen(Symmetric(G))
    p = reverse(eachindex(F.values))
    return F.values[p], F.vectors[:, p]
end

"""
    _qr_basis(A)

The compressions of the leading columns of `A`: the first `size(A, 2)` columns of the thin
Householder `Q` of `qr(A)`.

Householder QR keeps the nested spans, so the first `j` columns of the result span `A[:, 1:j]` for
every `j`, and each leading projector is the projector of `A` itself. This is what replaces
`A = S·Ω·Λ^(-1/2)`, which divides by `√|λ|` of the round-off eigenvalues of a rank-deficient
snapshot and is off by `O(1)` there.
"""
function _qr_basis(A::AbstractMatrix)
    return Matrix(qr(A).Q)[:, 1:size(A, 2)]
end

"""
    _reduce(::PODEVD, S, parameters, paramspace, k, tolerance)

The POD basis of the `N × m` snapshot `S` from the eigendecomposition of the `m × m` Gram matrix
`SᵀS`. The spectrum `Λ` is the Gram spectrum, which squared entries make `σ²`, and it is cut to
`min(N, m)` entries — past the rank of `S` its remaining eigenvalues are round-off. `Φ` is the
orthonormal basis of `S·Ω[:, 1:k]`.
"""
function _reduce(alg::PODEVD, S::AbstractMatrix, parameters::NamedTuple,
        paramspace::ParameterSpace, k, tolerance)
    Λ, Ω = _sorted_eigen(S' * S)
    Λ = Λ[1:min(size(S, 1), size(S, 2))]
    k = _basis_size(Λ, k, tolerance)
    return ReducedBasis(
        alg, parameters, paramspace, Λ, k, _qr_basis(S * Ω[:, 1:k]), nothing)
end

"""
    _reduce(::CotangentLiftEVD, S, parameters, paramspace, k, tolerance)

The cotangent-lift basis of the stacked `2N × m` snapshot `S = [Q; P]`, from the eigendecomposition
of the `2m × 2m` Gram matrix `[Q P]ᵀ[Q P]`, assembled from the views `Q` and `P` without forming
`[Q P]`. Its spectrum is cut to `min(N, 2m)` entries, and `Φ` is the orthonormal basis of
`Q·Ω[1:m, 1:k] + P·Ω[m+1:2m, 1:k]`.
"""
function _reduce(alg::CotangentLiftEVD, S::AbstractMatrix, parameters::NamedTuple,
        paramspace::ParameterSpace, k, tolerance)
    N = size(S, 1) ÷ 2
    m = size(S, 2)
    Q = @view S[1:N, :]
    P = @view S[(N + 1):(2N), :]

    QQ = Q' * Q
    PP = P' * P
    QP = Q' * P
    Λ, Ω = _sorted_eigen([QQ QP; QP' PP])
    Λ = Λ[1:min(N, 2m)]

    k = _basis_size(Λ, k, tolerance)
    Φ = _qr_basis(Q * Ω[1:m, 1:k] + P * Ω[(m + 1):(2m), 1:k])
    return ReducedBasis(alg, parameters, paramspace, Λ, k, Φ, nothing)
end
