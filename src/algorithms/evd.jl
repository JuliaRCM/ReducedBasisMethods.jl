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

The orthonormal basis of the columns of `A`: the thin Householder `Q` of `qr(A)`.

Householder QR keeps the nested spans, so the first `j` columns of the result span `A[:, 1:j]` for
every `j`, and each leading projector is the projector of `A` itself. `A = S·Ω·Λ^(-1/2)` is not
used: it divides by `√|λ|` of the round-off eigenvalues of a rank-deficient snapshot and is off by
`O(1)` there.
"""
function _qr_basis(A::AbstractMatrix)
    return Matrix(qr(A).Q)
end

"""
    _reduce(::PODEVD, S, parameters, paramspace, k, tolerance)

The POD basis of the `N × m` snapshot `S` from the eigendecomposition of the `m × m` Gram matrix
`SᵀS`. The spectrum is the whole Gram spectrum, `σ²` with the round-off eigenvalues of both signs
kept, and it is stored as it is: a caller that plots it sees every eigenvalue of `SᵀS`. The basis
is cut to `kmax = min(N, m)` entries, the most a snapshot of `N` rows admits, and `Φ` is the
orthonormal basis of `S·Ω[:, 1:k]`.
"""
function _reduce(alg::PODEVD, S::AbstractMatrix, parameters::NamedTuple,
        paramspace::ParameterSpace, k, tolerance)
    Λ, Ω = _sorted_eigen(S' * S)
    k = _basis_size(Λ, k, tolerance, min(size(S, 1), size(S, 2)))
    return ReducedBasis(
        alg, parameters, paramspace, Λ, k, _qr_basis(S * Ω[:, 1:k]), nothing)
end

"""
    _reduce(::CotangentLiftEVD, S, parameters, paramspace, k, tolerance)

The cotangent-lift basis of the stacked `2N × m` snapshot `S = [Q; P]`, from the eigendecomposition
of the `2m × 2m` Gram matrix `[Q P]ᵀ[Q P]`, assembled from the views `Q` and `P` without forming
`[Q P]`. The spectrum is the whole Gram spectrum, round-off eigenvalues of both signs kept, and it
is stored as it is. The basis is cut to `kmax = min(N, 2m)` entries, and `Φ` is the orthonormal
basis of `Q·Ω[1:m, 1:k] + P·Ω[m+1:2m, 1:k]`.
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

    k = _basis_size(Λ, k, tolerance, min(N, 2m))
    Φ = _qr_basis(Q * Ω[1:m, 1:k] + P * Ω[(m + 1):(2m), 1:k])
    return ReducedBasis(alg, parameters, paramspace, Λ, k, Φ, nothing)
end
