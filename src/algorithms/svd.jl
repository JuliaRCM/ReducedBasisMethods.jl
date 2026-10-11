"""
    _svd(M)

The thin singular value decomposition of `M` as `(U, σ)`, from LAPACK's `gesvd!` with
`jobu = 'S'` and `jobvt = 'N'`.

The right factor is not formed because none of the reduced bases uses it, and for the snapshot
matrices this package reduces it costs as much memory as the snapshot itself. `M` is copied to
the element type it is asked for, since `gesvd!` overwrites its argument.
"""
function _svd(M::AbstractMatrix{T}) where {T}
    U, σ, _ = LinearAlgebra.LAPACK.gesvd!('S', 'N', Matrix{T}(M))
    return U, σ
end

"""
    _reduce(::PODSVD, S, parameters, paramspace, k, tolerance)

The POD basis of the `N × m` snapshot `S` from its thin SVD. Its spectrum is `Λ = σ²`, and the
basis is the first `k` left singular vectors.
"""
function _reduce(alg::PODSVD, S::AbstractMatrix, parameters::NamedTuple,
        paramspace::ParameterSpace, k, tolerance)
    U, σ = _svd(S)
    Λ = σ .^ 2
    k = _basis_size(Λ, k, tolerance)
    return ReducedBasis(alg, parameters, paramspace, Λ, k, U[:, 1:k], nothing)
end

"""
    _reduce(::CotangentLiftSVD, S, parameters, paramspace, k, tolerance)

The cotangent-lift basis of the stacked `2N × m` snapshot `S = [Q; P]`, from the thin SVD of
`[Q P]`. Its spectrum is `Λ = σ²`, and `Φ` is the first `k` left singular vectors of `[Q P]`, so
that the basis is `diag(Φ, Φ)`.
"""
function _reduce(alg::CotangentLiftSVD, S::AbstractMatrix, parameters::NamedTuple,
        paramspace::ParameterSpace, k, tolerance)
    N = size(S, 1) ÷ 2
    Q = @view S[1:N, :]
    P = @view S[(N + 1):(2N), :]
    U, σ = _svd(hcat(Q, P))
    Λ = σ .^ 2
    k = _basis_size(Λ, k, tolerance)
    return ReducedBasis(alg, parameters, paramspace, Λ, k, U[:, 1:k], nothing)
end

"""
    _reduce(::ComplexSVD, S, parameters, paramspace, k, tolerance)

The complex-SVD basis of the stacked `2N × m` snapshot `S = [Q; P]`, from the thin SVD of the
complex `C = Q + iP`. Its spectrum is `Λ = σ²`, and `Φ` and `Ψ` are the real and imaginary parts
of the first `k` left singular vectors of `C`, so that the basis is `[Φ −Ψ; Ψ Φ]` and the lift of
`[a; b]` is `U(a + ib)`.

The imaginary part is `imag(U)` and never `conj(U)`: a basis built from `conj(U)` is symplectic
too, but it spans `X − iV`.
"""
function _reduce(alg::ComplexSVD, S::AbstractMatrix, parameters::NamedTuple,
        paramspace::ParameterSpace, k, tolerance)
    N = size(S, 1) ÷ 2
    U, σ = _svd(S[1:N, :] + im * S[(N + 1):(2N), :])
    Λ = σ .^ 2
    k = _basis_size(Λ, k, tolerance)
    U = U[:, 1:k]
    return ReducedBasis(alg, parameters, paramspace, Λ, k, real(U), imag(U))
end
