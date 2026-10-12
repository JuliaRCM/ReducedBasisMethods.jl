"""
    _svd_input(algorithm, S)

The matrix whose thin SVD gives the basis, as the owned copy that `gesvd!` overwrites: the
`_left_matrix` of `S` for POD and the cotangent lift, and the complex `Q + iP` of the stacked
`S = [Q; P]` for `ComplexSVD`.
"""
function _svd_input(algorithm::ReductionAlgorithm, S::AbstractMatrix)
    Matrix(_left_matrix(algorithm, S))
end
function _svd_input(::ComplexSVD, S::AbstractMatrix)
    N = size(S, 1) ÷ 2
    return complex.(view(S, 1:N, :), view(S, (N + 1):(2N), :))
end

"The basis factors `(Φ, Ψ)` of the leading left singular vectors `U`."
_factors(::ReductionAlgorithm, U::Matrix) = (U, nothing)
_factors(::ComplexSVD, U::Matrix) = (real(U), imag(U))

"""
    _reduce(::Union{PODSVD, CotangentLiftSVD, ComplexSVD}, S, parameters, paramspace, k, tolerance)

The basis from the thin SVD of `S` for `PODSVD`, of `[Q P]` for `CotangentLiftSVD` and of the
complex `C = Q + iP` for `ComplexSVD`, where `S = [Q; P]`. Its spectrum is `Λ = σ²`, and the basis
is built from the first `k` left singular vectors `U`: `Φ = U` for POD and the cotangent lift, so
that the cotangent-lift basis is `diag(Φ, Φ)`, and `Φ = Re U`, `Ψ = Im U` for the complex SVD, so
that its basis is `[Φ −Ψ; Ψ Φ]` and the lift of `[a; b]` is `U(a + ib)`.

The imaginary part is `imag(U)` and never `conj(U)`: a basis built from `conj(U)` is symplectic
too, but it spans `X − iV`.

LAPACK's `gesvd!` runs with `jobu = 'S'` and `jobvt = 'N'`. The right factor is not formed because
no basis uses it, and for the snapshot matrices this package reduces it costs as much memory as the
snapshot itself.
"""
function _reduce(algorithm::Union{PODSVD, CotangentLiftSVD, ComplexSVD}, S::AbstractMatrix,
        parameters::NamedTuple, paramspace::ParameterSpace, k, tolerance)
    U, σ, _ = LinearAlgebra.LAPACK.gesvd!('S', 'N', _svd_input(algorithm, S))
    Λ = σ .^ 2
    U = U[:, 1:_basis_size(Λ, k, tolerance)]
    return ReducedBasis(algorithm, parameters, paramspace, Λ, _factors(algorithm, U)...)
end
