"""
    _reduce(::Union{PODEVD, CotangentLiftEVD}, S, parameters, paramspace, k, tolerance)

The basis from the eigendecomposition of the Gram matrix `MᵀM` of `M = _left_matrix(S)`: the
`m × m` matrix `SᵀS` for `PODEVD`, and the `2m × 2m` Gram matrix of `[Q P]`, up to a permutation,
for `CotangentLiftEVD`, where `S = [Q; P]`.

The spectrum is the whole Gram spectrum, `σ²` with the round-off eigenvalues of both signs kept,
sorted descending by value, never by magnitude: `|λ|` would put the round-off eigenvalues of a
rank-deficient snapshot among the leading ones. It is stored as it is, so a caller that plots it
sees every eigenvalue of `MᵀM`. The basis is cut to `kmax = min(size(M)...)` entries, the most `M`
admits.

`Φ` is the thin Householder `Q` of `M·Ω[:, 1:k]`, where `Ω` holds the sorted eigenvectors. Its
columns are orthonormal even where `M·Ω[:, 1:k]` loses rank, and no division by a round-off
eigenvalue is needed. For each full-rank leading block of `M·Ω`, the first `j` columns of `Φ` span
that block; at rank loss, `Φ` completes an orthonormal basis, so its projector can include
directions outside the column space of `M`.

The Gram matrix squares the snapshot entries and can overflow for large finite inputs.
"""
function _reduce(algorithm::Union{PODEVD, CotangentLiftEVD}, S::AbstractMatrix,
        parameters::NamedTuple, paramspace::ParameterSpace, k, tolerance)
    M = _left_matrix(algorithm, S)
    F = eigen(Symmetric(M' * M))
    Λ = reverse(F.values)
    Ω = reverse(F.vectors; dims = 2)
    k = _basis_size(Λ, k, tolerance, _kmax(algorithm, S))
    Φ = Matrix(qr(M * view(Ω, :, 1:k)).Q)
    return ReducedBasis(algorithm, parameters, paramspace, Λ, Φ, nothing)
end
