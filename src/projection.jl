"""
    project(b::ReducedBasis, x)

The reduced coordinates of `x` in the basis `b`, for a vector or a matrix of snapshots.

The map is the symplectic inverse `A⁺` of the assembled `A = Matrix(b)` for the two PSD
structures and `Φᵀ` for the POD ones:

- POD: `Φᵀx`;
- cotangent lift: `[Φᵀq; Φᵀp]` with `x = [q; p]`;
- complex SVD: `[Φᵀq + Ψᵀp; Φᵀp - Ψᵀq]` with `x = [q; p]`.

`A⁺` is applied blockwise, and `J` is never formed. For both PSD structures `A⁺ = J_kᵀAᵀJ_N`,
which the tests assert against a dense `J`.

`project` is `ReducedComplexityModeling.project`: this package imports the generic and adds a
method, so a caller that loads both packages sees one name.
"""
function project(b::ReducedBasis{<:Any, <:PODAlgorithm}, x::AbstractMatrix)
    return b.Φ' * x
end

function project(b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        x::AbstractMatrix)
    N = size(b.Φ, 1)
    return [b.Φ' * x[1:N, :]; b.Φ' * x[(N + 1):(2N), :]]
end

function project(b::ReducedBasis{<:Any, <:ComplexSVD}, x::AbstractMatrix)
    N = size(b.Φ, 1)
    q = x[1:N, :]
    p = x[(N + 1):(2N), :]
    return [b.Φ' * q + b.Ψ' * p; b.Φ' * p - b.Ψ' * q]
end

project(b::ReducedBasis, x::AbstractVector) = vec(project(b, reshape(x, :, 1)))

"""
    lift(b::ReducedBasis, z)

The full-space vector of the reduced coordinates `z`, for a vector or a matrix of columns. It is
the assembled `A = Matrix(b)` applied to `z`, and the right inverse of `project`: a POD basis
lifts to `Φz`, the cotangent lift to `[Φa; Φb]` with `z = [a; b]`, and the complex SVD to
`[Φa - Ψb; Ψa + Φb]`.
"""
function lift(b::ReducedBasis{<:Any, <:PODAlgorithm}, z::AbstractMatrix)
    return b.Φ * z
end

function lift(b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        z::AbstractMatrix)
    k = b.k
    return [b.Φ * z[1:k, :]; b.Φ * z[(k + 1):(2k), :]]
end

function lift(b::ReducedBasis{<:Any, <:ComplexSVD}, z::AbstractMatrix)
    k = b.k
    a = z[1:k, :]
    c = z[(k + 1):(2k), :]
    return [b.Φ * a - b.Ψ * c; b.Ψ * a + b.Φ * c]
end

lift(b::ReducedBasis, z::AbstractVector) = vec(lift(b, reshape(z, :, 1)))

"""
    projection_error(b::ReducedBasis, X)

`norm(X - lift(b, project(b, X)))`, the absolute Frobenius norm of what the basis's own truncation
removes from `X`, for a vector or a matrix. A relative error is the caller's division.
"""
projection_error(b::ReducedBasis, X) = norm(X - lift(b, project(b, X)))

"""
    reduction_error(b::ReducedBasis, X, Z)

`norm(X - lift(b, Z))`, the absolute Frobenius norm of the error of the reduced trajectory `Z`
against the full `X`, for a vector or a matrix. It lifts `Z`, and does not project `X` again.
"""
reduction_error(b::ReducedBasis, X, Z) = norm(X - lift(b, Z))
