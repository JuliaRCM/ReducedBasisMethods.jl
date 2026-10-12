"""
    project(b::ReducedBasis, x)

The reduced coordinates of `x` in the basis `b`, for a vector or a matrix of snapshots.

The map is the symplectic inverse `A⁺` of the assembled `A = Matrix(b)` for the two PSD
structures and `Φᵀ` for the POD ones:

- POD: `Φᵀx`;
- cotangent lift: `[Φᵀq; Φᵀp]` with `x = [q; p]`;
- complex SVD: `[Φᵀq + Ψᵀp; Φᵀp - Ψᵀq]` with `x = [q; p]`.

For both PSD structures `A` commutes with `J` (`J_N A = A J_k`), so `A⁺ = J_kᵀAᵀJ_N = Aᵀ`, which
the tests assert against a dense `J`. `Aᵀ` is applied blockwise, and neither `A` nor `J` is formed.

The two PSD structures read `x` through views, so a call allocates the result and `k`-row
temporaries, and nothing proportional to the number of rows of `x`.

`project` is `ReducedComplexityModeling.project`: this package imports the generic and adds a
method, so a caller that loads both packages sees one name.
"""
function project(b::ReducedBasis{<:Any, <:PODAlgorithm}, x::AbstractMatrix)
    return b.Φ' * x
end

function project(b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        x::AbstractMatrix)
    N = size(b.Φ, 1)
    _check_rows(x, 2N, "a cotangent-lift projection")
    return [b.Φ' * view(x, 1:N, :); b.Φ' * view(x, (N + 1):(2N), :)]
end

function project(b::ReducedBasis{<:Any, <:ComplexSVD}, x::AbstractMatrix)
    N = size(b.Φ, 1)
    _check_rows(x, 2N, "a complex-SVD projection")
    q = view(x, 1:N, :)
    p = view(x, (N + 1):(2N), :)
    return [b.Φ' * q + b.Ψ' * p; b.Φ' * p - b.Ψ' * q]
end

project(b::ReducedBasis, x::AbstractVector) = vec(project(b, reshape(x, :, 1)))

"""
    lift(b::ReducedBasis, z)

The full-space vector of the reduced coordinates `z`, for a vector or a matrix of columns. It is
the assembled `A = Matrix(b)` applied to `z`, and a right inverse of `project`:
`project(b, lift(b, z)) ≈ z`. A POD basis lifts to `Φz`, the cotangent lift to `[Φa; Φb]` with
`z = [a; b]`, and the complex SVD to `[Φa - Ψb; Ψa + Φb]`.
"""
function lift(b::ReducedBasis{<:Any, <:PODAlgorithm}, z::AbstractMatrix)
    return b.Φ * z
end

function lift(b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        z::AbstractMatrix)
    N, k = size(b.Φ)
    _check_rows(z, 2k, "a cotangent-lift lift")
    return reshape(b.Φ * reshape(z, k, :), 2N, :)
end

function lift(b::ReducedBasis{<:Any, <:ComplexSVD}, z::AbstractMatrix)
    N, k = size(b.Φ)
    _check_rows(z, 2k, "a complex-SVD lift")
    a = view(z, 1:k, :)
    c = view(z, (k + 1):(2k), :)
    T = promote_type(eltype(b.Φ), eltype(z))
    x = similar(b.Φ, T, 2N, size(z, 2))
    mul!(view(x, 1:N, :), b.Φ, a)
    mul!(view(x, 1:N, :), b.Ψ, c, -one(T), one(T))
    mul!(view(x, (N + 1):(2N), :), b.Ψ, a)
    mul!(view(x, (N + 1):(2N), :), b.Φ, c, one(T), one(T))
    return x
end

lift(b::ReducedBasis, z::AbstractVector) = vec(lift(b, reshape(z, :, 1)))

"""
    _check_rows(X, n, what)

The row guard the blockwise maps owe before they slice: `X` holds `n` rows, or the slice would
drop the surplus ones in silence where a matrix product raises a `DimensionMismatch`.
"""
function _check_rows(X::AbstractMatrix, n::Int, what::AbstractString)
    size(X, 1) == n ||
        throw(DimensionMismatch("$what expects $n rows, got $(size(X, 1))"))
    return nothing
end

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
