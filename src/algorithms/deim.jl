"""
    _deim_indices(Ψ)

The rows selected by the greedy DEIM algorithm (Algorithm 1 of
[doi:10.1137/090766498](https://doi.org/10.1137/090766498)) on the basis `Ψ`, whose `N × k`
columns are the basis vectors.

The index of column `i` is the row of the largest magnitude of the residual
`r = Ψ[:, i] - Ψ[:, 1:i-1] * c`, where `c` solves the interpolation problem of the columns
selected so far. The residual of column 1 is the column itself. The magnitude decides, so a
residual that peaks at a negative entry selects that entry; ties go to the first index.

Throws an `ArgumentError` where `k > N`, and for a column whose largest residual magnitude `ρ`
is not both positive and finite — `ρ == 0` for a zero column or one exactly dependent on the
ones before it, and `NaN` or `Inf` for a `NaN` or an `Inf` in `Ψ`. Every entry of a column
enters that column's residual, so the test of each column covers the whole matrix.
"""
function _deim_indices(Ψ::AbstractMatrix)
    N, k = size(Ψ)
    k ≤ N || throw(ArgumentError(
        "the DEIM basis has $k columns but only $N rows, so k ≤ N is required"))

    k == 0 && return Int[]

    p = Vector{Int}(undef, k)
    r = Vector{float(eltype(Ψ))}(undef, N)

    # One residual test per column, whether the residual is the column itself or what the
    # interpolation of the columns already selected leaves behind: it returns the row of the
    # largest magnitude of `r`, and throws, naming the column, where that magnitude is not both
    # positive and finite.
    function select_row(r, i)
        ρ, j = findmax(abs, r)
        0 < ρ < Inf || throw(ArgumentError(
            "max|r| = $ρ for column $i of the DEIM basis, so the column is zero or dependent " *
            "on the ones before it, or it holds a NaN or an Inf"))
        return j
    end

    r .= @view Ψ[:, 1]
    p[1] = select_row(r, 1)

    for i in 2:k
        c = @views (Ψ[p[1:(i - 1)], 1:(i - 1)]) \ (Ψ[p[1:(i - 1)], i])
        r .= @views Ψ[:, i] .- Ψ[:, 1:(i - 1)] * c
        p[i] = select_row(r, i)
    end

    return p
end

"""
    get_DEIM_interpolation_matrix(Ψ)

Calculates the DEIM interpolation matrix as described in Algorithm 1 in
[doi:10.1137/090766498](https://doi.org/10.1137/090766498) to cheaply evaluate f(x).

The input matrix Ψ holds the eigenvectors of the snapshot matrix fᵢ(xⱼ), 1 ≤ i ≤ N, 1 ≤ j ≤ k.
Returns the `N × k` matrix `Π` whose column `i` is the unit vector of the row that
[`_deim_indices`](@ref) selects for that column, so that `f(x) ≈ Ψ * inv(Π' * Ψ) * Π' * f(x)`.

Its element type is `float(eltype(Ψ))`, so a `Float32` basis gives a `Float32` interpolation
matrix. The selection, and with it the `ArgumentError` for `k > N` and for a column whose
largest residual magnitude is not finite and positive, is [`_deim_indices`](@ref)'s.

Each column of `Π` is a unit vector, so the output is exact whatever the input: the `jldoctest`
below prints the matrix itself rather than a rounded form of it. A column of `Ψ` that is an exact
combination of the ones before it is refused instead — its residual is zero, which is the
`ArgumentError` the note above names.

```jldoctest
julia> using ReducedBasisMethods

julia> get_DEIM_interpolation_matrix([1 1; 2 -1])
2×2 Matrix{Float64}:
 0.0  1.0
 1.0  0.0
```
"""
function get_DEIM_interpolation_matrix(Ψ::AbstractMatrix)
    N, k = size(Ψ)
    p = _deim_indices(Ψ)
    Π = zeros(float(eltype(Ψ)), N, k)
    for i in 1:k
        Π[p[i], i] = one(eltype(Π))
    end
    return Π
end
