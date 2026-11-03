"""
    get_DEIM_indices(Ψ)

Returns the `k` rows that the greedy DEIM algorithm (Algorithm 1 of
[doi:10.1137/090766498](https://doi.org/10.1137/090766498)) selects on the basis `Ψ`, whose
`N × k` columns are the basis vectors.

This is the index form of [`get_DEIM_interpolation_matrix`](@ref): for
`p = get_DEIM_indices(Ψ)` and `Π = get_DEIM_interpolation_matrix(Ψ)`, `Π' * A == A[p, :]` for
every finite `N`-row matrix `A`, so `A[p, :]` evaluates `Π' * A` without forming `Π`.

The index of column `i` is the row of the largest magnitude of the residual
`r = Ψ[:, i] - Ψ[:, 1:i-1] * c`, where `c` solves the interpolation problem of the columns
selected so far. The residual of column 1 is the column itself. The magnitude decides, so a
residual that peaks at a negative entry selects that entry; ties go to the first index.

Throws an `ArgumentError` where `k > N`, and for a column whose largest residual magnitude `ρ`
is not both positive and finite — `ρ == 0` for a zero column or one exactly dependent on the
ones before it, and `NaN` or `Inf` for a `NaN` or an `Inf` in `Ψ`. Every entry of a column
enters that column's residual, so the test of each column covers the whole matrix.
"""
function get_DEIM_indices(Ψ::AbstractMatrix)
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
[`get_DEIM_indices`](@ref) selects for that column, so that `f(x) ≈ Ψ * inv(Π' * Ψ) * Π' * f(x)`.

Its element type is `float(eltype(Ψ))`, so a `Float32` basis gives a `Float32` interpolation
matrix. The selection, and with it the `ArgumentError` for `k > N` and for a column whose
largest residual magnitude is not finite and positive, is [`get_DEIM_indices`](@ref)'s.

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
    p = get_DEIM_indices(Ψ)
    Π = zeros(float(eltype(Ψ)), N, k)
    for i in 1:k
        Π[p[i], i] = one(eltype(Π))
    end
    return Π
end

"""
    save_DEIM_interpolation_matrix(h5::H5DataStore, Π::AbstractMatrix; path = "/")

Writes the interpolation matrix `Π` of [`get_DEIM_interpolation_matrix`](@ref) as the dense
dataset `"Π"` of the group at `path`, and creates that group where it does not exist. A wrapper
such as an `Adjoint` is stored as `Matrix(Π)`. HDF5 keeps the element type.

A group at `path` that already holds a dataset `"Π"` raises HDF5's error, and the stored matrix
stays. `Π` is stored as it is given: nothing checks that it is an interpolation matrix.

`h5` is an open HDF5 file or group. A `path` that starts with `/`, the default included, starts
at the root of the file also where `h5` is a group. Any other `path` starts at `h5`.
"""
function save_DEIM_interpolation_matrix(
        h5::H5DataStore, Π::AbstractMatrix; path::AbstractString = "/")
    group = _create_group(h5, path)
    group["Π"] = Matrix(Π)
end

"""
    load_DEIM_interpolation_matrix(h5::H5DataStore; path = "/")

Returns the matrix that [`save_DEIM_interpolation_matrix`](@ref) wrote to the group at `path`,
with its element type. Without a group at `path` or a dataset `"Π"` in it, HDF5's `KeyError`
stands.

`h5` and `path` read as they do in [`save_DEIM_interpolation_matrix`](@ref): the default `path`
is the root of the file also where `h5` is a group.
"""
function load_DEIM_interpolation_matrix(h5::H5DataStore; path::AbstractString = "/")
    read(h5[path]["Π"])
end
