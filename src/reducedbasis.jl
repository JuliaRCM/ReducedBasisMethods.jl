abstract type ReductionAlgorithm end

"POD from the thin SVD of the `N × m` snapshot matrix `S`."
struct PODSVD <: ReductionAlgorithm end

"POD from the eigendecomposition of the Gram matrix `SᵀS`, which is `m × m`."
struct PODEVD <: ReductionAlgorithm end

"Cotangent lift from the thin SVD of `[Q P]`, where `S = [Q; P]`."
struct CotangentLiftSVD <: ReductionAlgorithm end

"Cotangent lift from the eigendecomposition of the Gram matrix `[Q P]ᵀ[Q P]`, which is `2m × 2m`."
struct CotangentLiftEVD <: ReductionAlgorithm end

"Complex SVD of `Q + iP`, where `S = [Q; P]`."
struct ComplexSVD <: ReductionAlgorithm end

"`PODSVD` and `PODEVD`, whose basis is a single `N × k` block."
const PODAlgorithm = Union{PODSVD, PODEVD}

"`CotangentLiftSVD`, `CotangentLiftEVD` and `ComplexSVD`, whose snapshot is the stacked `[q; p]`
and whose basis is `2N × 2k`."
const SymplecticAlgorithm = Union{CotangentLiftSVD, CotangentLiftEVD, ComplexSVD}

"Every tag, in the order the HDF5 `algorithm` key is read back."
const ALGORITHM_TAGS = (PODSVD, PODEVD, CotangentLiftSVD, CotangentLiftEVD, ComplexSVD)

"""
    ReducedBasis{DT, ALG, PAR, PS, ΨT}

A reduced basis of element type `DT`, built by the algorithm `algorithm`.

The fields are

- `algorithm`: the tag that built the basis;
- `parameters` and `paramspace`: what the basis belongs to, and where the tolerance that chose
  `k` can be recorded;
- `Λ`: the full spectrum, sorted descending and with its negative entries kept, so that a caller
  can plot it;
- `k`: the basis size, `1 ≤ k ≤ length(Λ)`;
- `Φ`: the `N × k` (POD) or `N × k` real factor of the basis;
- `Ψ`: the imaginary part `Im U` of the singular vectors whose real part is `Φ`, for `ComplexSVD`,
  and `nothing` for every other algorithm.

`Matrix(b)` assembles `Φ` for the POD algorithms, `diag(Φ, Φ)` for the cotangent lift and
`[Φ −Ψ; Ψ Φ]` for the complex SVD. `Ψ` is a type parameter, so the field costs nothing for an
algorithm that does not use it.
"""
struct ReducedBasis{DT <: Number, ALG <: ReductionAlgorithm, PAR <: NamedTuple,
    PS <: ParameterSpace, ΨT <: Union{Nothing, Matrix{DT}}}
    algorithm::ALG
    parameters::PAR
    paramspace::PS
    Λ::Vector{DT}
    k::Int
    Φ::Matrix{DT}
    Ψ::ΨT
end

"""
    _check_snapshot(S, algorithm)

The guards the constructor owes before it decomposes anything: a snapshot with a zero dimension
has no basis, and a PSD algorithm reduces the stacked `[q; p]` snapshot, which has an even number
of rows.
"""
function _check_snapshot(S::AbstractMatrix, algorithm::ReductionAlgorithm)
    if size(S, 1) == 0 || size(S, 2) == 0
        throw(ArgumentError("the snapshot matrix $(size(S)) has a zero dimension"))
    end
    if algorithm isa SymplecticAlgorithm && isodd(size(S, 1))
        throw(ArgumentError("a $(typeof(algorithm)) snapshot holds [q; p] and has an even " *
                            "number of rows, got $(size(S, 1))"))
    end
    return nothing
end

"""
    ReducedBasis(algorithm, S, parameters, paramspace; k = nothing, tolerance = nothing)

The reduced basis of the snapshot matrix `S`, built by `algorithm`.

`S` is the `N × m` snapshot for `PODSVD` and `PODEVD`, and the stacked `2N × m` matrix `[Q; P]` for
`CotangentLiftSVD`, `CotangentLiftEVD` and `ComplexSVD`.

Exactly one of `k` and `tolerance` says how large the basis is. `k` counts the retained columns
of `Φ` (and of `Φ + iΨ` for `ComplexSVD`); a PSD basis assembles `2k` columns. `tolerance` keeps
the leading factors that carry all but `tolerance` of the energy `sum(max.(Λ, 0))`.

Throws an `ArgumentError` when `S` has a zero dimension, when a PSD snapshot has an odd number of
rows, when `k` is outside `1:kmax`, where `kmax = min(N, m)` for the POD and complex algorithms and
`min(N, 2m)` for the cotangent lift, when `tolerance` is outside `[0, 1)`, and when both or neither
of `k` and `tolerance` are given.
"""
function ReducedBasis(algorithm::ReductionAlgorithm, S::AbstractMatrix,
        parameters::NamedTuple, paramspace::ParameterSpace;
        k::Union{Nothing, Integer} = nothing, tolerance::Union{Nothing, Real} = nothing)
    _check_snapshot(S, algorithm)
    return _reduce(algorithm, S, parameters, paramspace, k, tolerance)
end

"""
    _basis_size(Λ, k, tolerance, kmax = length(Λ))

The leading number of entries of the spectrum `Λ` the caller asked for, at most `kmax`.

`k` is checked in its own integer type and converted to `Int` only once it is inside `1:kmax`, so
`typemax(UInt)` and an oversized `BigInt` raise the `ArgumentError` the range promises, and not the
`InexactError` an earlier conversion would.

Under `tolerance`, the energy `Λ⁺ = max.(Λ, 0)` is accumulated and the first index whose running sum
reaches `(1 - tolerance)` of it is kept, capped at `kmax`. The clamp is what makes a negative entry,
which a Gram matrix of a rank-deficient snapshot produces, contribute no energy; the last index
always qualifies for finite cumulative energy, so the result stays inside `length(Λ)` even at
`tolerance = 0`. If non-finite energy leaves `findfirst` with nothing to find, the result is
`kmax`; non-finite snapshots are outside the supported input domain.
"""
function _basis_size(Λ::AbstractVector, k::Union{Nothing, Integer},
        tolerance::Union{Nothing, Real}, kmax::Int = length(Λ))
    if k === nothing && tolerance === nothing
        throw(ArgumentError("one of `k` and `tolerance` says how large the basis is"))
    elseif k !== nothing && tolerance !== nothing
        throw(ArgumentError("`k` and `tolerance` are mutually exclusive"))
    elseif k !== nothing
        1 ≤ k ≤ kmax ||
            throw(ArgumentError("k = $k is outside 1:$kmax"))
        return Int(k)
    end

    tol = float(tolerance)
    0 ≤ tol < 1 || throw(ArgumentError("tolerance = $tol is outside [0, 1)"))

    E = cumsum(max.(Λ, zero(eltype(Λ))))
    return min(something(findfirst(≥((1 - tol) * last(E)), E), kmax), kmax)
end

"""
    Matrix(b::ReducedBasis)

The assembled basis matrix: `Φ` for the POD algorithms, `diag(Φ, Φ)` for the cotangent lift and
`[Φ −Ψ; Ψ Φ]` for the complex SVD. Each is `2N × 2k` except the POD one, which is `N × k`.
"""
function Base.Matrix(b::ReducedBasis{DT, <:PODAlgorithm}) where {DT}
    return copy(b.Φ)
end

function Base.Matrix(b::ReducedBasis{
        DT, <:Union{CotangentLiftSVD, CotangentLiftEVD}}) where {DT}
    N = size(b.Φ, 1)
    A = zeros(DT, 2N, 2b.k)
    A[1:N, 1:b.k] .= b.Φ
    A[(N + 1):(2N), (b.k + 1):(2b.k)] .= b.Φ
    return A
end

function Base.Matrix(b::ReducedBasis{DT, <:ComplexSVD}) where {DT}
    N = size(b.Φ, 1)
    k = b.k
    A = zeros(DT, 2N, 2k)
    A[1:N, 1:k] .= b.Φ
    A[1:N, (k + 1):(2k)] .= .-b.Ψ
    A[(N + 1):(2N), 1:k] .= b.Ψ
    A[(N + 1):(2N), (k + 1):(2k)] .= b.Φ
    return A
end

"`string(nameof(typeof(algorithm)))` back to the algorithm, or an `ArgumentError`."
function _algorithm(name::AbstractString)
    for T in ALGORITHM_TAGS
        string(nameof(T)) == name && return T()
    end
    throw(ArgumentError("unknown reduction algorithm $name"))
end

function ReducedBasis(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    algorithm = _algorithm(read(group["algorithm"]))
    Λ = read(group["Λ"])
    k = read(group["k"])
    Φ = read(group["Φ"])
    Ψ = haskey(group, "Ψ") ? read(group["Ψ"]) : nothing

    parameters = read_parameters(group, "parameters")
    paramspace = ParameterSpace(group, "parameterspace")

    return ReducedBasis(algorithm, parameters, paramspace, Λ, k, Φ, Ψ)
end

function ReducedBasis(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        ReducedBasis(file, path)
    end
end

function h5save(h5::H5DataStore, b::ReducedBasis; path::AbstractString = "/")
    group = _create_group(h5, path)

    group["algorithm"] = string(nameof(typeof(b.algorithm)))

    group["k"] = b.k
    group["Λ"] = b.Λ
    group["Φ"] = b.Φ
    b.Ψ === nothing || (group["Ψ"] = b.Ψ)

    save_parameters(group, b.parameters; path = "parameters")
    h5save(group, b.paramspace; path = "parameterspace")
end

function h5load(::Type{ReducedBasis}, h5::H5DataStore; path::AbstractString = "/")
    ReducedBasis(h5, path)
end
