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
- `k`: the basis size, `size(Φ, 2)`, which the constructor sets;
- `Φ`: the `N × k` basis for POD, and the `N × k` real factor of the basis otherwise;
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

    function ReducedBasis(algorithm::ReductionAlgorithm, parameters::NamedTuple,
            paramspace::ParameterSpace, Λ::Vector{DT}, Φ::Matrix{DT},
            Ψ::Union{Nothing, Matrix{DT}}) where {DT}
        return new{DT, typeof(algorithm), typeof(parameters), typeof(paramspace), typeof(Ψ)}(
            algorithm, parameters, paramspace, Λ, size(Φ, 2), Φ, Ψ)
    end
end

"""
    _left_matrix(algorithm, S)

The matrix whose leading left singular vectors are `Φ`: the snapshot `S` itself for POD, and the
`N × 2m` reshape of the stacked `S = [Q; P]` for the cotangent lift. The reshape interleaves the
columns of `Q` and `P`, a column permutation of `[Q P]`, so it has the left singular vectors and
the singular values of `[Q P]`, and it copies nothing.
"""
_left_matrix(::PODAlgorithm, S::AbstractMatrix) = S
function _left_matrix(::Union{CotangentLiftSVD, CotangentLiftEVD}, S::AbstractMatrix)
    return reshape(S, size(S, 1) ÷ 2, :)
end

"The largest basis size `k` the snapshot `S` admits under `algorithm`."
function _kmax(algorithm::ReductionAlgorithm, S::AbstractMatrix)
    minimum(size(_left_matrix(algorithm, S)))
end
_kmax(::ComplexSVD, S::AbstractMatrix) = min(size(S, 1) ÷ 2, size(S, 2))

"""
    _check_arguments(algorithm, S, k, tolerance)

The guards the constructor owes before it decomposes anything: a snapshot with a zero dimension
has no basis, a PSD algorithm reduces the stacked `[q; p]` snapshot, which has an even number of
rows, and exactly one of `k` and `tolerance` is given, inside its range.

`k` is compared in its own integer type, so `typemax(UInt)` and an oversized `BigInt` raise the
`ArgumentError` the range promises, and not the `InexactError` of a conversion to `Int`.
"""
function _check_arguments(algorithm::ReductionAlgorithm, S::AbstractMatrix,
        k::Union{Nothing, Integer}, tolerance::Union{Nothing, Real})
    if size(S, 1) == 0 || size(S, 2) == 0
        throw(ArgumentError("the snapshot matrix $(size(S)) has a zero dimension"))
    end
    if algorithm isa SymplecticAlgorithm && isodd(size(S, 1))
        throw(ArgumentError("a $(typeof(algorithm)) snapshot holds [q; p] and has an even " *
                            "number of rows, got $(size(S, 1))"))
    end
    if (k === nothing) == (tolerance === nothing)
        throw(ArgumentError("exactly one of `k` and `tolerance` says how large the basis is"))
    end
    kmax = _kmax(algorithm, S)
    k === nothing || 1 ≤ k ≤ kmax || throw(ArgumentError("k = $k is outside 1:$kmax"))
    tolerance === nothing || 0 ≤ tolerance < 1 ||
        throw(ArgumentError("tolerance = $tolerance is outside [0, 1)"))
    return nothing
end

"""
    ReducedBasis(algorithm, S, parameters, paramspace; k = nothing, tolerance = nothing)

The reduced basis of the real snapshot matrix `S`, built by `algorithm`.

`S` is the `N × m` snapshot for `PODSVD` and `PODEVD`, and the stacked `2N × m` matrix `[Q; P]` for
`CotangentLiftSVD`, `CotangentLiftEVD` and `ComplexSVD`. Its element type is `Float32` or
`Float64`, the real types LAPACK computes in, and the basis has the same element type.

Exactly one of `k` and `tolerance` says how large the basis is. `k` counts the retained columns
of `Φ` (and of `Φ + iΨ` for `ComplexSVD`); a PSD basis assembles `2k` columns. `tolerance` keeps
the leading factors that carry all but `tolerance` of the energy `sum(max.(Λ, 0))`.

Throws an `ArgumentError`, before any decomposition, when `S` has a zero dimension, when a PSD
snapshot has an odd number of rows, when `k` is outside `1:kmax`, where `kmax = min(N, m)` for the
POD and complex algorithms and `min(N, 2m)` for the cotangent lift, when `tolerance` is outside
`[0, 1)`, and when both or neither of `k` and `tolerance` are given.
"""
function ReducedBasis(
        algorithm::ReductionAlgorithm, S::AbstractMatrix{<:Union{Float32, Float64}},
        parameters::NamedTuple, paramspace::ParameterSpace;
        k::Union{Nothing, Integer} = nothing, tolerance::Union{Nothing, Real} = nothing)
    _check_arguments(algorithm, S, k, tolerance)
    return _reduce(algorithm, S, parameters, paramspace, k, tolerance)
end

"""
    _basis_size(Λ, k, tolerance, kmax = length(Λ))

The basis size: `k` as an `Int` when it is given, and otherwise the energy rule on the spectrum
`Λ`, at most `kmax`. The constructor has checked both keywords.

The energy `Λ⁺ = max.(Λ, 0)` is accumulated, and the first of the leading `kmax` indices whose
running sum reaches `(1 - tolerance)` of the whole energy is kept, or `kmax` if none does. The
clamp is what makes a negative entry, which a Gram matrix of a rank-deficient snapshot produces,
contribute no energy. The last index of `Λ` always qualifies for finite cumulative energy, so the
rule never runs past the spectrum, even at `tolerance = 0`. Non-finite energy leaves nothing to
find, and the result is `kmax`; non-finite snapshots are outside the supported input domain.
"""
function _basis_size(Λ::AbstractVector, k::Union{Nothing, Integer},
        tolerance::Union{Nothing, Real}, kmax::Int = length(Λ))
    k === nothing || return Int(k)
    E = cumsum(max.(Λ, zero(eltype(Λ))))
    return something(findfirst(≥((1 - tolerance) * last(E)), view(E, 1:kmax)), kmax)
end

"""
    Matrix(b::ReducedBasis)

The assembled basis matrix: `Φ` for the POD algorithms, `diag(Φ, Φ)` for the cotangent lift and
`[Φ −Ψ; Ψ Φ]` for the complex SVD. Each is `2N × 2k` except the POD one, which is `N × k`.
"""
Base.Matrix(b::ReducedBasis{<:Any, <:PODAlgorithm}) = copy(b.Φ)
function Base.Matrix(b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}})
    return [b.Φ zero(b.Φ); zero(b.Φ) b.Φ]
end
Base.Matrix(b::ReducedBasis{<:Any, <:ComplexSVD}) = [b.Φ -b.Ψ; b.Ψ b.Φ]

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
    Φ = read(group["Φ"])
    Ψ = haskey(group, "Ψ") ? read(group["Ψ"]) : nothing

    parameters = read_parameters(group, "parameters")
    paramspace = ParameterSpace(group, "parameterspace")

    return ReducedBasis(algorithm, parameters, paramspace, Λ, Φ, Ψ)
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
