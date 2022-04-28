
abstract type ReductionAlgorithm end

struct UnspecifiedAlgorithm <: ReductionAlgorithm end

struct CotangentLift <: ReductionAlgorithm end
struct EVD <: ReductionAlgorithm end
# ...


struct ReducedBasis{DT <: Number, ALG <: ReductionAlgorithm, PS <: ParameterSpace}
    paramspace::PS
    algorithm::ALG

    Λₚ::Vector{DT}
    Ωₚ::Matrix{DT}
    kₚ::Int
    Ψₚ::Matrix{DT}

    Λₑ::Vector{DT}
    Ωₑ::Matrix{DT}
    kₑ::Int
    Ψₑ::Matrix{DT}

    function ReducedBasis(pspace::PS, algorithm::ALG,
                          Λₚ::AbstractArray{DT}, Ωₚ::AbstractArray{DT}, kₚ::Int, Ψₚ::AbstractArray{DT},
                          Λₑ::AbstractArray{DT}, Ωₑ::AbstractArray{DT}, kₑ::Int, Ψₑ::AbstractArray{DT}) where {DT,ALG,PS}
        new{DT,ALG,PS}(pspace, algorithm, Λₚ, Ωₚ, kₚ, Ψₚ, Λₑ, Ωₑ, kₑ, Ψₑ)
    end
end

function ReducedBasis(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    Λₚ = read(group["Λp"])
    Ωₚ = read(group["Ωp"])
    kₚ = read(group["kp"])
    Ψₚ = read(group["Ψp"])

    Λₑ = read(group["Λe"])
    Ωₑ = read(group["Ωe"])
    kₑ = read(group["ke"])
    Ψₑ = read(group["Ψe"])

    pspace = ParameterSpace(group, "parameterspace")

    ReducedBasis(pspace, UnspecifiedAlgorithm(), Λₚ, Ωₚ, kₚ, Ψₚ, Λₑ, Ωₑ, kₑ, Ψₑ)
end

function ReducedBasis(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        ReducedBasis(file, path)
    end
end

function h5save(h5::H5DataStore, rb::ReducedBasis; path::AbstractString = "/")
    group = _create_group(h5, path)

    group["Λp"] = rb.Λₚ
    group["Ωp"] = rb.Ωₚ
    group["kp"] = rb.kₚ
    group["Ψp"] = rb.Ψₚ

    group["Λe"] = rb.Λₑ
    group["Ωe"] = rb.Ωₑ
    group["ke"] = rb.kₑ
    group["Ψe"] = rb.Ψₑ

    h5save(group, TS.paramspace; path="parameterspace")
end

function h5load(::Type{ReducedBasis}, h5::H5DataStore; path::AbstractString = "/")
    ReducedBasis(h5, path)
end
