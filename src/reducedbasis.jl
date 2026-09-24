
abstract type ReductionAlgorithm end

struct UnspecifiedAlgorithm <: ReductionAlgorithm end

struct CotangentLiftEVD <: ReductionAlgorithm end
struct CotangentLiftSVD <: ReductionAlgorithm end
# ...
# add parameters (e.g., k, tolerance) to structs

struct ReducedBasis{
    DT <: Number, ALG <: ReductionAlgorithm, PAR <: NamedTuple, PS <: ParameterSpace}
    algorithm::ALG

    parameters::PAR
    paramspace::PS

    Λₚ::Vector{DT}
    kₚ::Int
    Ψₚ::Matrix{DT}

    Λₑ::Vector{DT}
    kₑ::Int
    Ψₑ::Matrix{DT}

    Πₑ::Matrix{DT}

    function ReducedBasis(algorithm::ALG, parameters::PAR, paramspace::PS,
            Λₚ::AbstractArray{DT}, Ψₚ::AbstractArray{DT},
            Λₑ::AbstractArray{DT}, Ψₑ::AbstractArray{DT},
            Πₑ::AbstractArray{DT}) where {DT, ALG, PAR, PS}
        kₚ = length(axes(Ψₚ, 2))
        kₑ = length(axes(Ψₑ, 2))

        new{DT, ALG, PAR, PS}(
            algorithm, parameters, paramspace, Λₚ, kₚ, Ψₚ, Λₑ, kₑ, Ψₑ, Πₑ)
    end
end

function ReducedBasis(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    Λₚ = read(group["Λp"])
    Ψₚ = read(group["Ψp"])

    Λₑ = read(group["Λe"])
    Ψₑ = read(group["Ψe"])
    Πₑ = read(group["Πe"])

    parameters = read_parameters(group, "parameters")
    paramspace = ParameterSpace(group, "parameterspace")

    ReducedBasis(UnspecifiedAlgorithm(), parameters, paramspace, Λₚ, Ψₚ, Λₑ, Ψₑ, Πₑ)
end

function ReducedBasis(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        ReducedBasis(file, path)
    end
end

function h5save(h5::H5DataStore, rb::ReducedBasis; path::AbstractString = "/")
    group = _create_group(h5, path)

    # parameters
    group["kp"] = rb.kₚ
    group["ke"] = rb.kₑ

    # projections
    group["Λp"] = rb.Λₚ
    group["Ψp"] = rb.Ψₚ

    group["Λe"] = rb.Λₑ
    group["Ψe"] = rb.Ψₑ
    group["Πe"] = rb.Πₑ

    save_parameters(group, rb.parameters; path = "parameters")
    h5save(group, rb.paramspace; path = "parameterspace")
end

function h5load(::Type{ReducedBasis}, h5::H5DataStore; path::AbstractString = "/")
    ReducedBasis(h5, path)
end
