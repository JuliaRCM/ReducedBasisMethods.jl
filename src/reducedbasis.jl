
abstract type ReductionAlgorithm end

struct UnspecifiedAlgorithm <: ReductionAlgorithm end

struct CotangentLift <: ReductionAlgorithm end
struct EVD <: ReductionAlgorithm end
# ...


struct ReducedBasis{DT <: Number, ALG <: ReductionAlgorithm, PAR <: NamedTuple, PS <: ParameterSpace, ICS <: ParticleList{DT}, IP <: IntegratorParameters, PO <: PoissonSolver}
    algorithm::ALG

    parameters::PAR
    paramspace::PS
    initconds::ICS
    integrator::IP
    poisson::PO

    Λₚ::Vector{DT}
    Ωₚ::Matrix{DT}
    kₚ::Int
    Ψₚ::Matrix{DT}

    Λₑ::Vector{DT}
    Ωₑ::Matrix{DT}
    kₑ::Int
    Ψₑ::Matrix{DT}

    function ReducedBasis(algorithm::ALG, parameters::PAR, paramspace::PS, initconds::ICS, integrator::IP, poisson::POI,
                          Λₚ::AbstractArray{DT}, Ωₚ::AbstractArray{DT}, kₚ::Int, Ψₚ::AbstractArray{DT},
                          Λₑ::AbstractArray{DT}, Ωₑ::AbstractArray{DT}, kₑ::Int, Ψₑ::AbstractArray{DT}) where {DT,ALG,PAR,PS,ICS,IP,POI}
        new{DT,ALG,PAR,PS,ICS,IP,POI}(algorithm, parameters, paramspace, initconds, integrator, poisson, Λₚ, Ωₚ, kₚ, Ψₚ, Λₑ, Ωₑ, kₑ, Ψₑ)
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

    parameters = read_parameters(group, "parameters")
    paramspace = ParameterSpace(group, "parameterspace")
    initconds = ParticleList(group, "initial_conditions")
    integrator = IntegratorParameters(group, "integrator")
    poisson = PoissonSolverPBSplines(group)#, "poisson")

    ReducedBasis(UnspecifiedAlgorithm(), parameters, paramspace, initconds, integrator, poisson, Λₚ, Ωₚ, kₚ, Ψₚ, Λₑ, Ωₑ, kₑ, Ψₑ)
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
    group["Ωp"] = rb.Ωₚ
    group["Ψp"] = rb.Ψₚ

    group["Λe"] = rb.Λₑ
    group["Ωe"] = rb.Ωₑ
    group["Ψe"] = rb.Ψₑ

    save_parameters(group, rb.parameters; path = "parameters")
    h5save(group, rb.paramspace; path="parameterspace")
    Particles.h5save(group, rb.initconds; path = "initial_conditions")
    h5save(group, rb.integrator; path = "integrator")
    h5save(group, rb.poisson)#; path = "poisson")
end

function h5load(::Type{ReducedBasis}, h5::H5DataStore; path::AbstractString = "/")
    ReducedBasis(h5, path)
end
