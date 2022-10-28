
abstract type ReductionAlgorithm end

struct UnspecifiedAlgorithm <: ReductionAlgorithm end

struct CotangentLiftEVD <: ReductionAlgorithm end
struct CotangentLiftSVD <: ReductionAlgorithm end
# ...
# add parameters (e.g., k, tolerance) to structs


struct ReducedBasis{DT <: Number, ALG <: ReductionAlgorithm, PAR <: NamedTuple, PS <: ParameterSpace, ICS <: ParticleList{DT}, IP <: IntegratorParameters, PO <: PoissonSolver}
    algorithm::ALG

    parameters::PAR
    paramspace::PS
    initconds::ICS
    integrator::IP
    poisson::PO

    Λₚ::Vector{DT}
    kₚ::Int
    Ψₚ::Matrix{DT}

    Λₑ::Vector{DT}
    kₑ::Int
    Ψₑ::Matrix{DT}

    Πₑ::Matrix{DT}

    function ReducedBasis(algorithm::ALG, parameters::PAR, paramspace::PS, initconds::ICS, integrator::IP, poisson::POI,
                          Λₚ::AbstractArray{DT}, Ψₚ::AbstractArray{DT},
                          Λₑ::AbstractArray{DT}, Ψₑ::AbstractArray{DT},
                          Πₑ::AbstractArray{DT}) where {DT,ALG,PAR,PS,ICS,IP,POI}
        
        kₚ = length(axes(Ψₚ,2))
        kₑ = length(axes(Ψₑ,2))

        new{DT,ALG,PAR,PS,ICS,IP,POI}(algorithm, parameters, paramspace, initconds, integrator, poisson, Λₚ, kₚ, Ψₚ, Λₑ, kₑ, Ψₑ, Πₑ)
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
    initconds = ParticleList(group, "initial_conditions")
    integrator = IntegratorParameters(group, "integrator")
    poisson = PoissonSolverPBSplines(group)#, "poisson")

    ReducedBasis(UnspecifiedAlgorithm(), parameters, paramspace, initconds, integrator, poisson, Λₚ, Ψₚ, Λₑ, Ψₑ, Πₑ)
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
    ParticleMethods.h5save(group, rb.initconds; path = "initial_conditions")
    h5save(group, rb.integrator; path = "integrator")
    h5save(group, rb.poisson)#; path = "poisson")
end

function h5load(::Type{ReducedBasis}, h5::H5DataStore; path::AbstractString = "/")
    ReducedBasis(h5, path)
end
