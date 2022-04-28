
struct TrainingSet{DT <: Number, PAR <: NamedTuple, PS <: ParameterSpace, ICS <: ParticleList{DT}, SS <: Snapshots{DT}, IP <: IntegratorParameters, PO <: PoissonSolver}
    parameters::PAR
    paramspace::PS
    initconds::ICS
    snapshots::SS
    integrator::IP
    poisson::PO

    function TrainingSet(parameters::PAR, paramspace::PS, initconds::ICS, snapshots::SS, integrator::IP, poisson::POI) where {DT, PAR, PS, ICS <: ParticleList{DT}, SS <: Snapshots{DT}, IP, POI}
        new{DT,PAR,PS,ICS,SS,IP,POI}(parameters, paramspace, initconds, snapshots, integrator, poisson)
    end
end

function TrainingSet(DT, nd, np, nh, nt, parameters::NamedTuple, pspace::ParameterSpace, particles::ParticleList, ip::IntegratorParameters, poisson::PoissonSolver)
    snapshots = Snapshots(DT, nd, np, nh, nt, length(pspace))
    TrainingSet(parameters, pspace, particles, snapshots, ip, poisson)
end

function TrainingSet(particles::ParticleList{DT}, poisson::PoissonSolver{DT}, nt::Int, parameters::NamedTuple, pspace::ParameterSpace, ip::IntegratorParameters) where {DT}
    TrainingSet(DT, 1, length(particles), length(poisson), nt, parameters, pspace, particles, ip, poisson)
end

Base.:(==)(ts1::TrainingSet, ts2::TrainingSet) = (
                        ts1.parameters == ts2.parameters
                     && ts1.paramspace == ts2.paramspace
                     && ts1.initconds  == ts2.initconds
                     && ts1.snapshots  == ts2.snapshots
                     && ts1.integrator == ts2.integrator
                     && ts1.poisson    == ts2.poisson)


function TrainingSet(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    parameters = read_parameters(group, "parameters")
    paramspace = ParameterSpace(group, "parameterspace")
    initconds = ParticleList(group, "initial_conditions")
    snapshots = Snapshots(group, "snapshots")
    integrator = IntegratorParameters(group, "integrator")
    poisson = PoissonSolverPBSplines(group)#, "poisson")

    TrainingSet(parameters, paramspace, initconds, snapshots, integrator, poisson)
end

function TrainingSet(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        TrainingSet(file, path)
    end
end

function h5save(h5::H5DataStore, TS::TrainingSet; path::AbstractString = "/")
    group = _create_group(h5, path)

    save_parameters(group, TS.parameters; path = "parameters")
    h5save(group, TS.paramspace; path = "parameterspace")
    Particles.h5save(group, TS.initconds; path = "initial_conditions")
    h5save(group, TS.snapshots; path = "snapshots")
    h5save(group, TS.integrator; path = "integrator")
    h5save(group, TS.poisson)#; path = "poisson")
end

function h5load(::Type{TrainingSet}, h5::H5DataStore; path::AbstractString = "/")
    TrainingSet(h5, path)
end
