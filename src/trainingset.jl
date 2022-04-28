
struct TrainingSet{DT <: Number, PS <: ParameterSpace, ICS <: ParticleList{DT}, SS <: Snapshots{DT}}
    paramspace::PS
    initconds::ICS
    snapshots::SS

    function TrainingSet(paramspace::PS, initconds::ICS, snapshots::SS) where {DT, PS, ICS <: ParticleList{DT}, SS <: Snapshots{DT}}
        new{DT,PS,ICS,SS}(paramspace, initconds, snapshots)
    end
end

function TrainingSet(DT, nd, np, nh, nt, pspace::ParameterSpace, particles::ParticleList)
    snapshots = Snapshots(DT, nd, np, nh, nt, length(pspace))
    TrainingSet(pspace, particles, snapshots)
end

function TrainingSet(poisson::PoissonSolver{DT}, particles::ParticleList{DT}, nt::Int, pspace::ParameterSpace) where {DT}
    TrainingSet(DT, 1, length(particles), length(poisson), nt, pspace, particles)
end

Base.:(==)(ts1::TrainingSet, ts2::TrainingSet) = (
                        ts1.paramspace == ts2.paramspace
                     && ts1.initconds  == ts2.initconds
                     && ts1.snapshots  == ts2.snapshots)


function TrainingSet(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    paramspace = ParameterSpace(group, "parameterspace")
    initconds = ParticleList(group, "initial_conditions")
    snapshots = Snapshots(group, "snapshots")

    TrainingSet(paramspace, initconds, snapshots)
end

function TrainingSet(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        TrainingSet(file, path)
    end
end

function h5save(h5::H5DataStore, TS::TrainingSet; path::AbstractString = "/")
    group = _create_group(h5, path)

    h5save(group, TS.paramspace; path="parameterspace")
    Particles.h5save(group, TS.initconds; path="initial_conditions")
    h5save(group, TS.snapshots; path="snapshots")
end

function h5load(::Type{TrainingSet}, h5::H5DataStore; path::AbstractString = "/")
    TrainingSet(h5, path)
end
