
struct TrainingSet{DT <: Number, PS <: ParameterSpace}
    paramspace::PS

    X::Array{DT,4}
    V::Array{DT,4}
    A::Array{DT,4}
    Φ::Array{DT,4}
    W::Array{DT,2}
    K::Array{DT,2}
    M::Array{DT,2}

    function TrainingSet(pspace::PS, X::AbstractArray{DT}, V::AbstractArray{DT}, A::AbstractArray{DT}, Φ::AbstractArray{DT}, W::AbstractArray{DT}, K::AbstractArray{DT}, M::AbstractArray{DT}) where {DT,PS}
        new{DT,PS}(pspace, X, V, A, Φ, W, K, M)
    end
end

function TrainingSet(DT, nd, np, nh, nt, pspace::ParameterSpace)
    X = zeros(nd, np, nt, length(pspace))
    V = zeros(nd, np, nt, length(pspace))
    A = zeros(nd, np, nt, length(pspace))
    Φ = zeros(nd, nh, nt, length(pspace))
    W = zeros(nt, length(pspace))
    K = zeros(nt, length(pspace))
    M = zeros(nt, length(pspace))
    TrainingSet(pspace, X, V, A, Φ, W, K, M)
end

function TrainingSet(poisson::PoissonSolver{DT}, particles::ParticleList{DT}, nt::Int, pspace::ParameterSpace) where {DT}
    TrainingSet(DT, 1, length(particles), length(poisson), nt, pspace)
end

Base.:(==)(ts1::TrainingSet, ts2::TrainingSet) = (
                        ts1.paramspace == ts2.paramspace
                     && ts1.X == ts2.X
                     && ts1.V == ts2.V
                     && ts1.A == ts2.A
                     && ts1.Φ == ts2.Φ
                     && ts1.W == ts2.W
                     && ts1.K == ts2.K
                     && ts1.M == ts2.M)


function TrainingSet(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    X = read(group["X"])
    V = read(group["V"])
    A = read(group["A"])
    Φ = read(group["Φ"])
    W = read(group["W"])
    K = read(group["K"])
    M = read(group["M"])

    pspace = ParameterSpace(group, "parameterspace")

    TrainingSet(pspace, X, V, A, Φ, W, K, M)
end

function TrainingSet(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        TrainingSet(file, path)
    end
end

function h5save(h5::H5DataStore, TS::TrainingSet; path::AbstractString = "/")
    group = _create_group(h5, path)

    group["X"] = TS.X
    group["V"] = TS.V
    group["A"] = TS.A
    group["Φ"] = TS.Φ
    group["W"] = TS.W
    group["K"] = TS.K
    group["M"] = TS.M

    h5save(group, TS.paramspace; path="parameterspace")
end

function h5load(::Type{TrainingSet}, h5::H5DataStore; path::AbstractString = "/")
    TrainingSet(h5, path)
end
