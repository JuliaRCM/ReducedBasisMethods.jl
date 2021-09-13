
struct TrainingSet{DT <: Number, PS <: ParameterSpace}
    paramspace::PS

    X::Array{DT,4}
    V::Array{DT,4}
    A::Array{DT,4}
    Φ::Array{DT,4}
    W::Array{DT,2}
    K::Array{DT,2}
    M::Array{DT,2}

    function TrainingSet(pspace::PS, X::Array{DT}, V::Array{DT}, A::Array{DT}, Φ::Array{DT}, W::Array{DT}, K::Array{DT}, M::Array{DT}) where {DT,PS}
        new{DT,PS}(pspace, X, V, A, Φ, W, K, M)
    end
end

function TrainingSet(DT, ns, nd, np, nh, pspace::ParameterSpace)
    X = zeros(nd, np, ns, length(pspace))
    V = zeros(nd, np, ns, length(pspace))
    A = zeros(nd, np, ns, length(pspace))
    Φ = zeros(nd, nh, ns, length(pspace))
    W = zeros(ns, length(pspace))
    K = zeros(ns, length(pspace))
    M = zeros(ns, length(pspace))
    TrainingSet(pspace, X, V, A, Φ, W, K, M)
end

function TrainingSet(ns::Int, poisson::PoissonSolver{DT}, particles::ParticleList{DT}, pspace::ParameterSpace) where {DT}
    TrainingSet(DT, ns, 1, length(particles), length(poisson), pspace)
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


function TrainingSet(h5::H5DataStore, path::AbstractString="/")
    X = read(h5["$path/X"])
    V = read(h5["$path/V"])
    A = read(h5["$path/A"])
    Φ = read(h5["$path/Φ"])
    W = read(h5["$path/W"])
    K = read(h5["$path/K"])
    M = read(h5["$path/M"])

    pspace = ParameterSpace(h5, "$path/parameterspace")

    TrainingSet(pspace, X, V, A, Φ, W, K, M)
end

function h5save(TS::TrainingSet, h5::H5DataStore, path::AbstractString="/")
    s = _create_group(h5, path)
    s["X"] = TS.X
    s["V"] = TS.V
    s["A"] = TS.A
    s["Φ"] = TS.Φ
    s["W"] = TS.W
    s["K"] = TS.K
    s["M"] = TS.M

    h5save(TS.paramspace, h5, "$path/parameterspace")
end

function h5load(::Type{TrainingSet}, h5::H5DataStore; path::AbstractString="/")
    TrainingSet(h5, path)
end
