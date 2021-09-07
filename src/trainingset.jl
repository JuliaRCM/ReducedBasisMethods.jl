
struct TrainingSet{DT <: Number, PS <: ParameterSpace}
    paramspace::PS

    X::Array{DT,4}
    V::Array{DT,4}
    A::Array{DT,4}
    Φ::Array{DT,4}
    W::Array{DT,2}
    K::Array{DT,2}
    M::Array{DT,2}

    function TrainingSet{DT}(ns, nd, np, nh, pspace::PS) where {DT,PS}
        X = zeros(nd, np, ns, length(pspace))
        V = zeros(nd, np, ns, length(pspace))
        A = zeros(nd, np, ns, length(pspace))
        Φ = zeros(nd, nh, ns, length(pspace))
        W = zeros(ns, length(pspace))
        K = zeros(ns, length(pspace))
        M = zeros(ns, length(pspace))
        new{DT,PS}(pspace, X, V, A, Φ, W, K, M)
    end
end


function TrainingSet(ns::Int, poisson::PoissonSolver{DT}, particles::ParticleList{DT}, pspace::ParameterSpace) where {DT}
    TrainingSet{DT}(ns, 1, length(particles), length(poisson), pspace)
end
