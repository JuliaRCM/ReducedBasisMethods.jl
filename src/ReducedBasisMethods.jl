module ReducedBasisMethods

    using HDF5
    using HDF5: H5DataStore
    using LinearAlgebra
    using ParticleMethods
    using PoissonSolvers
    using TypedTables
    using LazyArrays
    using OffsetArrays

    include("utils.jl")

    include("parameter.jl")

    export Parameter, hassamples

    include("parametersampler.jl")

    export ParameterSampler, CartesianParameterSampler, sample

    include("parameterspace.jl")

    export ParameterSpace

    include("regression.jl")

    export get_regression_αβ

    include("particles/poisson.jl")

    include("particles/time_marching.jl")

    export IntegratorParameters, IntegratorCache, ReducedIntegratorCache
    export integrate_vp, reduced_integrate_vp

    include("particles/snapshots.jl")

    export Snapshots

    include("trainingset.jl")

    export TrainingSet

    include("reducedbasis.jl")

    export ReducedBasis, CotangentLiftEVD, CotangentLiftSVD

    include("eigen.jl")

    export sorteigen

    include("algorithms/evd.jl")

    export get_PODBasis_EVD, get_PODBasis_cotangentLiftEVD

    include("algorithms/deim.jl")
    
    export get_DEIM_interpolation_matrix

    include("h5routines.jl")

    export h5save, h5load, read_sampling_parameters

    

    include("gridbased/poisson.jl")

    export _apply_Δₓ!, _apply_Δₓ₄!, _apply_Rₓ!, _apply_∫dv!

    include("gridbased/bracket_operators.jl")

    export _apply_P_ϕ!, _apply_P_h!

    include("gridbased/bracket_tensors.jl")

    export PoissonTensor, PoissonOperator, PotentialReducedTensor, VelocityReducedMatrix, ReducedTensor, Arakawa

end
