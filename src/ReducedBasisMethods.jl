module ReducedBasisMethods

    using HDF5
    using HDF5: H5DataStore
    using LinearAlgebra
    using Particles
    using TypedTables
    using LazyArrays

    include("utils.jl")

    include("parameter.jl")

    export Parameter, hassamples

    include("parametersampler.jl")

    export ParameterSampler, CartesianParameterSampler, sample

    include("parameterspace.jl")

    export ParameterSpace

    include("poisson.jl")

    include("regression.jl")

    export get_regression_αβ

    include("time_marching.jl")

    export IntegratorParameters, IntegratorCache, ReducedIntegratorCache
    export integrate_vp, reduced_integrate_vp

    include("snapshots.jl")

    export Snapshots

    include("trainingset.jl")

    export TrainingSet

    include("reducedbasis.jl")

    export ReducedBasis, EVD

    include("eigen.jl")

    export sorteigen

    include("algorithms/evd.jl")

    export get_PODBasis_EVD, get_PODBasis_cotangentLiftEVD

    include("algorithms/deim.jl")
    
    export get_DEIM_interpolation_matrix

    include("h5routines.jl")

    export h5save, h5load, read_sampling_parameters


end
