module ReducedBasisMethods

    using HDF5
    using HDF5: H5DataStore
    using LinearAlgebra
    using Particles
    using TypedTables

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

    export ReducedBasis, EVD

    include("eigen.jl")

    export sorteigen

    include("algorithms/evd.jl")

    export get_Ψ, get_ΛΩ_particles, get_ΛΩ_efield

    include("algorithms/deim.jl")
    
    export deim_get_Π

    include("h5routines.jl")

    export h5save, h5load, read_sampling_parameters


end
