module ReducedBasisMethods

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

    include("trainingset.jl")

    export TrainingSet

    include("time_marching.jl")

    export IntegratorParameters, IntegratorCache, ReducedIntegratorCache
    export integrate_vp, reduced_integrate_vp

    include("eigen.jl")

    export sorteigen

    include("projection.jl")

    export get_Ψ

    include("deim.jl")
    
    export deim_get_Π

    include("h5routines.jl")

    export h5save, read_sampling_parameters

end
