module ReducedBasisMethods

    using Particles

    include("parameters.jl")

    export Parameter, hassamples
    export ParameterSampler, CartesianParameterSampler, sample
    export ParameterSpace

    include("regression.jl")

    export get_regression_αβ

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
