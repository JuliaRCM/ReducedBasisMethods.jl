module ReducedBasisMethods

    using Particles

    include("parameters.jl")

    export Parameter, ParameterSpace, ParameterSampler,
           parameter_grid

    include("regression.jl")

    export get_regression_αβ

    include("time_marching.jl")

    export IntegratorCache, IntegratorParameters, integrate_vp

    include("eigen.jl")

    export sorteigen

    include("projection.jl")

    export get_Ψ

    include("deim.jl")
    
    export deim_get_Π

    include("h5routines.jl")

    export h5save, read_sampling_parameters

end
