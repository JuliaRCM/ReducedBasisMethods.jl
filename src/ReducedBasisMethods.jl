module ReducedBasisMethods

    using Particles

    include("parameters.jl")

    export Parameter, ParameterSpace, ParameterSampler,
           parameter_grid

    include("regression.jl")

    export get_regression_αβ

    include("time_marching.jl")

    export IntegratorCache, IntegratorParameters, integrate_vp

    include("h5routines.jl")

    export save_h5

end
