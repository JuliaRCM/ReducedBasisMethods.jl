using LinearAlgebra
using ReducedBasisMethods
using Test

@testset "ReducedBasisMethods.jl" begin
    include("multi_index_axis_tests.jl")
    include("multi_index_array_tests.jl")
    include("multi_index_lazy_array_tests.jl")
    
    include("parameter_tests.jl")
    include("parameterspace_tests.jl")
    include("trainingset_tests.jl")
    
    include("deim_test.jl")

    include("poisson_test.jl")
    include("bracket_test.jl")
end
