using LinearAlgebra
using ReducedBasisMethods
using Test
using Random

@testset "ReducedBasisMethods.jl" begin
    include("parameter_tests.jl")
    include("parameterspace_tests.jl")
    include("trainingset_tests.jl")
    include("reducedbasis_tests.jl")
    
    include("deim_test.jl")

    include("poisson_test.jl")
    include("bracket_operators_test.jl")
    #include("tensors_test.jl")
end
