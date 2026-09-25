using LinearAlgebra
using ReducedBasisMethods
using Test
using Random

@testset "ReducedBasisMethods.jl" begin
    include("skeleton_tests.jl")
    include("reducedbasis_tests.jl")

    include("deim_test.jl")

    #include("tensors_test.jl")
end
