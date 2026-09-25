using LinearAlgebra
using ReducedBasisMethods
using Test

@testset "ReducedBasisMethods.jl" begin
    include("skeleton_tests.jl")
    include("reducedbasis_tests.jl")

    include("deim_test.jl")
end
