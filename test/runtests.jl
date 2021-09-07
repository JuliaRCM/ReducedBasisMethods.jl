using ReducedBasisMethods
using Test

@testset "ReducedBasisMethods.jl" begin
    include("parameter_tests.jl")
    include("parameterspace_tests.jl")
end
