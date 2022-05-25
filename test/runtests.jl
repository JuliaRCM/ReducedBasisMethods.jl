using ReducedBasisMethods
using Test

@testset "ReducedBasisMethods.jl" begin
    include("parameter_tests.jl")
    include("parameterspace_tests.jl")
    include("trainingset_tests.jl")
    include("deim_test.jl")
end
