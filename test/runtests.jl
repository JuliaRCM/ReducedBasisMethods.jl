using ReducedBasisMethods, LinearAlgebra
using Test

@testset "ReducedBasisMethods.jl" begin
    include("parameter_tests.jl")
    include("parameterspace_tests.jl")
    include("trainingset_tests.jl")
    include("deim_test.jl")

    include("poisson_test.jl")
    include("bracket_test.jl")
end
