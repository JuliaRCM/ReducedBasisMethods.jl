using Aqua
using ReducedBasisMethods
using Test

# Aqua checks package-wide methods, exports, dependency usage and compatibility.
# Its checks use their own testsets; this wrapper groups their results under one
# suite heading.
aqua = @testset "Aqua" begin
    Aqua.test_all(ReducedBasisMethods)
end

# Require nonzero cumulative passes so a vacuous Aqua run fails.
@test Test.get_test_counts(aqua).cumulative_passes > 0
