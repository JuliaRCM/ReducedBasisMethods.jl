using Aqua
using ReducedBasisMethods
using Test

# Package-level quality assurance: type piracy, method ambiguities, stale and duplicated
# dependencies, undefined exports, unbound type parameters, and a `[compat]` entry for every
# entry of `[deps]`, `[weakdeps]` and `[extras]`. These are the faults the rest of the suite is
# structurally unable to see — it exercises behaviour, and every one of them is a property of the
# package as a whole.
#
# Check 4 asks for the call inside a `@testset`, so each check's own passes and failures carry a
# name and a line in the suite's summary instead of one anonymous failure in the run's output.
aqua = @testset "Aqua" begin
    Aqua.test_all(ReducedBasisMethods)
end

# The wrapper is not vacuous: every check `test_all` runs makes at least one `@test` of its own and
# carries it in a sub-testset, so this testset reports more than zero passed tests. Its summary
# covers `stale_deps` and `undefined_exports` among those passes, and a `CairoMakie` whose
# `[compat]` entry is missing is reported by `deps_compat` in the same summary.
@test Test.get_test_counts(aqua).cumulative_passes > 0
