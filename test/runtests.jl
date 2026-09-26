using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Generic-only skeleton" include("integration/skeleton.jl")
    @safetestset "Reduced basis" include("reducedbasis.jl")
    @safetestset "Reduced tensor" include("reduced_tensor.jl")
    @safetestset "DEIM" include("algorithms/deim.jl")
end
