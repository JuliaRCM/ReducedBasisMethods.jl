using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "Generic-only skeleton" include("integration/skeleton.jl")
    @safetestset "Reference spectra script" include("integration/reference_spectra.jl")
    @safetestset "Reduced basis" include("reducedbasis.jl")
    @safetestset "Reduced tensor" include("reduced_tensor.jl")
    @safetestset "SVD routes" include("algorithms/svd.jl")
    @safetestset "EVD routes" include("algorithms/evd.jl")
    @safetestset "Cross-route checks" include("algorithms/routes.jl")
    @safetestset "Projections" include("projection.jl")
    @safetestset "DEIM" include("algorithms/deim.jl")
    @safetestset "Symplectic DEIM" include("algorithms/symplectic_deim.jl")
end
