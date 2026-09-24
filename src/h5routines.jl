
import ReducedComplexityModeling: _create_group

"""
save testing parameters
"""
function save_testing_parameters(fpath::AbstractString, μₜ::Matrix)
    h5open(fpath, "r+") do file
        g = _create_group(file, "parameters")
        g["mu_test"] = μₜ
    end
end
