
import PoissonSolvers.PoissonSolverPBSplines

import ReducedComplexityModeling: _create_group


function save_tests(fpath::String, Rtest, Rrm, Ψ)
    h5open(fpath, "w") do file
        s = create_group(file, "tests")
        s["X_test"] = Rtest.X
        s["V_test"] = Rtest.V
        s["Phi_test"] = Rtest.Φ
        s["X_rm"] = Ψ * Rrm.Zₓ
        s["V_rm"] = Ψ * Rrm.Zᵥ
        s["Phi_rm"] = Rrm.Φ
    end
end


"""
save testing parameters
"""
function save_testing_parameters(fpath::AbstractString, μₜ::Matrix)
    h5open(fpath, "r+") do file
        g = _create_group(file, "parameters")
        g["mu_test"] = μₜ
    end
end


"""
save testing data
"""
function h5save(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, μtrain::Matrix{T}, μtest::Matrix{T}, Rtest, Rrm, Ψ) where {T}
    # create file and save test data
    save_tests(fpath, Rtest, Rrm, Ψ)
    h5save(fpath, P)
    h5save(fpath, IP)
    save_parameters(fpath, sampling_params; path = "parameters")
    save_testing_parameters(fpath, μtest)
end
