
using HDF5

import Particles.PoissonSolverPBSplines


function _create_parameter_group(file)
    if haskey(file, "parameters")
        g = file["parameters"]
    else
        g = create_group(file, "parameters")
    end
    return g
end


function save_projections(fpath::String, k, kₑ, Ψ, Ψₑ, Πₑ, P₀) where {T}
    h5open(fpath, "w") do file
        g = _create_parameter_group(file)
        g["k"] = k
        g["k_e"] = kₑ
            
        f = create_group(file, "projections")
        f["Psi"] = Ψ
        f["Psi_e"] = Ψₑ
        f["Pi_e"] = Πₑ
        
        h = create_group(file, "initial_condition")
        h["x_0"] = P₀.x
        h["v_0"] = P₀.v
        h["w"] = P₀.w
    end
end


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
save sampling parameters
"""
function save_sampling_parameters(fpath::AbstractString, sampling_params::NamedTuple)
    h5open(fpath, "r+") do file
        g = _create_parameter_group(file)
        g["κ"]  = sampling_params.κ
        g["ε"]  = sampling_params.ε
        g["a"]  = sampling_params.a
        g["v₀"] = sampling_params.v₀
        g["σ"]  = sampling_params.σ
        g["χ"]  = sampling_params.χ
    end
end

function read_sampling_parameters(fpath::AbstractString)
    h5open(fpath, "r") do file
        (
            κ = read(file["parameters/κ"]),
            ε = read(file["parameters/ε"]),
            a = read(file["parameters/a"]),
            v₀= read(file["parameters/v₀"]),
            σ = read(file["parameters/σ"]),
            χ = read(file["parameters/χ"]),
        )
    end
end


"""
save testing parameters
"""
function save_testing_parameters(fpath::AbstractString, μₜ::Matrix)
    h5open(fpath, "r+") do file
        g = _create_parameter_group(file)
        g["mu_test"] = μₜ
    end
end


"""
save spline solver parameters
"""
function h5save(fpath::String, P::PoissonSolverPBSplines)
    h5open(fpath, "r+") do file
        attributes(file)["p"] = P.p
    end
end

function Particles.PoissonSolverPBSplines(fpath::String)
    h5open(fpath, "r") do file
        p = read(attributes(file)["p"])
        n = read(attributes(file)["nh"])
        κ = read(file["parameters/κ"])
        PoissonSolverPBSplines(p, n, 2π/κ)
    end
end


"""
save training data
"""
function h5save(fpath::String, TS::TrainingSet, IP::VPIntegratorParameters, poisson::PoissonSolverPBSplines, pspace::ParameterSpace, sampling_params::NamedTuple)
    # create file and save snapshots
    h5open(fpath, "w") do file
    end

    h5save(fpath, TS)
    h5save(fpath, pspace)
    h5save(fpath, poisson)
    h5save(fpath, IP)
    save_sampling_parameters(fpath, sampling_params)
end


"""
save projection data
"""
function h5save(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, μtrain::Matrix{T}, k, kₑ, Ψ, Ψₑ, Πₑ, P₀) where {T}
    # create file and save projections
    save_projections(fpath, k, kₑ, Ψ, Ψₑ, Πₑ, P₀)
    h5save(fpath, P)
    h5save(fpath, IP)
    save_sampling_parameters(fpath, sampling_params)
    save_training_parameters(fpath, μtrain)
end


"""
save testing data
"""
function h5save(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, μtrain::Matrix{T}, μtest::Matrix{T}, Rtest, Rrm, Ψ) where {T}
    # create file and save test data
    save_tests(fpath, Rtest, Rrm, Ψ)
    h5save(fpath, P)
    h5save(fpath, IP)
    save_sampling_parameters(fpath, sampling_params)
    save_training_parameters(fpath, μtrain)
    save_testing_parameters(fpath, μtest)
end
