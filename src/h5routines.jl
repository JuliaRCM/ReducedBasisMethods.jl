
using HDF5

import Particles.PoissonSolverPBSplines


function _name(h5::H5DataStore)
    name = HDF5.name(h5)
    name = name[findlast(isequal('/'), name)+1:end]
end


function _create_group(h5::H5DataStore, name)
    if haskey(h5, name)
        g = h5[name]
    else
        g = create_group(h5, name)
    end
    return g
end

function save_projections(h5::H5DataStore, k, kₑ, Ψ, Ψₑ, Πₑ, P₀; path::AbstractString = "/")
    group = _create_group(h5, path)

    g = _create_group(group, "parameters")
    g["k"] = k
    g["k_e"] = kₑ
        
    f = create_group(group, "projections")
    f["Psi"] = Ψ
    f["Psi_e"] = Ψₑ
    f["Pi_e"] = Πₑ
    
    h = create_group(group, "initial_condition")
    h["x_0"] = P₀.x
    h["v_0"] = P₀.v
    h["w"] = P₀.w
end

function save_projections(fpath::String, k, kₑ, Ψ, Ψₑ, Πₑ, P₀) where {T}
    h5open(fpath, "w") do file
        save_projections(file, k, kₑ, Ψ, Ψₑ, Πₑ, P₀)
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
function save_sampling_parameters(h5::H5DataStore, sampling_params::NamedTuple; path::AbstractString = "/")
    g = _create_group(h5, path)
    g["κ"]  = sampling_params.κ
    g["ε"]  = sampling_params.ε
    g["a"]  = sampling_params.a
    g["v₀"] = sampling_params.v₀
    g["σ"]  = sampling_params.σ
    g["χ"]  = sampling_params.χ
end

function save_sampling_parameters(fpath::AbstractString, sampling_params::NamedTuple)
    h5open(fpath, "r+") do file
        save_sampling_parameters(file, sampling_params; path = "parameters")
    end
end

function read_sampling_parameters(h5::H5DataStore; path::AbstractString = "/")
    group = h5[path]
    (
        κ = read(group["κ"]),
        ε = read(group["ε"]),
        a = read(group["a"]),
        v₀= read(group["v₀"]),
        σ = read(group["σ"]),
        χ = read(group["χ"]),
    )
end

function read_sampling_parameters(fpath::AbstractString)
    h5open(fpath, "r") do file
        read_sampling_parameters(file; path = "parameters")
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
save spline solver parameters
"""
function h5save(h5::H5DataStore, P::PoissonSolverPBSplines)
    attributes(h5)["p"] = P.p
end

function h5save(fpath::String, P::PoissonSolverPBSplines)
    h5open(fpath, "r+") do file
        h5save(file, P)
    end
end

function Particles.PoissonSolverPBSplines(h5::H5DataStore)
    p = read(attributes(h5)["p"])
    n = read(attributes(h5)["nh"])
    κ = read(h5["parameters/κ"])
    PoissonSolverPBSplines(p, n, 2π/κ)
end

function Particles.PoissonSolverPBSplines(fpath::AbstractString)
    h5open(fpath, "r") do file
        PoissonSolverPBSplines(file)
    end
end

"""
save training data
"""
function h5save(fpath::String, TS::TrainingSet, IP::VPIntegratorParameters, poisson::PoissonSolverPBSplines, sampling_params::NamedTuple)
    # create file and save snapshots
    h5open(fpath, "w") do file
        h5save(file, TS; path = "snapshots")
        h5save(file, IP, length(TS.paramspace))
        h5save(file, poisson)
        save_sampling_parameters(file, sampling_params; path = "parameters")
    end
end


"""
save projection data
"""
function h5save(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, training_params::ParameterSpace, k, kₑ, Ψ, Ψₑ, Πₑ, P₀) where {T}
        # create file and save projections
    h5open(fpath, "w") do file
        save_projections(file, k, kₑ, Ψ, Ψₑ, Πₑ, P₀)
        h5save(file, P)
        h5save(file, IP)
        h5save(file, training_params; path = "parameterspace")
        save_sampling_parameters(file, sampling_params; path = "parameters")
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
    save_sampling_parameters(fpath, sampling_params; path = "parameters")
    save_training_parameters(fpath, μtrain)
    save_testing_parameters(fpath, μtest)
end



function h5save(data, fpath::String, args...; mode="r+", kwargs...)
    h5open(fpath, mode) do file
        h5save(data, file, args...; kwargs...)
    end
end


function h5load(T::Type, fpath::AbstractString, args...; mode="r", kwargs...)
    h5open(fpath, mode) do file
        h5load(T, file, args...; kwargs...)
    end
end
