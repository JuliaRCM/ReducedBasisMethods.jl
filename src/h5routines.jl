
using HDF5

import PoissonSolvers.PoissonSolverPBSplines


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
save fixed parameters
"""
function save_parameters(h5::H5DataStore, params::NamedTuple; path::AbstractString = "/")
    g = _create_group(h5, path)

    for key in keys(params)
        g[string(key)] = params[key]
    end
end

function save_parameters(fpath::AbstractString, params::NamedTuple)
    h5open(fpath, "r+") do file
        save_parameters(file, params; path = "parameters")
    end
end

"""
read fixed parameters
"""
function read_parameters(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]

    paramkeys = Tuple(Symbol.(keys(group)))
    paramvals = Tuple(read(group[key]) for key in keys(group))

    NamedTuple{paramkeys}(paramvals)
end

function read_parameters(fpath::AbstractString)
    h5open(fpath, "r") do file
        read_parameters(file; path = "parameters")
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



function h5save(fpath::AbstractString, data, args...; mode="w", kwargs...)
    h5open(fpath, mode) do file
        h5save(file, data, args...; kwargs...)
    end
end

function h5load(T::Type, fpath::AbstractString, args...; mode="r", kwargs...)
    h5open(fpath, mode) do file
        h5load(T, file, args...; kwargs...)
    end
end
