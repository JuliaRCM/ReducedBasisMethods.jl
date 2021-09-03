
using HDF5

import Particles.PoissonSolverPBSplines


function save_snapshots(fpath::String, X::Matrix{T}, V::Matrix{T}, E::Matrix{T}, Φ::Matrix{T}) where {T}
    h5open(fpath, "w") do file
        s = create_group(file, "snapshots")
        s["X"] = X
        s["V"] = V
        s["E"] = E
        s["Φ"] = Φ
    end
end


function save_projections(fpath::String, k, kₑ, Ψ, Ψₑ, Πₑ, P₀) where {T}
    h5open(fpath, "w") do file
        g = create_group(file, "parameters") # create a group
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


"""
save sampling parameters
"""
function save_sampling_parameters(fpath::AbstractString, sampling_params::NamedTuple, μₜ::Matrix)
    h5open(fpath, "r+") do file
        if haskey(file, "parameters")
            g = file["parameters"]
        else
            g = create_group(file, "parameters")
        end

        g["κ"]  = sampling_params.κ
        g["ε"]  = sampling_params.ε
        g["a"]  = sampling_params.a
        g["v₀"] = sampling_params.v₀
        g["σ"]  = sampling_params.σ
        g["χ"]  = sampling_params.χ

        g["mu_train"] = μₜ
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
function h5save(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, μₜ::Matrix{T}, Result) where {T}
    # create file and save snapshots
    save_snapshots(fpath,
             Result.X,
             Result.V,
             Result.E,
             Result.Φ)
    
    h5save(fpath, P)
    h5save(fpath, IP)
    save_sampling_parameters(fpath, sampling_params, μₜ)
end



"""
save projection data
"""
function h5save(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, μₜ::Matrix{T}, k, kₑ, Ψ, Ψₑ, Πₑ, P₀) where {T}
    # create file and save projections
    save_projections(fpath, k, kₑ, Ψ, Ψₑ, Πₑ, P₀)
    
    h5save(fpath, P)
    h5save(fpath, IP)
    save_sampling_parameters(fpath, sampling_params, μₜ)
end
