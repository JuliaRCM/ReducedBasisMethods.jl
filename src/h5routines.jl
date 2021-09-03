
using HDF5


function save_h5(fpath::String, Nₚ::Int, nₕ::Int, p::Int, dt::T,
                 nₜ::Int, nₛ::Int, nₚ::Int, sampling_params::NamedTuple, μₜ::Array{T},
                 X::Matrix{T}, V::Matrix{T}, E::Matrix{T}, Φ::Matrix{T}) where {T}
    h5open(fpath, "w") do file
        g = create_group(file, "parameters") # create a group
        g["N_p"] = Nₚ
        g["n_h"] = nₕ
        g["p"] = p
        g["dt"] = dt
        g["n_t"] = nₜ
        g["n_s"] = nₛ
        g["n_p"] = nₚ

        g["κ"] = sampling_params.κ
        g["ε"] = sampling_params.ε
        g["a"] = sampling_params.a
        g["v₀"] = sampling_params.v₀
        g["σ"] = sampling_params.σ
        g["χ"] = sampling_params.χ

        g["mu_train"] = μₜ

        f = create_group(file, "snapshots")
        f["X"] = X
        f["V"] = V
        f["E"] = E
        f["Φ"] = Φ
    end
end

function save_h5(fpath::String, IP::IntegratorParameters, P::PoissonSolverPBSplines{T}, sampling_params::NamedTuple, μₜ::Array{T}, Result) where {T}
    save_h5(fpath,
             IP.nₚ,
             P.nx,
             P.p,
             IP.dt,
             IP.nₜ,
             IP.nₛ,
             IP.nparam,
             sampling_params,
             μₜ,
             Result.X,
             Result.V,
             Result.E,
             Result.Φ)
end
