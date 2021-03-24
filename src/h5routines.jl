
using HDF5


function save_h5(fpath::String, Nₚ::Int, nₕ::Int, p::Int, dt::T,
                 nₜ::Int, nₛ::Int, nₚ::Int, μₛ::Array{T}, μₜ::Array{T},
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
        g["mu_samp"] = μₛ
        g["mu_train"] = μₜ

        f = create_group(file, "snapshots")
        f["X"] = X
        f["V"] = V
        f["E"] = E
        f["Φ"] = Φ
    end
end

function save_h5(fpath::String, IP::IntegratorParameters, S::PBSpline{T}, μₛ::Array{T}, μₜ::Array{T}, Result) where {T}
    save_h5(fpath,
             IP.Nₚ,
             S.nₕ,
             S.p,
             IP.dt,
             IP.nₜ,
             IP.nₛ,
             IP.nₚ,
             μₛ,
             μₜ,
             Result.X,
             Result.V,
             Result.E,
             Result.Φ)
end
