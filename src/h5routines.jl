function save_h5(fpath::String, Nₚ::Int, S::PBSpline{T}, dt::T, nₜ::Int, μₛ::Array{T}, μₜ::Array{T},
                    X::Matrix{T}, V::Matrix{T}, E::Matrix{T}, D::Matrix{T}, Φ::Matrix{T}) where {T}
    h5open(fpath, "w") do file
        g = g_create(file, "parameters") # create a group
        g["N_p"] = Nₚ
        g["n_h"] = nₕ
        g["p"] = S_degree
        g["dt"] = dt
        g["n_t"] = nₜ
        g["n_p"] = nₚ
        g["mu_samp"] = μₛₐₘₚ
        g["mu_train"] = μ
        f = g_create(file, "snapshots")
        f["X"] = X
        f["V"] = V
        f["E"] = E
        f["D"] = D
        f["Phi"] = Φ
    end
end
