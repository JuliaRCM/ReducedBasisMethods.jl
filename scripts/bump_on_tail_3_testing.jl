
using HDF5
using FastGaussQuadrature
using LaTeXStrings
using LinearAlgebra
using Particles
using Random
using ReducedBasisMethods
using SparseArrays
using Statistics


fpath = "../runs/BoT_Np5e4_k_010_050_np_10_T25_projections.h5"

params = read_sampling_parameters(fpath)

μₜᵣₐᵢₙ = h5read(fpath, "parameters/mu_train")

IP = IntegratorParameters(fpath)

poisson = PoissonSolverPBSplines(fpath)

Ψ = h5read(fpath, "projections/Psi")
Ψₑ = h5read(fpath, "projections/Psi_e")
Πₑ = sparse(h5read(fpath, "projections/Pi_e"))


# Reference draw
P₀ = ParticleList(h5read(fpath, "initial_condition/x_0"),
                  h5read(fpath, "initial_condition/v_0"),
                  h5read(fpath, "initial_condition/w") )


nₜₑₛₜ = 10
κₜₑₛₜ_ₘᵢₙ = 0.1
κₜₑₛₜ_ₘₐₓ = 0.5

μₜₑₛₜ = zeros(nₜₑₛₜ, 5)
for i in 1:nₜₑₛₜ
    μₜₑₛₜ[i,:] = [κₜₑₛₜ_ₘᵢₙ, params.ε, params.a, params.v₀, params.σ]
end

λ = 0
for i in 1:nₜₑₛₜ
    if nₜₑₛₜ > 1
        μₜₑₛₜ[i,1] = rand(1)[1]*(κₜₑₛₜ_ₘₐₓ - κₜₑₛₜ_ₘᵢₙ) + κₜₑₛₜ_ₘᵢₙ
#         μₜₑₛₜ[i,1] = (1-λ)*κₜₑₛₜ_ₘᵢₙ + λ*κₜₑₛₜ_ₘₐₓ
#         λ += 1/(nₜₑₛₜ-1)
    end
end  

μₜₑₛₜ = μₜₑₛₜ[sortperm(μₜₑₛₜ[:, 1]), :]


GC.gc()


IPₜₑₛₜ = IntegratorParameters(IP.dt, IP.nₜ, IP.nₜ+1, IP.nₕ, IP.nₚ, nₜₑₛₜ)
ICₜₑₛₜ = IntegratorCache(IPₜₑₛₜ)


@time Rₜₑₛₜ = ReducedBasisMethods.integrate_vp(P₀, μₜₑₛₜ, params, poisson, IPₜₑₛₜ, ICₜₑₛₜ;
                                              given_phi = false, save = true)
# Xₜₑₛₜ = Rₜₑₛₜ.X
# Vₜₑₛₜ = Rₜₑₛₜ.V
# Φₜₑₛₜ = Rₜₑₛₜ.Φ;


Φₜₑₛₜ = copy(Rₜₑₛₜ.Φ)


# no saving
@time ReducedBasisMethods.integrate_vp(P₀, μₜₑₛₜ, params, poisson, IPₜₑₛₜ, ICₜₑₛₜ;
                                        given_phi = false, save = false)


# Reduced Model
k = size(Ψ)[2]
kₑ = size(Ψₑ)[2]

RIC = ReducedIntegratorCache(IPₜₑₛₜ, k, kₑ)

ΨᵀPₑ = Ψ' * Ψₑ * inv(Πₑ' * Ψₑ)
ΠₑᵀΨ = Πₑ' * Ψ

@time Rᵣₘ = reduced_integrate_vp(P₀, Ψ, ΨᵀPₑ, ΠₑᵀΨ, μₜₑₛₜ, params, poisson, IPₜₑₛₜ, RIC;
                                   DEIM=true, given_phi = false, save = true)
# Xᵣₘ = Ψ * Rᵣₘ.Zₓ
# Vᵣₘ = Ψ * Rᵣₘ.Zᵥ
# Φᵣₘ = Rᵣₘ.Φ;


# no saving
@time reduced_integrate_vp(P₀, Ψ, ΨᵀPₑ, ΠₑᵀΨ, μₜₑₛₜ, params, poisson, IPₜₑₛₜ, RIC;
                            DEIM=true, given_phi = false, save=false)


# Saving
h5save("../runs/BoT_Np5e4_k_010_050_np_10_T25_DEIM.h5", IPₜₑₛₜ, poisson, params, μₜᵣₐᵢₙ, μₜₑₛₜ, Rₜₑₛₜ, Rᵣₘ, Ψ);


println(norm(Rᵣₘ.Φ - Φₜₑₛₜ))
println(norm(Rₜₑₛₜ.Φ - Φₜₑₛₜ))
