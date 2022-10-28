
using HDF5
using LaTeXStrings
using LinearAlgebra
using PoissonSolvers
using ParticleMethods
using Plots
using Random
using ReducedBasisMethods
using SparseArrays
using Statistics
using VlasovMethods

using ReducedBasisMethods: read_parameters


function ReducedBasisMethods.IntegratorParameters(ip::VPIntegratorParameters, pspace::ParameterSpace)
    IntegratorParameters(ip.dt, ip.nₜ, ip.nₛ, ip.nₕ, ip.nₚ, length(pspace))
end
# TODO: Clean up (this does not belong here!)


# HDF5 file to store training data
runid = "BoT_Np5e4_k_010_050_np_10_T25"
fpath = "../runs/$(runid)_projections.h5"
fpath_test = "../runs/$(runid)_test.h5"
fpath_out = "../runs/$(runid)_reduced.h5"


h5open(fpath, "r") do file
    global paramspace = ParameterSpace(file, "parameterspace")

    global params = read_parameters(file, "parameters")

    global integrator = IntegratorParameters(file, "integrator")
    
    global poisson = PoissonSolverPBSplines(file)

    global rbasis = ReducedBasis(file)
    
    global particles = ParticleList(file, "initial_conditions")
end

# sampling parameters
nₜₑₛₜ = 10
# κₜₑₛₜ_ₘᵢₙ = 0.05
# κₜₑₛₜ_ₘₐₓ = 0.35
κₜₑₛₜ_ₘᵢₙ = 0.1
κₜₑₛₜ_ₘₐₓ = 0.5

# set random generator seed
Random.seed!(1234)

κₜₑₛₜ = rand(nₜₑₛₜ) .* (κₜₑₛₜ_ₘₐₓ - κₜₑₛₜ_ₘᵢₙ) .+ κₜₑₛₜ_ₘᵢₙ
κₜₑₛₜ = κₜₑₛₜ[sortperm(κₜₑₛₜ)]

χ = Parameter(:χ,  κₜₑₛₜ_ₘᵢₙ / params.κ,  κₜₑₛₜ_ₘₐₓ / params.κ,  κₜₑₛₜ ./ params.κ)
ε = Parameter(:ε,  0.03, 0.03, 1 )    # amplitude of spatial perturbation
a = Parameter(:a,  0.1,  0.1,  1 )    # fast particle share
v₀= Parameter(:v₀, 4.5,  4.5,  1 )    # velocity
σ = Parameter(:σ,  0.5,  0.5,  1 )    # temperature

# parameter space
pspace = ParameterSpace(χ, ε, a, v₀, σ)

# full model integrator
IP = VPIntegratorParameters(integrator.dt, integrator.nₜ, integrator.nₛ, integrator.nₕ, integrator.nₚ)
IC = VPIntegratorCache(IP)

# test set
TS = TrainingSet(particles, poisson, integrator.nₜ+1, params, pspace, IntegratorParameters(IP, pspace))
SS = TS.snapshots

# loop over parameter set
for p in eachindex(pspace)

    # get parameter tuple
    lparams = merge(pspace(p), params)

    # integrate particles for parameter
    integrate_vp!(particles, poisson, lparams, IP, IC; save=true, given_phi=false)

    # copy solution
    SS.X[1,:,:,p] .= IC.X
    SS.V[1,:,:,p] .= IC.V
    SS.A[1,:,:,p] .= IC.A
    SS.Φ[1,:,:,p] .= IC.Φ

    # copy diagnostics
    SS.W[:,p] .= IC.W
    SS.K[:,p] .= IC.K
    SS.M[:,p] .= IC.M
end


# save testset results to HDF5
h5open(fpath_test, "w") do file
    h5save(file, TS)
end


# Reduced Model
Ψₚ = rbasis.Ψₚ
Ψₑ = rbasis.Ψₑ
Πₑ = sparse(rbasis.Πₑ)
kₚ = size(Ψₚ)[2]
kₑ = size(Ψₑ)[2]

RIC = ReducedIntegratorCache(IntegratorParameters(IP, pspace), kₚ, kₑ)
RTS = TrainingSet(particles, poisson, integrator.nₜ+1, params, pspace, IntegratorParameters(IP, pspace))
RSS = RTS.snapshots

ΨₚᵀPₑ = Ψₚ' * Ψₑ * inv(Πₑ' * Ψₑ)
ΠₑᵀΨₚ = Πₑ' * Ψₚ

@time Rᵣₘ = reduced_integrate_vp(particles, Ψₚ, ΨₚᵀPₑ, ΠₑᵀΨₚ, pspace, params, poisson, RSS, IntegratorParameters(IP, pspace), RIC;
                                   DEIM=true, given_phi = false, save = true)

# save reduced basis results to HDF5
h5open(fpath_out, "w") do file
    h5save(file, RTS)
end


# plot
plot(xlabel = L"$n_t$", yscale = :log10, legend = :none,
    grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
plot!(IP.t, RSS.W[:,1:5], linewidth = 2, alpha = 1)
plot!(IP.t,  SS.W[:,1:5], linewidth = 2, alpha = 0.25)
savefig("../runs/$(runid)_reduced_plot2.pdf")
# TODO: Change filename to something meaningful!

# plot
plot(xlabel = L"$n_t$", yscale = :log10, legend = :none,
    grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
plot!(IP.t, RSS.W[:,6:10], linewidth = 2, alpha = 1)
plot!(IP.t,  SS.W[:,6:10], linewidth = 2, alpha = 0.25)
savefig("../runs/$(runid)_reduced_plot3.pdf")
# TODO: Change filename to something meaningful!


# println(norm(RSS.Φ - SS.Φ))
# println(norm(Rₜₑₛₜ.Φ - Φₜₑₛₜ))
