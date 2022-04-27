
using HDF5
using FastGaussQuadrature
using LaTeXStrings
using LinearAlgebra
using Particles
using Plots
using Random
using ReducedBasisMethods
using Statistics


# HDF5 file containing training data
runid = "BoT_Np5e4_k_010_050_np_10_T25"
fpath = "../runs/$(runid).h5"
ppath = "../runs/$(runid)_projections.h5"


# read sampling parameters
params = read_sampling_parameters(fpath)

# read training parameters
μₜ = ParameterSpace(fpath, "parameterspace")

# read integrator parameters
IP = IntegratorParameters(fpath)

# create spline Poisson solver
poisson = PoissonSolverPBSplines(fpath)

# read snaptshot data
# X = h5read(fpath, "snapshots/X")
# V = h5read(fpath, "snapshots/V")
E = h5read(fpath, "snapshots/E")
# D = h5read(fpath, "snapshots/D")
# Φ = h5read(fpath, "snapshots/Phi")


# Reference draw
P₀ = ParticleList(X[:,1], V[:,1], ones(IP.nₚ) .* poisson.L ./ IP.nₚ)


# EVD

function get_ΛΩ(fpath, IP; tolerance = 1e-4, k = 0)
     X = h5read(fpath, "snapshots/X")
     V = h5read(fpath, "snapshots/V")

     XV = X
     for p in 1:IP.nparam
          # XV = hcat(XV, V[:,1+(p-1)*IP.nₛ])
          XV = hcat(XV, V[:,1+(p-1)*IP.nₛ]) #, E[:,p*IP.nₛ])
          #print(1+(p-1)*IP.nₛ, " ", p*IP.nₛ, "\n" )
     end

     # XV = hcat(X, V);

     @time F = eigen(XV' * XV)
     @time Λ, Ω = sorteigen(F.values, F.vectors)

     # Projection Matrices
     @time k, Ψ = get_Ψ(XV, Λ, Ω, tolerance, k)

     return Λ, Ω, k, Ψ
end

function get_ΛΩe(E; tolerance = 1e-4, k = 0)
     @time Fₑ = eigen(E' * E)
     @time Λₑ, Ωₑ = sorteigen(Fₑ.values, Fₑ.vectors)
     
     # Projection Matrices
     @time kₑ, Ψₑ = get_Ψ(E, Λₑ, Ωₑ, tolerance, k)

     return Λₑ, Ωₑ, kₑ, Ψₑ
end


Λ, Ω, k, Ψ = get_ΛΩ(fpath, IP)

Λₑ, Ωₑ, kₑ, Ψₑ = get_ΛΩe(h5read(fpath, "snapshots/E"))

# Λₑₓₜ, Ωₑₓₜ, kₑₓₜ, Ψₑₓₜ = get_ΛΩe(Xₑₓₜ)

# clear
# GC.gc()


# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5)
plot!(abs.(Λ )[1:1000], linewidth = 2, alpha = 0.25, label = L"$X$")
plot!(abs.(Λₑ)[1:1000], linewidth = 2, alpha = 0.5,  label = L"$F$")
savefig("../runs/$(runid)_SVDs_BoT_1.pdf")

# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5, legend = :none)
plot!(abs.(Λ ), linewidth = 2, alpha = 0.25, label = L"$X_v$")
plot!(abs.(Λₑ), linewidth = 2, alpha = 0.5,  label = L"$E$")
savefig("../runs/$(runid)_SVDs_BoT_2.pdf")


# DEIM
@time Πₑ = deim_get_Π(Ψₑ)


# save to HDF5
h5save(ppath, IP, poisson, params, μₜ, k, kₑ, Ψ, Ψₑ, Πₑ, P₀)
