
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
fpath = "../runs/BoT_Np5e4_k_010_050_np_10_T25.h5"

# read sampling parameters
params = read_sampling_parameters(fpath)

# read training parameters
μₜ = h5read(fpath, "parameters/mu_train")

# read integrator parameters
IP = IntegratorParameters(fpath)

# create spline Poisson solver
poisson = PoissonSolverPBSplines(fpath)

# read snaptshot data
X = h5read(fpath, "snapshots/X")
V = h5read(fpath, "snapshots/V")
E = h5read(fpath, "snapshots/E")
# D = h5read(fpath, "snapshots/D")
# Φ = h5read(fpath, "snapshots/Phi")


# Reference draw
P₀ = ParticleList(X[:,1], V[:,1], ones(IP.nₚ) .* poisson.L ./ IP.nₚ)


# EVD

# XV = hcat(X, V);
XV = copy(X)
for p in 1:IP.nparam
    # XV = hcat(XV, V[:,1+(p-1)*IP.nₛ])
    global XV = hcat(XV, V[:,1+(p-1)*IP.nₛ]) #, E[:,p*IP.nₛ])
    #print(1+(p-1)*IP.nₛ, " ", p*IP.nₛ, "\n" )
end

# clear 
X, V = 0, 0
GC.gc()


# @time F = eigen(Xₑₓₜ' * Xₑₓₜ)
# Λ, Ω = sorteigen(F.values, F.vectors)
@time F = eigen(XV' * XV)
@time Λ, Ω = sorteigen(F.values, F.vectors)

@time Fₑ = eigen(E' * E)
@time Λₑ, Ωₑ = sorteigen(Fₑ.values, Fₑ.vectors)


# clear
F, Fₑ = 0, 0
GC.gc()


# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5)
plot!(abs.(Λ )[1:1000], linewidth = 2, alpha = 0.25, label = L"$X$")
plot!(abs.(Λₑ)[1:1000], linewidth = 2, alpha = 0.5,  label = L"$F$")
savefig("../runs/BoT_Np5e4_k_010_050_np_10_T25_SVDs_BoT_1.pdf")

# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5, legend = :none)
plot!(abs.(Λ ), linewidth = 2, alpha = 0.25, label = L"$X_v$")
plot!(abs.(Λₑ), linewidth = 2, alpha = 0.5,  label = L"$E$")
savefig("../runs/BoT_Np5e4_k_010_050_np_10_T25_SVDs_BoT_2.pdf")


# Projection Matrices
@time k, Ψ = get_Ψ(XV, Λ, Ω, 1e-9, 0)
@time kₑ, Ψₑ = get_Ψ(E, Λₑ, Ωₑ, 1e-5, 0)

# clear 
XV, E = 0, 0
Λ, Λₑ, Ω, Ωₑ = 0, 0, 0, 0
GC.gc()


# DEIM
@time Πₑ = deim_get_Π(Ψₑ)


# save to HDF5
h5save("../runs/BoT_Np5e4_k_010_050_np_10_T25_projections.h5", IP, poisson, params, μₜ, k, kₑ, Ψ, Ψₑ, Πₑ, P₀)
