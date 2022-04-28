
using HDF5
using FastGaussQuadrature
using LaTeXStrings
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
ts = TrainingSet(fpath)
μₜ = ts.paramspace

# read integrator parameters
IP = ts.integrator

# create spline Poisson solver
poisson = ts.poisson

# read snapshot data
X = reshape(ts.snapshots.X, (IP.nₚ, IP.nₛ * IP.nparam))
V = reshape(ts.snapshots.V, (IP.nₚ, IP.nₛ * IP.nparam))
E = reshape(ts.snapshots.A, (IP.nₚ, IP.nₛ * IP.nparam))
# D = h5read(fpath, "snapshots/D")
# Φ = h5read(fpath, "snapshots/Phi")


# Reference draw
P₀ = ts.initconds


# EVD
Λₚ, Ωₚ, kₚ, Ψₚ = get_ΛΩ_particles(X, V, IP)
Λₑ, Ωₑ, kₑ, Ψₑ = get_ΛΩ_efield(E)

# Λₑₓₜ, Ωₑₓₜ, kₑₓₜ, Ψₑₓₜ = get_ΛΩe(Xₑₓₜ)

# Reduced Basis
rb = ReducedBasis(μₜ, EVD(), Λₚ, Ωₚ, kₚ, Ψₚ, Λₑ, Ωₑ, kₑ, Ψₑ)


# DEIM
@time Πₑ = deim_get_Π(Ψₑ)


# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5)
plot!(abs.(Λₚ)[1:1000], linewidth = 2, alpha = 0.25, label = L"$X$")
plot!(abs.(Λₑ)[1:1000], linewidth = 2, alpha = 0.5,  label = L"$F$")
savefig("../runs/$(runid)_SVDs_BoT_1.pdf")

# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5, legend = :none)
plot!(abs.(Λₚ), linewidth = 2, alpha = 0.25, label = L"$X_v$")
plot!(abs.(Λₑ), linewidth = 2, alpha = 0.5,  label = L"$E$")
savefig("../runs/$(runid)_SVDs_BoT_2.pdf")


# save to HDF5
h5save(ppath, IP, poisson, params, μₜ, kₚ, kₑ, Ψₚ, Ψₑ, Πₑ, P₀)
