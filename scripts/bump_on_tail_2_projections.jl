
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


# Create reduced basis using EVD
rb = ReducedBasis(EVD(), TrainingSet(fpath))

# DEIM
@time Πₑ = deim_get_Π(rb.Ψₑ)

# save to HDF5
h5save(ppath, rb)


# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5)
plot!(abs.(rb.Λₚ)[1:1000], linewidth = 2, alpha = 0.25, label = L"$X$")
plot!(abs.(rb.Λₑ)[1:1000], linewidth = 2, alpha = 0.5,  label = L"$F$")
savefig("../runs/$(runid)_SVDs_BoT_1.pdf")

# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5, legend = :none)
plot!(abs.(rb.Λₚ), linewidth = 2, alpha = 0.25, label = L"$X_v$")
plot!(abs.(rb.Λₑ), linewidth = 2, alpha = 0.5,  label = L"$E$")
savefig("../runs/$(runid)_SVDs_BoT_2.pdf")
