
using HDF5
using LaTeXStrings
using LinearAlgebra
using Plots
using Random
using ReducedBasisMethods
using Statistics


# HDF5 file containing training data
runid = "BoT_Np5e4_k_010_050_np_10_T25"
fpath = "../runs/$(runid).h5"
ppath = "../runs/$(runid)_projections.h5"


# Create reduced basis using EVD
ts = TrainingSet(fpath)
rb = ReducedBasis(CotangentLiftEVD(), ts)

println(" kₚ = ", rb.kₚ)
println(" kₑ = ", rb.kₑ)
println()

for p in eachindex(ts.paramspace)
    println("running parameter nb. $p with chi = ", ts.paramspace(p).χ)

    x = ts.snapshots.X[:,:,begin,p]
    x̃ = rb.Ψₚ * (rb.Ψₚ' * vec(x))
end


# save to HDF5
h5save(ppath, rb)

# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$", yscale = :log10, 
     grid = true, gridalpha = 0.5, xlim = (0,Inf))
plot!(abs.(rb.Λₚ ./ rb.Λₚ[begin])[1:minimum((100, length(rb.Λₚ)))], linewidth = 2, alpha = 0.25, label = L"$X$")
plot!(abs.(rb.Λₑ ./ rb.Λₑ[begin])[1:minimum((100, length(rb.Λₑ)))], linewidth = 2, alpha = 0.5,  label = L"$F$")
savefig("../runs/$(runid)_SVDs_BoT_1.pdf")

# plot
plot(xlabel = L"$i$", ylabel = L"$\lambda_i$",# yscale = :log10, 
     grid = true, gridalpha = 0.5, legend = :none)
plot!(abs.(rb.Λₚ ./ rb.Λₚ[begin]), linewidth = 2, alpha = 0.25, label = L"$X_v$")
plot!(abs.(rb.Λₑ ./ rb.Λₑ[begin]), linewidth = 2, alpha = 0.5,  label = L"$E$")
savefig("../runs/$(runid)_SVDs_BoT_2.pdf")
