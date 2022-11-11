
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
vpath = "../runs/$(runid)_vectorfield.h5"


# Create reduced basis using EVD

# particle_tol = 1e-8
# field_tol = 1e-4
particle_tol = 1e-4
field_tol = 1e-2

ts = TrainingSet(fpath)
rb = ReducedBasis(CotangentLiftEVD(), ts; particle_tol = particle_tol, field_tol = field_tol)

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


# reduce electric field
X = ts.snapshots.X
A = ts.snapshots.A

X̃ = zeros(size(rb.Ψₚ, 2), size(X, 3), size(X, 4))
Ã = zeros(size(rb.Ψₚ, 2), size(A, 3), size(A, 4))

for p in axes(A, 4)
    for it in axes(A, 3)
        X̃[:,it,p] .= rb.Ψₚ' * vec(X[:,:,it,p])
        Ã[:,it,p] .= rb.Ψₚ' * vec(A[:,:,it,p])
    end
end

h5open(vpath, "w") do file
    file["X"] = X̃
    file["A"] = Ã
end


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
