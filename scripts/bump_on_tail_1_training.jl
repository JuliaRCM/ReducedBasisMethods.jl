module BumpOnTailSimulation

using LinearAlgebra: dot
using Plots, LaTeXStrings
using ReducedBasisMethods
using Particles
using Particles.BumpOnTail
using Random


# parameters
const dt = 1e-1             # timestep
const T = 25                # final time
const nt = Int(div(T, dt))  # nb. of timesteps
const np = Int(5e3)         # nb. of particles
const ns = 16               # nb. of elements
const p = 3                 # spline degree

const nₚ₁ = 10
const nₚ₂ = 1
const nₚ₃ = 1
const nₚ₄ = 1
const nₚ₅ = 1
const nparam = nₚ₁*nₚ₂*nₚ₃*nₚ₄*nₚ₅

const vmax = +10
const vmin = -10

# bump-on tail instability - reference values: κ, ε, a, v₀, σ
params = (κ = 0.3,      # spatial perturbation wave number
          ε = 0.03,     # amplitude of spatial perturbation
          a = 0.1,      # fast particle share
          v₀= 4.5,      # velocity
          σ = 0.5,      # temperature
          χ = 1.0)

# sampling parameters
κrange  = (0.1, 0.5)
εrange  = (0.03, 0.03)
arange  = (0.1, 0.1)
v₀range = (4.5, 4.5)
σrange  = (0.5, 0.5)

κ = LinRange(κrange[begin], κrange[end], nₚ₁)

const μ = zeros(nparam, 5)
for i in axes(μ, 1)
    μ[i,:] = [κ[i], params.ε, params.a, params.v₀, params.σ]
end

const L = 2π/params.κ         # domain length
const h = L/ns                # element width
const χ = μ[:,1] ./ params.κ


function run()

    # integrator parameters
    IP = IntegratorParameters(dt, nt, nt+1, ns, np, nparam)

    # integrator cache
    IC = IntegratorCache(IP)
    
    # B-spline Poisson solver
    poisson = PoissonSolverPBSplines(p, IP.nₕ, L)

    # set random generator seed
    Random.seed!(1234)

    # initial data
    P = BumpOnTail.draw_accept_reject(np, params)
    # P = BumpOnTail.draw_importance_sampling(np, params)

    # integrate particles for all parameters
    Result = ReducedBasisMethods.integrate_vp(P, μ, params, poisson, IP, IC; save=true, given_phi=false)

    # save results to HDF5
    save_h5("../runs/BoT_Np5e4_k_010_050_np_10_T25.h5", IP, poisson, params, μ, Result)

    #
    W = zero(Result.Φ[1,:]);

    for i in eachindex(W)
        W[i] = 0.5 * dot(Result.Φ[:,i], poisson.M, Result.Φ[:,i])
    end

    W = reshape(W, (IP.nₛ, IP.nparam))

    for p in axes(W,2)
        W[:,p] .*= χ[p]^2
    end

    # plot
    plot(IP.t, W[:,:], linewidth = 2, xlabel = L"$n_t$", yscale = :log10, legend = :none,
        grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
    savefig("../runs/BoT_Np5e4_k_010_050_np_10_T25_plot1.png")
    # TODO: Change filename to something meaningful!

    #
    α, β = get_regression_αβ(IP.t, W, 2)

    #
    Wₗᵢₙ = zero(W)
    for i in axes(Wₗᵢₙ,2)
        Wₗᵢₙ[:,i] .= exp.(α[i] .+ β[i] .* IP.t)
    end

    # plot
    plot(xlabel = L"$n_t$", yscale = :log10, ylims = (1E-3,1E1), legend = :none,
        grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
    plot!(IP.t, W[:,1:5], linewidth = 2, alpha = 0.25)
    plot!(IP.t, Wₗᵢₙ[:,1:5], linewidth = 2, alpha = 0.5)
    savefig("../runs/BoT_Np5e4_k_010_050_np_10_T25_plot2.png")
    # TODO: Change filename to something meaningful!

    # plot
    plot(xlabel = L"$n_t$", yscale = :log10, ylims = (1E-3,1E1), legend = :none,
        grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
    plot!(IP.t, W[:,6:10], linewidth = 2, alpha = 0.25)
    plot!(IP.t, Wₗᵢₙ[:,6:10], linewidth = 2, alpha = 0.5)
    savefig("../runs/BoT_Np5e4_k_010_050_np_10_T25_plot3.png")
    # TODO: Change filename to something meaningful!

end

end


using .BumpOnTailSimulation

BumpOnTailSimulation.run()
