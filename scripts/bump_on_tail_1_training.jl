module BumpOnTailSimulation

using Plots, LaTeXStrings
using ReducedBasisMethods
using Particles
using Particles.BumpOnTail
using Random


# HDF5 file to store training data
runid = "BoT_Np5e4_k_010_050_np_10_T25"
fpath = "../runs/$runid.h5"

# parameters
const dt = 1e-1             # timestep
const T = 25                # final time
const nt = Int(div(T, dt))  # nb. of timesteps
const np = Int(5e3)         # nb. of particles
const nh = 16               # nb. of elements
const p = 3                 # spline degree

const nₚ₁ = 10
const nₚ₂ = 1
const nₚ₃ = 1
const nₚ₄ = 1
const nₚ₅ = 1
const nparam = nₚ₁ * nₚ₂ * nₚ₃ * nₚ₄ * nₚ₅

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
κ = Parameter(:κ,  0.1,  0.5,  nₚ₁)
ε = Parameter(:ε,  0.03, 0.03, nₚ₂)
a = Parameter(:a,  0.1,  0.1,  nₚ₃)
v₀= Parameter(:v₀, 4.5,  4.5,  nₚ₄)
σ = Parameter(:σ,  0.5,  0.5,  nₚ₅)

const pspace = ParameterSpace(κ, ε, a, v₀, σ)

const L = 2π/params.κ         # domain length
# const h = L/nh                # element width


function run()

    # integrator parameters
    IP = VPIntegratorParameters(dt, nt, nt+1, nh, np)

    # integrator cache
    IC = VPIntegratorCache(IP)
    
    # B-spline Poisson solver
    poisson = PoissonSolverPBSplines(p, nh, L)

    # set random generator seed
    Random.seed!(1234)

    # initial data
    particles = BumpOnTail.draw_accept_reject(np, params)
    # particles = BumpOnTail.draw_importance_sampling(np, params)

    # training set
    TS = TrainingSet(poisson, particles, nt+1, pspace)

    # loop over parameter set
    for p in eachindex(pspace)

        # get parameter tuple
        lparams = merge(pspace(p), (χ = pspace[p].κ / params.κ,))

        # integrate particles for parameter
        integrate_vp!(particles, poisson, lparams, IP, IC; save=true, given_phi=false)

        # copy solution
        TS.X[1,:,:,p] .= IC.X
        TS.V[1,:,:,p] .= IC.V
        TS.A[1,:,:,p] .= IC.A
        TS.Φ[1,:,:,p] .= IC.Φ

        # copy diagnostics
        TS.W[:,p] .= IC.W
        TS.K[:,p] .= IC.K
        TS.M[:,p] .= IC.M
    end

    # save results to HDF5
    h5save(fpath, TS, IP, poisson, params)

    # plot
    plot(IP.t, TS.W, linewidth = 2, xlabel = L"$n_t$", yscale = :log10, legend = :none,
        grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
    savefig("../runs/$(runid)_plot1.pdf")
    # TODO: Change filename to something meaningful!

    #
    α, β = get_regression_αβ(IP.t, TS.W, 2)

    #
    Wₗᵢₙ = zero(TS.W)
    for i in axes(Wₗᵢₙ,2)
        Wₗᵢₙ[:,i] .= exp.(α[i] .+ β[i] .* IP.t)
    end

    # plot
    plot(xlabel = L"$n_t$", yscale = :log10, ylims = (1E-3,1E1), legend = :none,
        grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
    plot!(IP.t, TS.W[:,1:5], linewidth = 2, alpha = 0.25)
    plot!(IP.t, Wₗᵢₙ[:,1:5], linewidth = 2, alpha = 0.5)
    savefig("../runs/$(runid)_plot2.pdf")
    # TODO: Change filename to something meaningful!

    # plot
    plot(xlabel = L"$n_t$", yscale = :log10, ylims = (1E-3,1E1), legend = :none,
        grid = true, gridalpha = 0.5, minorgrid = true, minorgridalpha = 0.2)
    plot!(IP.t, TS.W[:,6:10], linewidth = 2, alpha = 0.25)
    plot!(IP.t, Wₗᵢₙ[:,6:10], linewidth = 2, alpha = 0.5)
    savefig("../runs/$(runid)_plot3.pdf")
    # TODO: Change filename to something meaningful!

end

end


using .BumpOnTailSimulation

BumpOnTailSimulation.run()
