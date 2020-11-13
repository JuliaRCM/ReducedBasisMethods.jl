module VlasovPoissonROM

using LinearAlgebra, Plots, Sobol, FastGaussQuadrature, SpecialFunctions

include("bump_on_tail_distributions.jl")
include("splines.jl")
include("sampling.jl")
include("poisson_solver_splines.jl")
include("visualisation.jl")
include("postprocessing.jl")

# parameters

# bump-on tail instability - reference values: κ, ε, a, v₀, σ
# in order: spartial perturbation wave number and amplitude of s.p.,fast particle share, speed, and temperature
const μ = [0.5, 0.5, 0.0, 4.0, 0.5]

const dt = 1e-1             # timestep
const T = 50                # final time
const nₜ = Int(div(T, dt))  # nb. of timesteps

const L = 2.0*pi/μ[1]       # domain length
const n = 32                 # nb. of elements
const h = L/n               # element width
const p = 2                 # spline degree

const Nₚ = Int(5e4)         # nb. of particles

const vmax = 10
const vmin = -10

function run()

    # assembly
    M = massmatrix_BSpl(p,h,n);
    S = stiffnessmatrix_BSpl(p,h,n);
    S_aug = zeros(n,n); S_aug .= S
    # solubility
    S_aug[n,:] .= ones(n);

    # initial data
    P = draw_g_bumpontail(Nₚ, fₓ, μ)

    # background density
    rhs = ones(n) * h - rhs_particles_BSBasis(P, p, h, n)

    # solubility
    rhs_aug = zeros(n); rhs_aug .= rhs; rhs_aug[n] = 0.0

    # initial potential
    phi = S_aug\rhs_aug

    # diagnostics
    W = zeros(nₜ + 1); K = zeros(nₜ + 1); Mom = zeros(nₜ + 1)
    W[1] = 0.5*dot(phi,S*phi)
    K[1] = 0.5*sum(P.w .* P.v .* P.v)
    Mom[1] = sum(P.w .* P.v)

    F = plot_particles(collect(range(0,stop=L,length=1000)),
                    collect(range(vmin,stop=vmax,length=1000)), P, L)

    for t = 1:nₜ
        # integration step
        P, phi = integrate_vlasovpoisson(P, dt, h, n, S_aug)

        W[t+1] = 0.5*dot(phi,S*phi)
        K[t+1] = 0.5*dot(P.w .* P.v, P.v)
        Mom[t+1] = dot(P.w, P.v)

        if t % div(nₜ,100) == 0
            F = plot_particles(collect(range(0,stop=L,length=1000)),
                            collect(range(vmin,stop=vmax,length=1000)), P, L)
        end
    end
    display(plot(0:dt:dt*nₜ, W, yaxis = :log))

end

run()

end
