
using LinearAlgebra: lu, ldiv!
using PoissonSolvers: PBSpline, stiffnessmatrix, eval_deriv_PBSBasis, rhs_particles_PBSBasis

struct IntegratorParameters{T}
    dt::T          # time step
    nₜ::Int        # number of time steps
    nₛ::Int        # number of saved time steps
    nₕ::Int        # number of basis functions
    nₚ::Int        # number of particles
    nparam::Int    # number of equation parameters sampled

    t::Vector{T}

    function IntegratorParameters(dt::T, nₜ::Int, nₛ::Int, nₕ::Int, nₚ::Int, nparam::Int) where {T}
        t = collect(range(0, stop=dt*nₜ, length=nₛ))
        new{T}(dt,nₜ,nₛ,nₕ,nₚ,nparam,t)
    end
end

# function IntegratorParameters(ip::VPIntegratorParameters, pspace::ParameterSpace)
#     IntegratorParameters(ip.dt, ip.nₜ, ip.nₛ, ip.nₕ, ip.nₚ, length(pspace))
# end

function IntegratorParameters(h5::H5DataStore, path::AbstractString = "/")
    group = h5[path]
    IntegratorParameters(
        read(attributes(group)["dt"]),
        read(attributes(group)["nt"]),
        read(attributes(group)["ns"]),
        read(attributes(group)["nh"]),
        read(attributes(group)["np"]),
        read(attributes(group)["nparam"]),
    )
end

function IntegratorParameters(fpath::AbstractString, path::AbstractString = "/")
    h5open(fpath, "r") do file
        IntegratorParameters(file, path)
    end
end

"""
save integrator parameters
"""
function h5save(h5::H5DataStore, IP::IntegratorParameters; path = "/")
    g = _create_group(h5, path)
    attributes(g)["dt"] = IP.dt
    attributes(g)["nt"] = IP.nₜ
    attributes(g)["ns"] = IP.nₛ
    attributes(g)["nh"] = IP.nₕ
    attributes(g)["np"] = IP.nₚ
    attributes(g)["nparam"] = IP.nparam
end

mutable struct ReducedIntegratorCache{T}
    zₓ::Vector{T}
    zᵥ::Vector{T}

    x::Vector{T}
    xₑ::Vector{T}

    ρ::Vector{T}
    ρ₀::Vector{T}
    ϕ::Vector{T}
    rhs::Vector{T}

    Zₓ::Matrix{T}
    Zᵥ::Matrix{T}
    Φ::Matrix{T}
end

ReducedIntegratorCache(IP::IntegratorParameters{T},k::Int,kₑ::Int) where {T} = ReducedIntegratorCache(zeros(T,k), # zₓ
                                                                                zeros(T,k), # zᵥ
                                                                                zeros(T,IP.nₚ), # x
                                                                                zeros(T,kₑ), # xₑ
                                                                                zeros(T,IP.nₕ), # ρ
                                                                                zeros(T,IP.nₕ), # ρ₀
                                                                                zeros(T,IP.nₕ), # ϕ
                                                                                zeros(T,IP.nₕ), # rhs
                                                                                zeros(T,k,IP.nparam*IP.nₛ), # Zₓ
                                                                                zeros(T,k,IP.nparam*IP.nₛ), # Zᵥ
                                                                                zeros(T,IP.nₕ,IP.nparam*IP.nₛ) # Φ
                                                                                )

function reduced_integrate_vp(P₀::ParticleList{T},
                              Ψ::Array{T},
                              ΨᵀPₑ::Array{T},      # Ψ' * Ψₑ * inv(Πₑ' * Ψₑ)
                              ΠₑᵀΨ::Array{T},    # Πₑ' * Ψ
                              μ::Array{T},
                              params::NamedTuple,
                              P::PoissonSolverPBSplines{T},
                              IP::IntegratorParameters{T},
                              IC::ReducedIntegratorCache{T} = ReducedIntegratorCache(IP,size(Pₑ)[1],size(Pₑ)[2]);
                              DEIM = true,
                              given_phi = false,
                              Φₑₓₜ::Array{T} = zeros(T,IP.nₕ,IP.nparam*IP.nₛ),
                              save = true) where {T}

    # K needs to already be augmented for boundary conditions
    nₜₛ = div(IP.nₜ,IP.nₛ-1)
    nₕᵣₐₙ = 1:P.bspl.nₕ

    if given_phi
        @assert IP.nₛ == IP.nₜ + 1
    end

    for p in 1:IP.nparam

        χ = μ[p,1]/params.κ
        print("running parameter nb. ", p, " with chi = ", χ, "\n")

        # initial conditions
        IC.zₓ .= Ψ' * P₀.x;  IC.zᵥ .= Ψ' * P₀.v
        IC.ρ₀ .= P.bspl.h

        # save initial conditions
        if save
            # solve for potential
            if given_phi
                IC.ϕ .= Φₑₓₜ[:,1 + (p-1)*IP.nₛ]
            else
                IC.x .= Ψ * IC.zₓ
                solve!(P, IC.x, P₀.w)
                IC.ϕ .= P.ϕ ./ χ^2
            end

            IC.Zₓ[:,1 + (p-1)*IP.nₛ] .= IC.zₓ
            IC.Zᵥ[:,1 + (p-1)*IP.nₛ] .= IC.zᵥ
            IC.Φ[:,1 + (p-1)*IP.nₛ] .= IC.ϕ

        end

        tₛ = 1
        for t = 1:IP.nₜ
            if save || t == 1
                # half an advection step
                IC.zₓ .+= 0.5 * IP.dt * IC.zᵥ * χ
            end

            # solve for potential
            if given_phi
                IC.ϕ = Φₑₓₜ[:,t + 1 + (p-1)*IP.nₛ]
            else
                IC.x .= Ψ * IC.zₓ
                solve!(P, IC.x, P₀.w)
                IC.ϕ .= P.ϕ ./ χ^2
            end

            # acceleration step
            if DEIM
                IC.xₑ .= ΠₑᵀΨ * IC.zₓ
                IC.zᵥ .+= IP.dt .* ΨᵀPₑ * eval_deriv_PBSBasis(IC.ϕ,P.bspl,IC.xₑ) .* χ
            else
                if given_phi
                    IC.x .= Ψ * IC.zₓ
                end
                IC.zᵥ .+= IP.dt .* Ψ' * eval_deriv_PBSBasis(IC.ϕ,P.bspl,IC.x) .* χ
            end

            if save || t == IP.nₜ
                # half an advection step
                IC.zₓ .+= 0.5 .* IP.dt .* IC.zᵥ .* χ
            else
                # full advection step
                IC.zₓ .+= IP.dt .* IC.zᵥ .* χ
            end

            if save

                # solve for potential
                # if given_phi
                #     IC.ϕ = Φₑₓₜ[:,t+1]
                # else
                #     IC.x .= Ψ * IC.zₓ
                #     IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,P.bspl,IC.rhs)
                #     IC.rhs[S.nₕ] = 0.0
                #     IC.ϕ = K\IC.rhs ./ χ^2
                # end


                if t%nₜₛ == 0
                    IC.Zₓ[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.zₓ
                    IC.Zᵥ[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.zᵥ
                    IC.Φ[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.ϕ
                    tₛ += 1
                end

            end
        end
    end

    return IC
end
