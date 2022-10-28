
using LinearAlgebra: lu, ldiv!, norm
using PoissonSolvers: PBSpline, stiffnessmatrix, eval_deriv_PBSBasis, rhs_particles_PBSBasis
using ParticleMethods: ParticleList


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

function Base.:(==)(ip1::IntegratorParameters{T1}, ip2::IntegratorParameters{T2}) where {T1,T2}
    T1 == T2 &&
    ip1.dt == ip2.dt &&
    ip1.nₜ == ip2.nₜ &&
    ip1.nₛ == ip2.nₛ &&
    ip1.nₕ == ip2.nₕ &&
    ip1.nₚ == ip2.nₚ &&
    ip1.nparam == ip2.nparam &&
    ip1.t == ip2.t
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
                              pspace::ParameterSpace,
                              params::NamedTuple,
                              poisson::PoissonSolverPBSplines{T},
                              SS,
                              IP::IntegratorParameters{T},
                              IC::ReducedIntegratorCache{T} = ReducedIntegratorCache(IP,size(Pₑ)[1],size(Pₑ)[2]);
                              DEIM = true,
                              given_phi = false,
                              Φₑₓₜ::Array{T} = zeros(T,IP.nₕ,IP.nparam*IP.nₛ),
                              save = true) where {T}

    # K needs to already be augmented for boundary conditions
    nₜₛ = div(IP.nₜ,IP.nₛ-1)
    nₕᵣₐₙ = 1:poisson.bspl.nₕ

    if given_phi
        @assert IP.nₛ == IP.nₜ + 1
    end

    for p in eachindex(pspace)
    
        # initial conditions
        IC.zₓ .= Ψ' * vec(P₀.x)
        IC.zᵥ .= Ψ' * vec(P₀.v)
        IC.ρ₀ .= poisson.bspl.h
        w = vec(P₀.w)

        tₛ = 1

        χ = pspace(p).χ
        println("running parameter nb. ", p, " with chi = ", χ, ", and ic error = ", norm(P₀.x .- Ψ * IC.zₓ))

        # save initial conditions
        if save
            # solve for potential
            if given_phi
                IC.ϕ .= Φₑₓₜ[:,1 + (p-1)*IP.nₛ]
            else
                IC.x .= Ψ * IC.zₓ
                solve!(poisson, IC.x, w)
                IC.ϕ .= poisson.ϕ ./ χ^2
            end

            x = Ψ * IC.zₓ
            v = Ψ * IC.zᵥ

            # copy solution
            SS.X[1,:,tₛ,p] .= x
            SS.V[1,:,tₛ,p] .= v
            SS.Φ[1,:,tₛ,p] .= IC.ϕ

            # copy diagnostics
            SS.W[tₛ,p] = dot(IC.ϕ, poisson.S, IC.ϕ) / 2 * χ^2
            SS.K[tₛ,p] = dot(w .* v, v) / 2
            SS.M[tₛ,p] = dot(w, v)

            tₛ += 1            

        end

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
                solve!(poisson, IC.x, w)
                IC.ϕ .= poisson.ϕ ./ χ^2
            end

            # acceleration step
            if DEIM
                IC.xₑ .= ΠₑᵀΨ * IC.zₓ
                IC.zᵥ .+= IP.dt .* ΨᵀPₑ * eval_deriv_PBSBasis(IC.ϕ,poisson.bspl,IC.xₑ) .* χ
            else
                if given_phi
                    IC.x .= Ψ * IC.zₓ
                end
                IC.zᵥ .+= IP.dt .* Ψ' * eval_deriv_PBSBasis(IC.ϕ,poisson.bspl,IC.x) .* χ
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

                # if t%nₜₛ == 0
                    x = Ψ * IC.zₓ
                    v = Ψ * IC.zᵥ
    
                    # copy solution
                    SS.X[1,:,tₛ,p] .= x
                    SS.V[1,:,tₛ,p] .= v
                    SS.Φ[1,:,tₛ,p] .= IC.ϕ

                    # copy diagnostics
                    SS.W[tₛ,p] = dot(IC.ϕ, poisson.S, IC.ϕ) / 2 * χ^2
                    SS.K[tₛ,p] = dot(w .* v, v) / 2
                    SS.M[tₛ,p] = dot(w, v)

                    tₛ += 1
                # end

            end
        end
    end

    return IC
end
