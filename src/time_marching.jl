struct IntegratorParameters{T}
    dt::T       # time step
    nₜ::Int     # number of time steps
    nₛ::Int     # number of saved time steps
    nₚ::Int     # number of equation parameters sampled

    nₕ::Int     # number of basis functions

    Nₚ::Int     # number of particles

    # function IntegratorParameters(dt::T, nₜ::Int, nₛ::Int, nₚ::Int, nₕ::Int, Nₚ::Int) where {T}
    #     new{T}(dt,nₜ,nₛ,nₚ,nₕ,Nₚ)
    # end
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

    Zₓ::Array{T}
    Zᵥ::Array{T}
    Φ::Array{T}
end

mutable struct IntegratorCache{T}
    x::Vector{T}
    v::Vector{T}
    a::Vector{T}

    ρ::Vector{T}
    ρ₀::Vector{T}
    ϕ::Vector{T}
    rhs::Vector{T}

    X::Array{T}
    V::Array{T}
    E::Array{T}
    # D::Array{T}
    Φ::Array{T}
end

IntegratorCache(IP::IntegratorParameters{T}) where {T} = IntegratorCache(zeros(T,IP.Nₚ), # x
                                                            zeros(T,IP.Nₚ), # v
                                                            zeros(T,IP.Nₚ), # a
                                                            zeros(T,IP.nₕ), # ρ
                                                            zeros(T,IP.nₕ), # ρ₀
                                                            zeros(T,IP.nₕ), # ϕ
                                                            zeros(T,IP.nₕ), # rhs
                                                            zeros(T,IP.Nₚ,IP.nₚ*IP.nₛ), # X
                                                            zeros(T,IP.Nₚ,IP.nₚ*IP.nₛ), # V
                                                            zeros(T,IP.Nₚ,IP.nₚ*IP.nₛ), # E
                                                            # zeros(T,IP.Nₚ,IP.nₚ*IP.nₛ*IP.nₕ), # D
                                                            zeros(T,IP.nₕ,IP.nₚ*IP.nₛ) # Φ
                                                            )

ReducedIntegratorCache(IP::IntegratorParameters{T},k::Int,kₑ::Int) where {T} = ReducedIntegratorCache(zeros(T,k), # zₓ
                                                                                zeros(T,k), # zᵥ
                                                                                zeros(T,IP.Nₚ), # x
                                                                                zeros(T,kₑ), # xₑ
                                                                                zeros(T,IP.nₕ), # ρ
                                                                                zeros(T,IP.nₕ), # ρ₀
                                                                                zeros(T,IP.nₕ), # ϕ
                                                                                zeros(T,IP.nₕ), # rhs
                                                                                zeros(T,k,IP.nₚ*IP.nₛ), # Zₓ
                                                                                zeros(T,k,IP.nₚ*IP.nₛ), # Zᵥ
                                                                                zeros(T,IP.nₕ,IP.nₚ*IP.nₛ) # Φ
                                                                                )

function integrate_vp(P₀::Particles{T},
                      S::PBSpline{T},
                      μ::Array{T},
                      μₛₐₘₚ::Vector{T},
                      K::Matrix{T},
                      IP::IntegratorParameters{T},
                      IC::IntegratorCache{T} = IntegratorCache(IP);
                      given_phi = false,
                      Φₑₓₜ::Array{T} = zeros(T,IP.nₕ,IP.nₚ*IP.nₛ),
                      save = true) where {T}

    # K needs to already be augmented for boundary conditions
    nₜₛ = div(IP.nₜ,IP.nₛ-1)
    nₕᵣₐₙ = 1:S.nₕ

    if given_phi
        @assert IP.nₛ == IP.nₜ + 1
    end

    for p in 1:IP.nₚ

        print("parameter nb. ", p, "\n")

        χ = μ[p,1]/μₛₐₘₚ[1]

        # initial conditions
        IC.x .= P₀.x;  IC.v .= P₀.v
        IC.ρ₀ .= S.h

        # save initial conditions
        if save

            # solve for potential
            if given_phi
                IC.ϕ .= Φₑₓₜ[:,1 + (p-1)*IP.nₛ]
            else
                IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,S,IC.rhs)
                IC.rhs[S.nₕ] = 0.0
                IC.ϕ = K\IC.rhs ./ χ^2
            end

            # electric field
            IC.a .= eval_deriv_PBSBasis(IC.ϕ,S,IC.x,IC.a,nₕᵣₐₙ)

            IC.X[:,1 + (p-1)*IP.nₛ] .= IC.x
            IC.V[:,1 + (p-1)*IP.nₛ] .= IC.v
            IC.E[:,1 + (p-1)*IP.nₛ] .= IC.a
            # for i in nₕᵣₐₙ
            #     IC.D[:,i + (p-1)*IP.nₛ*S.nₕ] = eval_PBSpline.(S,i,IC.x)
            # end
            IC.Φ[:,1 + (p-1)*IP.nₛ] .= IC.ϕ

        end

        tₛ = 1
        for t = 1:IP.nₜ
            if save || t == 1
                # half an advection step
                IC.x .+= 0.5 * IP.dt * IC.v * χ
            end

            # solve for potential
            if given_phi
                IC.ϕ .= Φₑₓₜ[:, t + 1 + (p-1)*IP.nₛ]
            else
                IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,S,IC.rhs)
                IC.rhs[S.nₕ] = 0.0
                IC.ϕ = K\IC.rhs ./ χ^2
            end

            # electric field
            IC.a .= eval_deriv_PBSBasis(IC.ϕ,S,IC.x,IC.a,nₕᵣₐₙ)

            # acceleration step
            IC.v .+= IP.dt .* IC.a .* χ

            if save || t == IP.nₜ
                # half an advection step
                IC.x .+= 0.5 .* IP.dt .* IC.v .* χ
            else
                # full advection step
                IC.x .+= IP.dt .* IC.v .* χ
            end

            if save

                # solve for potential
                # if given_phi
                #     IC.ϕ = Φₑₓₜ[:,t+1]
                # else
                #     IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,S,IC.rhs)
                #     IC.rhs[S.nₕ] = 0.0
                #     IC.ϕ = K\IC.rhs ./ χ^2
                # end

                if t%nₜₛ == 0
                    IC.X[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.x
                    IC.V[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.v
                    IC.E[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.a
                    # for i in nₕᵣₐₙ
                    #     IC.D[:,i + (tₛ + 1 - 1)*S.nₕ + (p-1)*IP.nₛ*S.nₕ] = eval_PBSpline.(S,i,IC.x)
                    # end
                    IC.Φ[:,tₛ + 1 + (p-1)*IP.nₛ] .= IC.ϕ
                    tₛ += 1
                end

            end
        end
    end

    return IC
end

function reduced_integrate_vp(P₀::Particles{T},
                              Ψ::Array{T},
                              ΨᵀPₑ::Array{T},      # Ψ' * Ψₑ * inv(Πₑ' * Ψₑ)
                              ΠₑᵀΨ::Array{T},    # Πₑ' * Ψ
                              S::PBSpline{T},
                              μ::Array{T},
                              μₛₐₘₚ::Vector{T},
                              K::Matrix{T},
                              IP::IntegratorParameters{T},
                              IC::ReducedIntegratorCache{T} = ReducedIntegratorCache(IP,size(Pₑ)[1],size(Pₑ)[2]);
                              DEIM = true,
                              given_phi = false,
                              Φₑₓₜ::Array{T} = zeros(T,IP.nₕ,IP.nₚ*IP.nₛ),
                              save = true) where {T}

    # K needs to already be augmented for boundary conditions
    nₜₛ = div(IP.nₜ,IP.nₛ-1)
    nₕᵣₐₙ = 1:S.nₕ

    if given_phi
        @assert IP.nₛ == IP.nₜ + 1
    end

    for p in 1:IP.nₚ

        χ = μ[p,1]/μₛₐₘₚ[1]
        print("running parameter nb. ", p, " with chi = ", χ, "\n")

        # initial conditions
        IC.zₓ .= Ψ' * P₀.x;  IC.zᵥ .= Ψ' * P₀.v
        IC.ρ₀ .= S.h

        # save initial conditions
        if save
            # solve for potential
            if given_phi
                IC.ϕ .= Φₑₓₜ[:,1 + (p-1)*IP.nₛ]
            else
                IC.x .= Ψ * IC.zₓ
                IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,S,IC.rhs)
                IC.rhs[S.nₕ] = 0.0
                IC.ϕ = K\IC.rhs ./ χ^2
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
                IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,S,IC.rhs)
                IC.rhs[S.nₕ] = 0.0
                IC.ϕ = K\IC.rhs ./ χ^2
            end

            # acceleration step
            if DEIM
                IC.xₑ .= ΠₑᵀΨ * IC.zₓ
                IC.zᵥ .+= IP.dt .* ΨᵀPₑ * eval_deriv_PBSBasis(IC.ϕ,S,IC.xₑ) .* χ
            else
                if given_phi
                    IC.x .= Ψ * IC.zₓ
                end
                IC.zᵥ .+= IP.dt .* Ψ' * eval_deriv_PBSBasis(IC.ϕ,S,IC.x) .* χ
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
                #     IC.rhs .= IC.ρ₀ .- rhs_particles_PBSBasis(IC.x,P₀.w,S,IC.rhs)
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
