
function get_Ψ(S, Λ, Ω, tol, k=0)
    E = sum(Λ)
    Eᵣ = 0
    if k == 0
        k = 1
        while 1 - (Eᵣ + Λ[k])/E > tol
            Eᵣ += Λ[k]
            k+=1
        end
    end
    Ψ = S*Ω[:,1:k]
    for i in 1:k
        Ψ[:,i] ./= sqrt(abs.(Λ[i]))
    end
    return k, Ψ
end

function get_ΛΩ_particles(X, V, IP; tolerance = 1e-4, k = 0)
    XV = X
    for p in 1:IP.nparam
         # XV = hcat(XV, V[:,1+(p-1)*IP.nₛ])
         XV = hcat(XV, V[:,1+(p-1)*IP.nₛ]) #, E[:,p*IP.nₛ])
         #print(1+(p-1)*IP.nₛ, " ", p*IP.nₛ, "\n" )
    end

    # XV = hcat(X, V);

    @time F = eigen(XV' * XV)
    @time Λ, Ω = sorteigen(F.values, F.vectors)

    # Projection Matrices
    @time k, Ψ = get_Ψ(XV, Λ, Ω, tolerance, k)

    return Λ, Ω, k, Ψ
end

function get_ΛΩ_efield(E; tolerance = 1e-4, k = 0)
    @time Fₑ = eigen(E' * E)
    @time Λₑ, Ωₑ = sorteigen(Fₑ.values, Fₑ.vectors)
    
    # Projection Matrices
    @time kₑ, Ψₑ = get_Ψ(E, Λₑ, Ωₑ, tolerance, k)

    return Λₑ, Ωₑ, kₑ, Ψₑ
end


function ReducedBasis(alg::EVD, ts::TrainingSet)
    # read integrator parameters
    IP = ts.integrator
    
    # read snapshot data
    X = reshape(ts.snapshots.X, (IP.nₚ, IP.nₛ * IP.nparam))
    V = reshape(ts.snapshots.V, (IP.nₚ, IP.nₛ * IP.nparam))
    E = reshape(ts.snapshots.A, (IP.nₚ, IP.nₛ * IP.nparam))
    
    # EVD
    Λₚ, Ωₚ, kₚ, Ψₚ = get_ΛΩ_particles(X, V, IP)
    Λₑ, Ωₑ, kₑ, Ψₑ = get_ΛΩ_efield(E)
    
    ReducedBasis(alg, ts.parameters, ts.paramspace, ts.initconds, ts.integrator, ts.poisson, Λₚ, Ωₚ, kₚ, Ψₚ, Λₑ, Ωₑ, kₑ, Ψₑ)
end
