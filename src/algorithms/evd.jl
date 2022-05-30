
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


"""
Obtains an orthogonal basis based on the snapshot matrix XV = [X V] ∈ R^(N × 2m) by calculating an EVD of XV' * XV (size: 2m × 2m)
The size of the basis is set as k. 
If k = 0 is given, the size is determined by the tolerance parameter as 1 - tol < (E(k) + Λ[k+1])/E(N) where E(k) is the sum of the largest k eigenvalues of the snapshot matrix.
Returns the k largest left eigenvectors in the N × k matrix Ψ and the corresponding eigenvalues in Λ
"""
function get_PODBasis_cotangentLiftEVD(X, V; k = 0, tolerance = 1e-4)
    N, m = size(X)
    @assert N, m == size(V)
    XVᵀXV = zeros(2m, 2m)

    XV = ApplyArray(hcat, X, V)
    XVᵀXV .= XV' * XV

    EVD = eigen(XVᵀXV)
    Λ, Ω = sorteigen(EVD.values, EVD.vectors) # Ω (size: 2m × 2m) contains the right eigenvectors of XV as columns

    E = sum(Λ)
    Eᵣ = 0
    if k == 0   # use tolerance based on ∑ᵢλᵢ
        k = 1
        while 1 - tol > (Eᵣ + Λ[k])/E
            Eᵣ += Λ[k]
            k+=1
        end
    end

    # Ψ contains the left eigenvectors of XV, which can be recovered from Σ⁻¹ XV Ω
    Ψ = XV * Ω[:,1:k]
    for i in 1:k
        Ψ[:,i] ./= sqrt(abs.(Λ[i]))
    end

    return Ψ, Λ[1:k]
end

"""
Obtains an orthogonal basis based on the snapshot matrix S ∈ R^(N × m) by calculating an EVD of S' * S (size: m × m)
The size of the basis is set as k. 
If k = 0 is given, the size is determined by the tolerance parameter as 1 - tol < (E(k) + Λ[k+1])/E(N) where E(k) is the sum of the largest k eigenvalues of the snapshot matrix.
Returns the k largest left eigenvectors in the N × k matrix Ψ and the corresponding eigenvalues in Λ
"""
function get_PODBasis_EVD(S; k = 0, tolerance = 1e-4)
    N, m = size(S)
    SᵀS = S' * S

    EVD = eigen(SᵀS)
    Λ, Ω = sorteigen(EVD.values, EVD.vectors) # Ω (size: m × m) contains the right eigenvectors of XV as columns

    E = sum(Λ)
    Eᵣ = 0
    if k == 0   # use tolerance based on ∑ᵢλᵢ
        k = 1
        while 1 - tol > (Eᵣ + Λ[k])/E
            Eᵣ += Λ[k]
            k+=1
        end
    end

    # Ψ contains the left eigenvectors of XV, which can be recovered from Σ⁻¹ XV Ω
    Ψ = S * Ω[:,1:k]
    for i in 1:k
        Ψ[:,i] ./= sqrt(abs.(Λ[i]))
    end

    return Ψ, Λ[1:k]
end

#=function get_ΛΩ_particles(X, V, IP; tolerance = 1e-4, k = 0)
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
=#

function ReducedBasis(alg::CotangentLiftEVD, ts::TrainingSet)
    # read integrator parameters
    IP = ts.integrator
    
    # read snapshot data
    X = reshape(ts.snapshots.X, (IP.nₚ, IP.nₛ * IP.nparam))
    V = reshape(ts.snapshots.V, (IP.nₚ, IP.nₛ * IP.nparam))
    E = reshape(ts.snapshots.A, (IP.nₚ, IP.nₛ * IP.nparam))
    
    # EVD
    Ψₚ, Λₚ = get_PODBasis_cotangentLiftEVD(X, V)
    kₚ = length(Λₚ)
    Ψₑ, Λₑ = get_PODBasis_EVD(E)
    kₑ = length(Λₑ)
    Πₑ = get_DEIM_interpolation_matrix(Ψₑ)

    #Λₚ, Ωₚ, kₚ, Ψₚ = get_ΛΩ_particles(X, V, IP)
    #Λₑ, Ωₑ, kₑ, Ψₑ = get_ΛΩ_efield(E)
    
    ReducedBasis(alg, ts.parameters, ts.paramspace, ts.initconds, ts.integrator, ts.poisson, Λₚ, kₚ, Ψₚ, Λₑ, kₑ, Ψₑ, Πₑ)
end

