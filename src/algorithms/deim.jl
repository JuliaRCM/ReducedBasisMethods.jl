
"""
Calculates the DEIM interpolation matrix as desribed in Algorithm 1 in doi.org/10.1137/090766498 to cheaply evaluate f(x)
The input matrix Ψ holds the eigenvectors of the snapshot matrix fᵢ(xⱼ), 1 ≤ i ≤ N, 1 ≤ j ≤ k. 
"""
function get_DEIM_interpolation_matrix(Ψ::AbstractMatrix)
    N, k = size(Ψ)
    j = argmax(Ψ[:,1])
    Π = zeros(N, k)
    Π[j,1] = 1.0
    r = zeros(N)

    for i in 2:k
        c  = @views (Π[:,1:(i-1)]' * Ψ[:,1:(i-1)]) \ (Π[:,1:(i-1)]' * Ψ[:,i])
        r .= @views Ψ[:,i] .- Ψ[:,1:(i-1)] * c
        j = argmax(r)

        Π[j,i] = 1.0
    end
    return Π
end
