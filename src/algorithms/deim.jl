
function deim_get_Π(Ψ::AbstractMatrix)
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
