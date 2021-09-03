
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
