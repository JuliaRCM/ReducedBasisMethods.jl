
using Statistics

function find_maxima(w, n)
    maxima = []
    for i in (1+n):(length(w)-n)
        if issorted(w[i-n:i], lt=isless) && issorted(w[i:i+n], lt=isless, rev = true)
            append!(maxima, i)
        end
        end
    return maxima
end

function get_regression_αβ(t, W, n)
    nₚ = size(W, 2)
    α = zeros(nₚ)
    β = zeros(nₚ)
    for i in 1:nₚ
        m = find_maxima(W[:,i], n)
        #!! bump on tail
        m = m[2:5]
        β[i] = cov(t[m], log.(W[m,i]), corrected=false) / var(t[m], corrected=false) 
        α[i] = mean(log.(W[m,i])) - β[i]*mean(t[m])
    end
    return α, β
end
