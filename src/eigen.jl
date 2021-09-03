
function sorteigen(evals::Vector{T}, evecs::Matrix{T}) where {T<:Real}
    p = sortperm(abs.(evals), lt=isless, rev=true)
    return evals[p], evecs[:, p]
end

