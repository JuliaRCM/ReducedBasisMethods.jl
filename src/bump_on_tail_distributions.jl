# x-part of distribution function
# μ = (κ, ε, a, v₀, σ)
function fₓ(x::T, μ::Vector{T}) where {T}
    @. ( 1 - μ[2]*cos(μ[1]*x) ) # * μ[1] / (2.0*pi)
end

# v-part of distribution function
function fᵥ(v::T, μ::Vector{T}) where {T}
    @. ( (1-μ[3])*1/sqrt(2*pi)*exp(-v^2/2)
        + μ[3]*1/sqrt(2*pi*μ[5]^2)*exp(-(v-μ[4])^2/(2*μ[5]^2)) )
end

function f(x::T, v::T, μ::Vector{T}) where {T}
    fₓ(x, μ) .* fᵥ(v, μ)
end

"""
Particles struct.
fields: positon x, velocity v, weight w
"""
mutable struct Particles{T}
    x::Vector{T}
    v::Vector{T}
    w::Vector{T}
end
