"""
Example 1 from doi.org/10.1137/090766498
"""
function s(x, μ) 
    (1-x) * cos(3π * μ * (x+1)) * exp(-(1+x)*μ)
end

x = range(-1, 1, 100)
μₜ = range(1, π, 51) # training parameters
μᵥ = range(1, π, 101) # validation/testing parameters

S = [ s(x[i], μₜ[j]) for i in 1:100, j in 1:51 ]

Ψ, Λ = get_PODBasis_EVD(S; k = 20)

@test Λ[10] < 1e0
@test Λ[15] < 1e-2
@test Λ[20] < 1e-7

Π = get_DEIM_interpolation_matrix(Ψ)

Sᵥ = [ s(x[i], μᵥ[j]) for i in 1:100, j in 1:101 ] # validation set

D = Ψ * inv(Π' * Ψ)
xₑ = Π' * x

Sₐ = zero(Sᵥ) # DEIM approximation

for i in 1:101
    sx = x -> s(x, μᵥ[i])
    Sₐ[:,i] .= D * sx.(xₑ)
end

avg_deim_error = sum( [ norm(Sᵥ[:,i] - Sₐ[:,i], 2) for i in 1:101 ] ) / 101

@assert avg_deim_error < 5e-5