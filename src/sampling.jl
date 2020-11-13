"""
Returns particles drawn from g for the bump-on-tail case.

input: nb. pf particles Nₚ, x-marginal function gₓ, parameters μ
output: Particles struct
"""
function draw_g_bumpontail_accept_reject(N::Int, fₓ, μ::Vector{T}) where {T}

    x = zeros(N); v = zeros(N); w = zeros(N)

    # Sobol sampling
    s = SobolSeq(2); skip(s, 2*N)

    n = 1
    while n < (N+1)
        ###
        y = next!(s)
        x₀ = y[1]*2.0*pi/μ[1]  # proposal, uniform

        # accept-reject in x
        if rand(1)[1] > ( (fₓ(x₀, μ))/(1 + μ[2]) )
            continue                            # reject
        end
        x[n] = x₀                               # accept
        w[n] = 1.0 .* 2.0*pi/μ[1] ./ N

        # inverse CDF sampling in v
        v₀ = y[2]*2.0 - 1.0
        v[n] = sqrt(2.0) * erfinv(v₀)  # draw from bulk
        if rand(1)[1] > 1-μ[3]
            v[n] = v[n]*μ[5] + μ[4]  # draw from tail
        end
        n = n+1
    end

    return Particles(x, v, w)
end

"""
Returns particles drawn from g for the bump-on-tail case.

input: nb. pf particles Nₚ, x-marginal function gₓ, parameters μ
output: Particles struct
"""
function draw_g_bumpontail_importance_sampling(N::Int, fₓ, μ::Vector{T}) where {T}

    x = zeros(N); v = zeros(N); w = zeros(N)

    # Sobol sampling
    s = SobolSeq(2); skip(s, 2*N)

    n = 1
    while n < (N+1)
        ###
        y = next!(s)
        x₀ = y[1]*2.0*pi/μ[1]  # proposal, uniform

        # importance sampling in x
        x[n] = x₀
        w[n] = fₓ(x₀, μ) .* 2.0*pi/μ[1] ./ N    # *L is equivalent to 1/(1/L) i.e. uniform proposal. f is NOT normalized!

        # inverse CDF sampling in v
        v₀ = y[2]*2.0 - 1.0
        v[n] = sqrt(2.0) * erfinv(v₀)  # draw from bulk
        if rand(1)[1] > 1-μ[3]
            v[n] = v[n]*μ[5] + μ[4]  # draw from tail
        end
        n = n+1
    end

    return Particles(x, v, w)
end

"""
Returns re-weighted particles according to new parameters.

input: Particles struct P₀, proposal distribution function g, target distribution function f
output: Particles struct
"""
function weight_f_bumpontail(P₀, g, f)
    Px = zero(P₀.x); Pv = zero(P₀.v); Pw = zero(P₀.w)
    Px .= P₀.x; Pv .= P₀.v; Pw .= P₀.w
    P = Particles(Px, Pv, Pw)
    P.w .*= f(P₀.x, P₀.v) ./ g(P₀.x, P₀.v)
    return P
end
