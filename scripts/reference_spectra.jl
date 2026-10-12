# The paper's data, regenerated from the closed form `reference/Vlasov_Analytic.jl` writes, and
# the singular values of the three bases the reference scripts build from it.
#
#     julia --project=. scripts/reference_spectra.jl [stride] [nbeta]
#
# `reference/CalculateProjection_POD.jl`, `_PSD_Cotangent.jl` and `_PSD_Complex.jl` read six
# β-runs written by `reference/Vlasov_Analytic.jl` and take every tenth saved column of the
# 1000001 in each file, which is `t = 0.001 * (0:10:10^6)`: 600006 columns, and a 2000 × 600006
# POD matrix, which is about 8.94 GiB (9.60 GB) in `Float64`. Those HDF5 files are not in the
# tree, so this script regenerates the kept columns from the closed form instead. The generators are
# `Vlasov_Analytic.jl:27-46` (the initial conditions and the bump) and `:58-65` (the two
# closed-form lines), with its own constants and its own seeds.
#
# The defaults `stride = 10` and `nbeta = 6` are the reference's configuration. A larger stride
# or fewer β-runs gives a short run: the script prints the data scale it uses.
# The data comes from `Random.seed!`, which is Xoshiro and not the MersenneTwister
# stream the paper's files were written from, so these are regenerated singular values and not
# the paper's.
#
# Each route is reduced by the package — the SVD routes only, the EVD route is for `N ≫ m` — and
# compared with `svdvals` of the matrix the reference's own `svd` call decomposes, as
# `CalculateProjection_POD.jl:39`, `_PSD_Cotangent.jl:44-46` and `_PSD_Complex.jl:43-46` make it. Every singular value past the
# rank of the snapshot is round-off and is compared with the absolute Weyl bound
# `max(m, n) * eps(Float64) * σ₁`, never with a relative one.
#
# It exits non-zero when a route's singular values differ from the reference's, or when a
# round-off singular value exceeds that bound.

using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedComplexityModeling: Parameter, ParameterSpace

# the constants of reference/Vlasov_Analytic.jl
const BETA = (5.95, 5.97, 5.99, 6.01, 6.03, 6.05)
const N_PARTICLES = 1000
const DT = 0.001
const NT = 1_000_000
const SIGMAX = 10.0
const SIGMAV = 0.5
const V0 = 4.0
const A = 0.3

"the saved columns of `CalculateProjection_POD.jl`, `t = DT * (0:stride:NT)`"
time_columns(stride::Int) = DT .* (0:stride:NT)

"The initial conditions of `Vlasov_Analytic.jl:27-46`, with its three seeds."
function initial_conditions(n::Int)
    Random.seed!(1)
    q0 = SIGMAX .* randn(n)

    Random.seed!(10)
    p0 = randn(n)

    Random.seed!(100)
    for i in 1:n
        rand() >= 1 / (1 + A) && (p0[i] = SIGMAV * p0[i] + V0)
    end

    return q0, p0
end

"The `β`-run of `Vlasov_Analytic.jl:58-65` at the saved columns, into `q`."
function beta_run!(q::AbstractMatrix, q0::Vector{Float64}, p0::Vector{Float64},
        beta::Float64, t::AbstractVector{Float64})
    n = length(q0)
    for (i, ti) in enumerate(t)
        for j in 1:n
            q[j, i] = (p0[j] / beta) * sin(beta * ti) + q0[j] * cos(beta * ti)
            q[n + j, i] = p0[j] * cos(beta * ti) - beta * q0[j] * sin(beta * ti)
        end
    end
    return q
end

"""
    spectrum_deviation(σ, σref, reference)

The absolute deviation of the spectrum `σ` from the reference spectrum `σref`,
`maximum(abs.(σ .- σref))`, and the Weyl bound `max(m, n)·eps(Float64)·σ₁` it is compared with,
where `m` and `n` are the dimensions of the `reference` matrix and `σ₁` is `σref`'s largest
singular value.

Both numbers are absolute. Dividing the deviation by `σ₁` before the comparison — the bound
already carries `σ₁`, so the division buys a factor `σ₁` — accepts a deviation that many times the
bound.
"""
function spectrum_deviation(σ::AbstractVector, σref::AbstractVector, reference::AbstractMatrix)
    bound = max(size(reference)...) * eps(Float64) * σref[1]
    return maximum(abs(σ[i] - σref[i]) for i in eachindex(σ, σref)), bound
end

"""
The singular values of one route against the reference's own `svd`, and the energy-selected `k` at
each tolerance. One reduction gives the whole spectrum, so the tolerances are read off it rather
than reduced again.
"""
function route(name::AbstractString, alg, snapshot::AbstractMatrix,
        reference::AbstractMatrix, rank::Int, tolerances)
    pspace = ParameterSpace(Parameter(:β, BETA[1], BETA[end], length(BETA)))
    b = ReducedBasis(alg, snapshot, NamedTuple(), pspace; k = 1)

    σ = sqrt.(b.Λ)
    σref = svdvals(reference)
    σ₁ = σ[1]
    deviation, bound = spectrum_deviation(σ, σref, reference)

    tail = maximum(view(σ, (rank + 1):length(σ)))
    ks = [ReducedBasisMethods._basis_size(b.Λ, nothing, tol) for tol in tolerances]

    println(rpad(name, 20), " σ₁ = ", σ₁, ",  rank = ", rank)
    println(rpad("", 20), " σ = ", σ)
    println(rpad("", 20), " reference svd: ", deviation, " ≤ ",
        bound, "  ", deviation ≤ bound ? "ok" : "FAIL")
    println(rpad("", 20), " σ past rank:   ", tail, " ≤ ", bound, "  ",
        tail ≤ bound ? "ok" : "FAIL")
    println(rpad("", 20), " k at tolerance: ",
        join(["$tol → $k" for (tol, k) in zip(tolerances, ks)], ",  "))

    return deviation ≤ bound && tail ≤ bound
end

function main(stride::Int, nbeta::Int)
    n = N_PARTICLES
    t = time_columns(stride)
    q0, p0 = initial_conditions(n)

    M = Matrix{Float64}(undef, 2n, nbeta * length(t))
    for i in 1:nbeta
        beta_run!(view(M, :, ((i - 1) * length(t) + 1):(i * length(t))),
            q0, p0, BETA[i], t)
    end
    C = size(M, 2)

    println("ReducedBasisMethods — the paper's spectra, regenerated from the closed form")
    println("  random:  Random.seed! (Xoshiro): the seeds 1, 10 and 100 of Vlasov_Analytic.jl")
    println("  data:    n = $n, β = $(BETA[1:nbeta]), t = $DT * (0:$stride:$NT), ",
        "$(length(t)) columns per β-run")
    println("  matrix:  $(2n) × $C in Float64, ",
        round(8.0 * 2n * C / 2^30; digits = 2), " GiB")
    println("  these are regenerated singular values, not the paper's: the paper's files were")
    println("  written from the MersenneTwister stream, and are not in the tree")
    println()

    tolerances = (1e-4, 1e-8, 1e-12)
    ok = true

    ok &= route("POD, SVD", PODSVD(), M, M, nbeta == 1 ? 2 : 3, tolerances)
    Q = view(M, 1:n, :)
    P = view(M, (n + 1):(2n), :)
    ok &= route("cotangent lift, SVD", CotangentLiftSVD(), M, hcat(Q, P), 2, tolerances)
    ok &= route("complex SVD", ComplexSVD(), M, Q .+ im .* P, 2, tolerances)

    println()
    println(ok ? "PASS" : "FAIL")

    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10,
        length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6) ? 0 : 1)
end
