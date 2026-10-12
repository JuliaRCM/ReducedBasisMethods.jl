using ForwardDiff
using GeometricIntegrators
using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedBasisMethods: _deim_indices
using ReducedComplexityModeling: Parameter, ParameterSpace
using Test

# ---------------------------------------------------------------------------
# The test system: a chain of `n = 64` pendulums, with
#
#   H(x) = ½xᵀLx + Σᵢ (1 − cos qᵢ),  x = [q; p],  L = blockdiag(K, I),
#   K = 4·(periodic second difference: 2 on the diagonal, −1 off it and in the corners).
#
# The potential is not quadratic, so the hyper-reduced gradient of plain DEIM is not the
# gradient of a scalar, while symplectic DEIM's is exactly that. The two methods differ in
# nothing else, which is what makes the pair a test.
# ---------------------------------------------------------------------------
const NQ = 64

"""The POD basis of `S` at `k`, by the `ReducedBasis` route."""
function pod_basis(S; k)
    basis = ReducedBasis(PODEVD(), S, NamedTuple(),
        ParameterSpace(Parameter(:μ, 1.0, 3.0, size(S, 2))); k)
    basis.Φ
end

const KQ = let K = zeros(NQ, NQ)
    for i in 1:NQ
        K[i, i] = 2
        K[i, mod1(i + 1, NQ)] -= 1
        K[mod1(i + 1, NQ), i] -= 1
    end
    4 .* K
end

const LQ = [KQ zeros(NQ, NQ); zeros(NQ, NQ) Matrix{Float64}(I, NQ, NQ)]

const X0 = [[1.5 * sin(2π * (i - 1) / NQ) + 0.3 * cos(4π * (i - 1) / NQ) for i in 1:NQ];
            zeros(NQ)]

"""The full vector field `ẋ = J₂ₙ∇H(x)` of the test system."""
function full_vectorfield!(dx, t, x, params)
    dx[1:NQ] .= x[(NQ + 1):2NQ]
    dx[(NQ + 1):2NQ] .= .-(KQ * x[1:NQ]) .- sin.(x[1:NQ])
    return nothing
end

"""The full Hamiltonian `H(x)` of the test system."""
function full_hamiltonian(x)
    q = x[1:NQ]
    p = x[(NQ + 1):2NQ]
    return (p' * p + q' * (KQ * q)) / 2 + sum(1 .- cos.(q))
end

"""The density vector `[h₁(x) … h_M(x)]` of the test system."""
density(x::AbstractVector) = 1 .- cos.(x[1:NQ])

"""The density snapshots of a set of states, one state per column."""
density(X::AbstractMatrix) = 1 .- cos.(X[1:NQ, :])

# The training set: implicit midpoint in Float64, dt = 0.005 to T = 20, every 10th step saved.
const XTRAIN = let
    prob = ODEProblem(full_vectorfield!, (0.0, 20.0), 0.005, X0)
    sol = integrate(prob, ImplicitMidpoint())
    reduce(hcat, (sol.q[n] for n in 0:10:ntime(prob)))
end

const KT = 6 # the reduced size
const MT = 4 # the number of DEIM densities

# Φ: the first k left singular vectors of [Q P]. U: an orthonormal basis of the densities.
const ΦTRAIN = svd([XTRAIN[1:NQ, :] XTRAIN[(NQ + 1):2NQ, :]]).U[:, 1:KT]
const UTRAIN = svd(density(XTRAIN)).U[:, 1:MT]

# The stencil and the density of the test system: hᵢ reads qᵢ alone.
stencil_test(i) = [i]
h_test(xs, i) = 1 - cos(xs[1])
∇h_test(xs, i) = [sin(xs[1])]

"""The reduced bases of the test system, in `T`."""
function test_bases(::Type{T}) where {T}
    Φ = T.(ΦTRAIN)
    A = [Φ zeros(T, NQ, KT); zeros(T, NQ, KT) Φ]
    return (; Φ, A, L = T.(LQ), U = T.(UTRAIN), z₀ = A' * T.(X0))
end

"""`J₂ₖ = [0 I; −I 0]`, in `T`."""
function symplectic_matrix(::Type{T}) where {T}
    Ik = Matrix{T}(I, KT, KT)
    return [zeros(T, KT, KT) Ik; -Ik zeros(T, KT, KT)]
end

"""`∇H̃(z)`, in a new vector — the function `ForwardDiff` differentiates."""
function sd_gradient(sd, z)
    g = similar(z)
    hamiltonian_gradient!(g, sd, z)
    return g
end

"""`@allocated` of `f(sd, z, g)`, after a first call that warms the barrier."""
function allocations(f::F, sd::S, z::Z, g::G) where {F, S, Z, G}
    (f(sd, z, g); @allocated f(sd, z, g))
end

@testset "the reduced gradient is a gradient" begin
    for T in (Float32, Float64)
        b = test_bases(T)
        sd = SymplecticDEIM(b.A, b.L, b.U, stencil_test, h_test, ∇h_test)

        @test @inferred(sd(b.z₀)) isa T
        @test eltype(@inferred(hamiltonian_gradient!(similar(b.z₀), sd, b.z₀))) === T

        # The plain DEIM reduced gradient of Peng–Mohseni (5.8), built here and not in `src`.
        Ψ = T.(pod_basis(sin.(XTRAIN[1:NQ, :]); k = KT))
        Π = T.(get_DEIM_interpolation_matrix(Ψ))
        W = Ψ * inv(Π' * Ψ)
        plain(zz) = begin
            g = sd.L̃ * zz
            g[1:KT] .+= b.Φ' * (W * sin.(Π' * (b.Φ * zz[1:KT])))
            return g
        end

        Z = randn(Xoshiro(1), T, 2KT, 20)

        for n in 1:20
            z = Z[:, n]

            # The Jacobian of the hyper-reduced gradient is symmetric by construction.
            Jg = ForwardDiff.jacobian(zz -> sd_gradient(sd, zz), z)
            @test norm(Jg - Jg', 2) ≤ sqrt(eps(T)) * norm(Jg, 2)

            # It is the derivative of `sd`, which is what makes `H̃` an invariant.
            g = sd_gradient(sd, z)
            gf = ForwardDiff.gradient(sd, z)
            @test norm(g - gf) ≤ sqrt(eps(T)) * norm(gf)

            # Plain DEIM on the same system is not a gradient, and by a wide margin.
            Jp = ForwardDiff.jacobian(plain, z)
            @test norm(Jp - Jp', 2) ≥ 100 * sqrt(eps(T)) * norm(Jp, 2)
        end
    end
end

@testset "the field and its gradient allocate nothing" begin
    for T in (Float32, Float64)
        b = test_bases(T)
        # `stencil` and the two callbacks below allocate nothing, so the barrier measures the
        # methods' own code alone. Here `∇h` returns a `Tuple`.
        sd = SymplecticDEIM(b.A, b.L, b.U, i -> i:i, h_test, (xs, i) -> (sin(xs[1]),))
        z = b.z₀
        g = similar(z)

        @test @inferred(sd(z)) isa T
        @test @inferred(hamiltonian_gradient!(g, sd, z)) === g

        # A gradient a `Tuple`-valued `∇h` builds is the gradient a `Vector`-valued one builds,
        # on the same stencil: either is indexable, which is all the accumulation reads.
        sdv = SymplecticDEIM(b.A, b.L, b.U, i -> i:i, h_test, ∇h_test)
        @test hamiltonian_gradient!(g, sd, z) == hamiltonian_gradient!(similar(z), sdv, z)

        # Coverage instrumentation allocates on its own account, so the readings are taken only
        # where the runner does not instrument the code.
        if Base.JLOptions().code_coverage == 0
            @test allocations((s, z, g) -> collect(z), sd, z, g) > 0
            @test allocations((s, z, g) -> s(z), sd, z, g) == 0
            @test allocations((s, z, g) -> hamiltonian_gradient!(g, s, z), sd, z, g) == 0
        end
    end
end

@testset "the weights" begin
    for T in (Float32, Float64)
        b = test_bases(T)
        U = T.(svd(density(XTRAIN)).U)

        # At m = M the interpolation is exact for the densities: w = 𝟙 and H̃(z) = H(Az).
        sd = SymplecticDEIM(b.A, b.L, U, stencil_test, h_test, ∇h_test)

        @test _deim_indices(U) != 1:NQ
        @test norm(sd.w .- one(T), Inf) ≤ sqrt(eps(T))

        for z in eachslice(randn(Xoshiro(4), T, 2KT, 20); dims = 2)
            H = full_hamiltonian(b.A * z)
            @test abs(sd(z) - H) ≤ sqrt(eps(T)) * abs(H)
        end

        # At m = 8 the bound of the method holds on every training state. The weights are not
        # ones there — they sum to about `M/m` — so `w ≡ 1` violates the same bound.
        U₈ = U[:, 1:8]
        sd₈ = SymplecticDEIM(b.A, b.L, U₈, stencil_test, h_test, ∇h_test)
        PᵀU = U₈[sd₈.p, :]

        ratio = Float64[]
        for j in axes(XTRAIN, 2)
            z = b.A' * T.(XTRAIN[:, j])
            h = density(b.A * z)
            lhs = abs(sd₈(z) - full_hamiltonian(b.A * z))
            rhs = sqrt(NQ) * opnorm(inv(PᵀU), 2) * norm(h - U₈ * (U₈' * h))
            push!(ratio, lhs / rhs)
        end

        @test maximum(ratio) ≤ 1
    end

    # Every density the same is not the same as the right density: the index handed to `h`,
    # `∇h` and `stencil` is the DEIM index, not the loop counter.
    for T in (Float32, Float64)
        b = test_bases(T)
        U = T.(svd(density(XTRAIN)).U)
        hᵢ(xs, i) = (1 + i / NQ) * (1 - cos(xs[1]))
        ∇hᵢ(xs, i) = [(1 + i / NQ) * sin(xs[1])]
        sdᵢ = SymplecticDEIM(b.A, b.L, U, stencil_test, hᵢ, ∇hᵢ)

        @test _deim_indices(U) != 1:NQ

        for z in eachslice(randn(Xoshiro(5), T, 2KT, 20); dims = 2)
            Az = b.A * z
            ref = (z' * (sdᵢ.L̃ * z)) / 2 +
                  sum((1 + i / NQ) * (1 - cos(Az[i])) for i in 1:NQ)
            @test abs(sdᵢ(z) - ref) ≤ sqrt(eps(T)) * abs(ref)

            # The gradient carries the same labels as `H̃`: `∇h` receives `pⱼ`, the DEIM index,
            # and not the counter `j` of the loop over the selected densities. With a label in
            # the density itself, a swapped label moves the gradient away from the derivative
            # of `H̃`, which is what `ForwardDiff` gives.
            gᵢ = sd_gradient(sdᵢ, z)
            gfᵢ = ForwardDiff.gradient(sdᵢ, z)
            @test norm(gᵢ - gfᵢ) ≤ sqrt(eps(T)) * norm(gfᵢ)
        end
    end
end

@testset "the edges" begin
    for T in (Float32, Float64)
        b = test_bases(T)

        # A non-symmetric L gives the gradient of its symmetric part, and a symmetric Jacobian.
        S = randn(Xoshiro(2), 2NQ, 2NQ)
        S = S - S' # skew
        sd = SymplecticDEIM(b.A, b.L, b.U, stencil_test, h_test, ∇h_test)
        sdskew = SymplecticDEIM(b.A, b.L + T.(S), b.U, stencil_test, h_test, ∇h_test)

        z = randn(Xoshiro(3), T, 2KT)
        g = sd_gradient(sd, z)
        gskew = sd_gradient(sdskew, z)
        @test norm(gskew - g, Inf) ≤ sqrt(eps(T)) * norm(g, Inf)
        Jskew = ForwardDiff.jacobian(zz -> sd_gradient(sdskew, zz), z)
        @test norm(Jskew - Jskew', 2) ≤ sqrt(eps(T)) * norm(Jskew, 2)

        # An integer L with Float32 bases gives Float32 fields, not Float64 ones.
        sdint = SymplecticDEIM(b.A, Matrix{Int}(b.L), b.U, stencil_test, h_test, ∇h_test)
        @test eltype(sdint.w) === T
        @test eltype(sdint.L̃) === T
        @test sdint(z) isa T

        # A stencil may return any `AbstractVector{<:Integer}`.
        sd32 = SymplecticDEIM(b.A, b.L, b.U, i -> Int32[i], h_test, ∇h_test)
        sdrange = SymplecticDEIM(b.A, b.L, b.U, i -> i:i, h_test, ∇h_test)
        @test sd32(z) == sd(z)
        @test sdrange(z) == sd(z)

        # m = 0: no density enters H̃, and the reduced system is the linear one.
        sd0 = SymplecticDEIM(b.A, b.L, zeros(T, NQ, 0), stencil_test, h_test, ∇h_test)
        @test isempty(sd0.p)
        @test isempty(sd0.w)
        @test sd0(z) ≈ (z' * (sd0.L̃ * z)) / 2
        @test sd_gradient(sd0, z) ≈ sd0.L̃ * z

        # `L̃` is symmetrised after the product, so it is symmetric to the last bit and not only
        # to the round-off of the product. On the reduced basis that is not visible; a random
        # basis is what separates the two.
        Arand = randn(Xoshiro(6), T, 2NQ, 5)
        Lrand = randn(Xoshiro(7), T, 2NQ, 2NQ)
        sdrand = SymplecticDEIM(Arand, Lrand, b.U, stencil_test, h_test, ∇h_test)
        @test sdrand.L̃ == transpose(sdrand.L̃)

        # `L` is symmetrised before the product too. A skew `L` has a zero symmetric part, so
        # `L̃` is exactly zero, where symmetrising the product alone keeps its round-off.
        Lskew = Lrand - transpose(Lrand)
        @test iszero(SymplecticDEIM(Arand, Lskew, b.U, stencil_test, h_test, ∇h_test).L̃)

        # A non-finite U reaches the DEIM selection, which rejects it. The `Inf` stands outside
        # the rows the weight solve touches, so the `ArgumentError` here is the validator's: a
        # second index routine without the `0 < ρ < Inf` test would construct in silence.
        Uinf = T.(UTRAIN)
        Uinf[10, 1] = T(Inf)
        @test_throws ArgumentError SymplecticDEIM(
            b.A, b.L, Uinf, stencil_test, h_test, ∇h_test)

        # And so does a `NaN` in U.
        Unan = T.(UTRAIN)
        Unan[1, 1] = T(NaN)
        @test_throws ArgumentError SymplecticDEIM(
            b.A, b.L, Unan, stencil_test, h_test, ∇h_test)
    end
end

@testset "the horizon" begin
    for T in (Float32, Float64)
        b = test_bases(T)
        sd = SymplecticDEIM(b.A, b.L, b.U, stencil_test, h_test, ∇h_test)
        J = symplectic_matrix(T)
        abstol = T === Float32 ? T(1e-5) : T(1e-12)

        Ψ = T.(pod_basis(sin.(XTRAIN[1:NQ, :]); k = KT))
        Π = T.(get_DEIM_interpolation_matrix(Ψ))
        W = Ψ * inv(Π' * Ψ)

        # `maxₙ |H̃(zₙ) − H̃(z₀)|` over [0, T], the drift of the reduced system `v`.
        function drift(v, dt)
            prob = ODEProblem((dz, t, z, params) -> (v(dz, z); dz .= J * dz; nothing),
                (zero(T), T(100)), dt, copy(b.z₀))
            sol = integrate(prob, ImplicitMidpoint(), f_abstol = abstol)
            states = [sol.q[n] for n in 0:ntime(prob)]
            @test all(all(isfinite, s) for s in states)
            return maximum(abs, sd.(states) .- sd(states[1]))
        end

        symplectic(dz, z) = hamiltonian_gradient!(dz, sd, z)
        pengmohseni(dz, z) = begin
            dz .= sd.L̃ * z
            dz[1:KT] .+= b.Φ' * (W * sin.(Π' * (b.Φ * z[1:KT])))
            return nothing
        end

        Δ₂ = drift(symplectic, T(0.2))
        Δ₁ = drift(symplectic, T(0.1))
        Δ₁p = drift(pengmohseni, T(0.1))

        # Both are the method's own error: a per-step value can be exactly zero in Float32.
        @test Δ₂ > 0
        @test Δ₁ > 0

        # Halving the step divides the drift by 3 to 5 — second order.
        @test 3 ≤ Δ₂ / Δ₁ ≤ 5

        # The solver's own tolerance is far below the drift it is asked to resolve.
        @test abstol ≤ 1e-3 * min(Δ₂, Δ₁)

        # The same H̃, evaluated on the trajectory of plain DEIM, drifts by an order of
        # magnitude more: the invariant is the symplectic construction's, not the system's.
        @test Δ₁p ≥ 10 * Δ₁
    end
end
