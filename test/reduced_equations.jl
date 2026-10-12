#
# The reduced equation forms of T7.11: the symplectic form `ż = J_k Aᵀ ∇H(Az)`, the Galerkin form
# `ż = Uᵀ X_H(Uz)` of a POD basis, and the split PODE variant, emitted as `GeometricEquations`
# problems.
#
# Every behavioural check integrates with GeometricIntegrators' own `integrate`. The references
# are dense — `J_N`, `J_k` and `A = Matrix(<the problem's basis>)` — and use no `project`, no
# `lift` and no blockwise helper of the package.
#

using ForwardDiff
using GeometricIntegrators
using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedComplexityModeling: Parameter, ParameterSpace, project
using Test
using TOML

# The half-dimension: the lifted state is `2N` long, and a cotangent-lift basis is `N × k`.
const N = 100

# The 100 seeded points of checks 4, 5 and 6.
const NPTS = 100

# The five tags, split as the two forms.
const POD_TAGS = (PODSVD(), PODEVD())
const SYMPLECTIC_TAGS = (CotangentLiftSVD(), CotangentLiftEVD(), ComplexSVD())
const TAGS = (POD_TAGS..., SYMPLECTIC_TAGS...)

# The two methods, and the reduced sizes every tag is built at.
const METHODS = (ImplicitMidpoint(), ExplicitEuler())
const KS = (10, 1)

"`γ`, the `q`-`p` coupling of the second Hamiltonian."
const γ2 = 0.37

"A parameter space, as `test/projection.jl` builds one."
testparamspace() = ParameterSpace(Parameter(:σ, 0.0, 1.0, 2))

"The caller's parameters of the second Hamiltonian, in `T`."
params2(::Type{T}) where {T} = (α = T(1.3),)

"`(timespan, timestep, steps)`: `10⁴` steps in `Float64`, `10³` in `Float32`."
integration_window(::Type{Float64}) = (; span = (0.0, 100.0), dt = 0.01, steps = 10^4)
integration_window(::Type{Float32}) = (; span = (0.0f0, 10.0f0), dt = 0.01f0, steps = 10^3)

"The conservation bound of check 1, at the horizon of that element type."
conservation_bound(::Type{Float64}) = 1e-10
conservation_bound(::Type{Float32}) = 10^3 * eps(Float32)

"The solver tolerance of check 1: the section's value, or `max(8, 2k)·eps` in a small type."
solver_tolerance(::Type{Float64}, k::Int) = 1e-14
solver_tolerance(::Type{Float32}, k::Int) = max(8, 2k) * eps(Float32)

"""
The seeded snapshot, the initial state and the 100 points of the section's inputs, in `T`.

`S` is the `2N × 30` stacked snapshot `[Q; P]`, which every tag accepts: a POD basis of it is
`2N × k`, and a cotangent-lift or complex basis is `N × k` with a `2N × 2k` assembly.
"""
function fixtures(::Type{T}) where {T}
    (; S = randn(Xoshiro(1), T, 2N, 30),
        x₀ = randn(Xoshiro(2), T, 2N),
        X = randn(Xoshiro(3), T, 2N, NPTS))
end

"`J_n = [0 I; −I 0]`, in `T`, dense."
function symplectic_form(::Type{T}, n::Int) where {T}
    Iₙ = Matrix{T}(I, n, n)
    return [zeros(T, n, n) Iₙ; -Iₙ zeros(T, n, n)]
end

# The first Hamiltonian: the section's oscillator with `ω = 1`, `H(x) = ½xᵀx`, whose gradient is
# the identity. Its `∇H!` is elementwise, so it survives an `x` and `g` that are the same array.
oscillator!(g, t, x, params) = (g .= x)
oscillator_h(x) = sum(abs2, x) / 2

# The second Hamiltonian,
#
#   H(q, p) = ½|p|² + Σᵢ α(1 + i/N)(1 − cos qᵢ) + γ Σᵢ qᵢ p_{i+1} ,   cyclic,
#
# which is non-quadratic, asymmetric in `q` and `p`, index-dependent, coupled, and reads `params`.
function hamiltonian2(x, params)
    n = length(x) ÷ 2
    α = params.α
    γ = oftype(α, γ2)
    s = sum(abs2, view(x, (n + 1):2n)) / 2
    for i in 1:n
        s += α * oftype(α, 1 + i / n) * (1 - cos(x[i]))
    end
    return s + γ * sum(x[i] * x[n + mod1(i + 1, n)] for i in 1:n)
end

"""
The gradient of the second Hamiltonian: the `q`-half in a first loop, then the `p`-half in a
second loop that reads the `q`-half of `x`. A callable that writes `g` over `x` corrupts that
second loop, and an elementwise `∇H!` does not.
"""
function gradient2!(g, t, x, params)
    n = length(x) ÷ 2
    α = params.α
    γ = oftype(α, γ2)
    @inbounds for i in 1:n
        g[i] = α * oftype(α, 1 + i / n) * sin(x[i]) + γ * x[n + mod1(i + 1, n)]
    end
    @inbounds for i in 1:n
        g[n + i] = x[n + i] + γ * x[mod1(i - 1, n)]
    end
    return g
end

"""
A hyper-reduced system of `2n` coordinates, from a random basis and density basis.

Its `stencil` and `∇h` allocate nothing — `i -> i:i` and a `Tuple` — so an allocation reading of
the emitted callable is the callable's own code. The system's own reduced Hamiltonian is
differentiable, which is what `deim_reference` reads.
"""
function deim_system(::Type{T}, n::Int, k::Int, m::Int; columns::Int = 2k) where {T}
    K = zeros(T, n, n)
    for i in 1:n
        K[i, i] = 2
        K[i, mod1(i + 1, n)] -= 1
        K[mod1(i + 1, n), i] -= 1
    end
    L = [4K zeros(T, n, n); zeros(T, n, n) Matrix{T}(I, n, n)]
    A = randn(Xoshiro(11), T, 2n, columns)
    U = randn(Xoshiro(12), T, n, m)
    return SymplecticDEIM(A, L, U, i -> i:i, (xs, i) -> 1 - cos(xs[1]),
        (xs, i) -> (sin(xs[1]),))
end

"`J₂ₖ ∇H̃` of a hyper-reduced system, densely, from the system's own reduced Hamiltonian."
function deim_reference(sd, z)
    symplectic_form(eltype(z), length(z) ÷ 2) *
    ForwardDiff.gradient(sd, z)
end

"`‖w − ref‖_∞ / ‖ref‖_∞`, the relative bound of checks 4–6."
relative_error(w, ref) = norm(w - ref, Inf) / norm(ref, Inf)

"The reduced points of the 100 seeded states, as `project(b, x₀)` gives them."
reduced_points(b, X) = [project(b, X[:, n]) for n in 1:NPTS]

"`length(z)` `ForwardDiff` states of the type Newton's `JacobianAutodiff` passes."
function dual_state(::Type{T}, n::Int) where {T}
    [ForwardDiff.Dual{Nothing, T, 2}(T(0.3 + 0.01i), ForwardDiff.Partials((
         one(T), zero(T))))
     for i in 1:n]
end

"The time argument as the `Dual` of the same type."
function dual_time(::Type{T}) where {T}
    ForwardDiff.Dual{Nothing, T, 2}(zero(T), ForwardDiff.Partials((one(T), zero(T))))
end

"The reduced state of a PODE solution at step `n`, the two halves together."
pode_state(sol, n) = vcat(sol.q[n], sol.p[n])

"""
`@allocated` of one call of the ordinary field, after a warm call, through a fixed-arity barrier
whose arguments have concrete types.
"""
function field_allocations(v::V, w::W, t::TT, z::Z, params::P) where {V, W, TT, Z, P}
    v(w, t, z, params)
    return @allocated v(w, t, z, params)
end

"The positive control of that barrier: a callable of the same shape that must allocate."
control_field(w, t, z, params) = (collect(z); nothing)

"`@allocated` of one call of a partitioned field, through its own fixed-arity barrier."
function pode_allocations(
        v::V, w::W, t::TT, q::Q, p::P, params::PA) where {V, W, TT, Q, P, PA}
    v(w, t, q, p, params)
    return @allocated v(w, t, q, p, params)
end

"The positive control of the partitioned barrier."
control_pode(w, t, q, p, params) = (collect(q); nothing)

@testset "Reduced equations" begin
    # ---------------------------------------------------------------------------------------
    # Check 1: `ImplicitMidpoint` conserves a quadratic `H`, and every state is finite.
    # ---------------------------------------------------------------------------------------
    @testset "check 1: the symplectic method conserves the oscillator" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local w = integration_window(T)
            local b = ReducedBasis(CotangentLiftSVD(), f.S, NamedTuple(), testparamspace();
                k = 10)
            local z₀ = project(b, f.x₀)
            local prob = reduced_ode(b, oscillator!, w.span, w.dt, z₀)
            local sol = integrate(prob, ImplicitMidpoint();
                f_abstol = solver_tolerance(T, b.k))

            @test last(eachindex(sol.q)) == w.steps
            # The initial state is the reduced one, not dropped.
            @test sol.q[0] == z₀
            # The solver status is not read — `check_solver_status` is silent — so every saved
            # state is asserted finite, and the bound below is what guards convergence.
            @test all(isfinite, sol.q[0])

            local h₀ = oscillator_h(z₀)
            local worst = maximum(n -> abs(oscillator_h(sol.q[n]) - h₀) / h₀, eachindex(sol.q))
            # Only rounding (about `eps·steps`) and the solver's own tolerance remain.
            @test worst ≤ conservation_bound(T)
        end
    end

    # ---------------------------------------------------------------------------------------
    # Check 2: `ExplicitEuler` multiplies `H` by exactly `1 + dt²` a step.
    # ---------------------------------------------------------------------------------------
    @testset "check 2: the explicit method drifts by 1 + dt² a step" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local w = integration_window(T)
            local b = ReducedBasis(CotangentLiftSVD(), f.S, NamedTuple(), testparamspace();
                k = 10)
            local z₀ = project(b, f.x₀)
            local prob = reduced_ode(b, oscillator!, w.span, w.dt, z₀)
            local sol = integrate(prob, ExplicitEuler())
            local steps = last(eachindex(sol.q))

            @test steps == w.steps
            # The ratio is compared in `Float64`, against the Float64 of the Float32 `dt`.
            local ratio = Float64(oscillator_h(sol.q[steps]) / oscillator_h(z₀))
            local reference = (1 + Float64(w.dt)^2)^steps
            # The horizon separates the two methods by ten orders of magnitude.
            @test reference ≈ (T === Float32 ? 1.1051653876631566 : 2.7181459268249255) atol = 1e-9
            @test abs(ratio - reference) / reference ≤ conservation_bound(T)
        end
    end

    # ---------------------------------------------------------------------------------------
    # Check 3: the methods are selected by the trait, not by name.
    # ---------------------------------------------------------------------------------------
    @testset "check 3: the method is selected by its trait" begin
        @test issymplectic(ExplicitEuler()) === false
        @test issymplectic(ImplicitMidpoint()) === true
        @test ispodemethod(ExplicitEuler()) === false
        @test ispodemethod(ImplicitMidpoint()) === true

        # `issymplectic` is `missing` for a method without the trait, so `if issymplectic(m)`
        # would throw for that method. The selection compares with `true` and `false` instead.
        local conserving = only(filter(m -> issymplectic(m) === true, METHODS))
        local drifting = only(filter(m -> issymplectic(m) === false, METHODS))
        @test conserving === ImplicitMidpoint()
        @test drifting === ExplicitEuler()
    end

    # ---------------------------------------------------------------------------------------
    # Check 4: the field is `J_k Aᵀ ∇H(Az)`, on all five tags.
    # ---------------------------------------------------------------------------------------
    @testset "check 4: the field is the symplectic form" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local ps = params2(T)
            local JN = symplectic_form(T, N)

            for k in KS, tag in SYMPLECTIC_TAGS

                @testset "$(nameof(typeof(tag))), k = $k, $T" begin
                    local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = k)
                    local zs = reduced_points(b, f.X)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local v = equation(prob).v
                    # The basis is read from the problem's own callable, never built here.
                    local A = Matrix(v.basis)
                    local Jk = symplectic_form(T, k)

                    # The symplectic inverse identity holds inside the constructed problem.
                    @test norm(A' * JN * A - Jk, Inf) ≤ sqrt(eps(T))
                    # The complex basis really carries its imaginary part, so that a formula
                    # that drops `Ψ` is a change and not a coincidence.
                    tag isa ComplexSVD && @test norm(b.Ψ) > 0

                    # The emitted field equals `A⁺ J_N ∇H(Az)`, `A⁺ = J_kᵀ Aᵀ J_N`.
                    local ws = [zeros(T, length(z)) for z in zs]
                    for n in 1:NPTS
                        v(ws[n], 0.0, zs[n], ps)
                    end
                    local A⁺ = Jk' * A' * JN
                    local refs = [A⁺ *
                                  (JN * ForwardDiff.gradient(y -> hamiltonian2(y, ps),
                                      A * z)) for z in zs]
                    @test all(ref -> norm(ref, Inf) > 0, refs)
                    @test maximum(n -> relative_error(ws[n], refs[n]), 1:NPTS) ≤
                          sqrt(eps(T))
                end
            end

            for k in KS, tag in POD_TAGS

                @testset "$(nameof(typeof(tag))), k = $k, $T" begin
                    local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = k)
                    local zs = reduced_points(b, f.X)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local v = equation(prob).v
                    local U = Matrix(v.basis)

                    # A POD basis is not a symplectic one: at `k = 10`, where the shapes match,
                    # the identity fails. That is what shows the test reads the problem's basis.
                    if k == 10
                        @test norm(U' * JN * U - symplectic_form(T, 5), Inf) > sqrt(eps(T))
                    end

                    local ws = [zeros(T, length(z)) for z in zs]
                    for n in 1:NPTS
                        v(ws[n], 0.0, zs[n], ps)
                    end
                    local refs = [U' * (JN *
                                   ForwardDiff.gradient(y -> hamiltonian2(y, ps), U * z))
                                  for z in zs]
                    @test all(ref -> norm(ref, Inf) > 0, refs)
                    @test maximum(n -> relative_error(ws[n], refs[n]), 1:NPTS) ≤
                          sqrt(eps(T))
                end
            end
        end
    end

    # ---------------------------------------------------------------------------------------
    # Check 5: the split PODE variant — its two halves, its integration, and `[deps]`.
    # ---------------------------------------------------------------------------------------
    @testset "check 5: the split PODE form" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local ps = params2(T)

            for tag in SYMPLECTIC_TAGS
                @testset "$(nameof(typeof(tag))), $T" begin
                    local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = 10)
                    local k = b.k
                    local zs = reduced_points(b, f.X)
                    local ode = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local pode = reduced_pode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local reference = equation(ode).v
                    local eq = equation(pode)

                    # `v` at all 100 points first, `f` at all 100 afterwards: an `f` that reuses
                    # the last `v`'s gradient fails at 99 of them.
                    local vs = [zeros(T, k) for _ in 1:NPTS]
                    local fs = [zeros(T, k) for _ in 1:NPTS]
                    for n in 1:NPTS
                        eq.v(vs[n], 0.0, zs[n][1:k], zs[n][(k + 1):2k], ps)
                    end
                    for n in 1:NPTS
                        eq.f(fs[n], 0.0, zs[n][1:k], zs[n][(k + 1):2k], ps)
                    end

                    local ws = [zeros(T, 2k) for _ in 1:NPTS]
                    for n in 1:NPTS
                        reference(ws[n], 0.0, zs[n], ps)
                    end
                    @test all(n -> norm(ws[n], Inf) > 0, 1:NPTS)
                    @test maximum(n -> relative_error(vs[n], ws[n][1:k]), 1:NPTS) ≤
                          sqrt(eps(T))
                    @test maximum(n -> relative_error(fs[n], ws[n][(k + 1):2k]), 1:NPTS) ≤
                          sqrt(eps(T))

                    # The initial state is split, and compared exactly. `ImplicitMidpoint` is a
                    # PODE method; `ExplicitEuler` is not.
                    local z₀ = project(b, f.x₀)
                    local sol = integrate(pode, ImplicitMidpoint())
                    @test sol.q[0] == z₀[1:k]
                    @test sol.p[0] == z₀[(k + 1):2k]
                end
            end

            # `ImplicitMidpoint` on the PODE meets the bound of check 1.
            @testset "the PODE conserves the oscillator, $T" begin
                local w = integration_window(T)
                local b = ReducedBasis(CotangentLiftSVD(), f.S, NamedTuple(),
                    testparamspace(); k = 10)
                local z₀ = project(b, f.x₀)
                local prob = reduced_pode(b, oscillator!, w.span, w.dt, z₀)
                local sol = integrate(prob, ImplicitMidpoint();
                    f_abstol = solver_tolerance(T, b.k))
                local h₀ = oscillator_h(z₀)
                local worst = maximum(n -> abs(oscillator_h(pode_state(sol, n)) - h₀) / h₀,
                    eachindex(sol.q))
                @test worst ≤ conservation_bound(T)
            end
        end

        # `[deps]` holds GeometricEquations, and no integrator.
        @testset "the integrator is the caller's" begin
            local project_toml = TOML.parsefile(joinpath(pkgdir(ReducedBasisMethods),
                "Project.toml"))
            @test haskey(project_toml["deps"], "GeometricEquations")
            @test !haskey(project_toml["deps"], "GeometricIntegrators")
            @test !haskey(project_toml["deps"], "GeometricIntegratorsBase")
        end
    end

    # ---------------------------------------------------------------------------------------
    # Check 6: the POD form is the Galerkin form `ż = Uᵀ X_H(Uz)`.
    # ---------------------------------------------------------------------------------------
    @testset "check 6: the POD form is the Galerkin form" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local ps = params2(T)
            local JN = symplectic_form(T, N)

            # An odd `k = 5`: `J_5` does not exist, and the POD form asks for no `J_k`. And
            # `k = 10`, where one exists and the two forms differ. And the smallest, `k = 1`.
            for k in (5, 10, 1)
                @testset "PODSVD, k = $k, $T" begin
                    local b = ReducedBasis(
                        PODSVD(), f.S, NamedTuple(), testparamspace(); k = k)
                    local zs = reduced_points(b, f.X)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local v = equation(prob).v
                    local U = Matrix(v.basis)

                    local ws = [zeros(T, length(z)) for z in zs]
                    for n in 1:NPTS
                        v(ws[n], 0.0, zs[n], ps)
                    end
                    local refs = [U' * (JN *
                                   ForwardDiff.gradient(y -> hamiltonian2(y, ps), U * z))
                                  for z in zs]
                    @test all(ref -> norm(ref, Inf) > 0, refs)
                    @test maximum(n -> relative_error(ws[n], refs[n]), 1:NPTS) ≤
                          sqrt(eps(T))

                    if k == 10
                        # The parity branch the form must not take. `k = 10` is even, so a
                        # symplectic form of the reduced state exists, and it gives another
                        # field than the Galerkin one — which is what the assertion above
                        # discriminates against.
                        local Jhalf = symplectic_form(T, k ÷ 2)
                        local other = [Jhalf * (U' *
                                        ForwardDiff.gradient(y -> hamiltonian2(y, ps),
                                           U * z)) for z in zs]
                        @test maximum(n -> relative_error(other[n], refs[n]), 1:NPTS) >
                              sqrt(eps(T))
                    end
                end
            end

            # For a cotangent-lift `A`, `Aᵀ J_N = J_k Aᵀ`, so the two forms agree there, at
            # `k = 10` and at `k = 1`.
            for k in (10, 1)
                @testset "the two forms agree for a cotangent lift, k = $k, $T" begin
                    local b = ReducedBasis(CotangentLiftSVD(), f.S, NamedTuple(),
                        testparamspace(); k = k)
                    local zs = reduced_points(b, f.X)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local A = Matrix(equation(prob).v.basis)
                    local Jk = symplectic_form(T, k)
                    @test norm(A' * JN - Jk * A', Inf) ≤ sqrt(eps(T)) * norm(A' * JN, Inf)

                    local galerkin = [A' * (JN *
                                       ForwardDiff.gradient(y -> hamiltonian2(y, ps),
                                          A * z)) for z in zs]
                    local psd = [Jk * (A' *
                                  ForwardDiff.gradient(y -> hamiltonian2(y, ps), A * z))
                                 for z in zs]
                    @test all(ref -> norm(ref, Inf) > 0, psd)
                    @test maximum(n -> relative_error(galerkin[n], psd[n]), 1:NPTS) ≤
                          sqrt(eps(T))
                end
            end
        end
    end

    # ---------------------------------------------------------------------------------------
    # The guards the emitters owe, before GeometricEquations sees anything.
    # ---------------------------------------------------------------------------------------
    @testset "the guards" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local ps = params2(T)
            local cot = ReducedBasis(
                CotangentLiftSVD(), f.S, NamedTuple(), testparamspace();
                k = 10)
            local pod = ReducedBasis(PODSVD(), f.S, NamedTuple(), testparamspace(); k = 10)

            # A POD basis of an odd row count has no split into two halves for `J_N`.
            local odd = ReducedBasis(PODSVD(), randn(Xoshiro(1), T, 2N + 1, 30),
                NamedTuple(), testparamspace(); k = 5)
            @test size(odd.Φ, 1) == 2N + 1
            @test_throws ArgumentError reduced_ode(odd, gradient2!, (0.0, 1.0), 0.01,
                zeros(T, 5); parameters = ps)

            # A state of the wrong length would let a blockwise lift drop a surplus entry.
            @test_throws DimensionMismatch reduced_ode(cot, gradient2!, (0.0, 1.0), 0.01,
                zeros(T, 2 * cot.k + 1); parameters = ps)
            @test_throws DimensionMismatch reduced_ode(pod, gradient2!, (0.0, 1.0), 0.01,
                zeros(T, pod.k + 1); parameters = ps)
            @test_throws DimensionMismatch reduced_pode(cot, gradient2!, (0.0, 1.0), 0.01,
                zeros(T, 2 * cot.k + 1); parameters = ps)

            # A state of another element type is refused, and so is an integer state.
            local f32 = ReducedBasis(
                CotangentLiftSVD(), randn(Xoshiro(4), Float32, 2N, 30),
                NamedTuple(), testparamspace(); k = 10)
            @test_throws ArgumentError reduced_ode(f32, gradient2!, (0.0, 1.0), 0.01,
                zeros(Float64, 20); parameters = ps)
            @test_throws ArgumentError reduced_ode(cot, gradient2!, (0.0, 1.0), 0.01,
                zeros(Int, 2 * cot.k); parameters = ps)

            # A POD basis has no `q` and `p` halves, so it has no PODE form at all.
            @test_throws MethodError reduced_pode(pod, gradient2!, (0.0, 1.0), 0.01,
                zeros(T, pod.k); parameters = ps)
        end
    end

    # ---------------------------------------------------------------------------------------
    # A hyper-reduced system, emitted through `hamiltonian_gradient!`.
    # ---------------------------------------------------------------------------------------
    @testset "the hyper-reduced system" begin
        for T in (Float32, Float64)
            local k = 6
            local sd = deim_system(T, 32, k, 4)
            @test size(sd.A, 2) == 2k

            local zs = [randn(Xoshiro(100 + i), T, 2k) for i in 1:NPTS]
            local prob = reduced_ode(sd, (0.0, 1.0), 0.01, randn(Xoshiro(21), T, 2k))
            local v = equation(prob).v
            local ws = [zeros(T, 2k) for _ in 1:NPTS]
            for i in 1:NPTS
                v(ws[i], 0.0, zs[i], NullParameters())
            end
            local refs = [deim_reference(sd, z) for z in zs]
            @test all(ref -> norm(ref, Inf) > 0, refs)
            @test maximum(i -> relative_error(ws[i], refs[i]), 1:NPTS) ≤ sqrt(eps(T))

            # The PODE's halves are the halves of the same field.
            local pode = reduced_pode(sd, (0.0, 1.0), 0.01, randn(Xoshiro(22), T, 2k))
            local eq = equation(pode)
            local vs = [zeros(T, k) for _ in 1:NPTS]
            local fs = [zeros(T, k) for _ in 1:NPTS]
            for i in 1:NPTS
                eq.v(vs[i], 0.0, zs[i][1:k], zs[i][(k + 1):2k], NullParameters())
            end
            for i in 1:NPTS
                eq.f(fs[i], 0.0, zs[i][1:k], zs[i][(k + 1):2k], NullParameters())
            end
            @test maximum(i -> relative_error(vs[i], refs[i][1:k]), 1:NPTS) ≤ sqrt(eps(T))
            @test maximum(i -> relative_error(fs[i], refs[i][(k + 1):2k]), 1:NPTS) ≤
                  sqrt(eps(T))

            # An odd column count has no split into two halves for `J_2k`.
            local odd = deim_system(T, 32, k, 4; columns = 2k - 1)
            @test isodd(size(odd.A, 2))
            @test_throws ArgumentError reduced_ode(odd, (0.0, 1.0), 0.01, zeros(T, 2k - 1))
            @test_throws DimensionMismatch reduced_ode(sd, (0.0, 1.0), 0.01, zeros(T, 2k +
                                                                                      1))
        end
    end

    # ---------------------------------------------------------------------------------------
    # The callable is inferred and allocation-free, for `T` and for the `Dual` states Newton
    # passes.
    # ---------------------------------------------------------------------------------------
    @testset "the field allocates nothing" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local ps = params2(T)
            local cot = ReducedBasis(
                CotangentLiftSVD(), f.S, NamedTuple(), testparamspace();
                k = 10)
            local pod = ReducedBasis(PODSVD(), f.S, NamedTuple(), testparamspace(); k = 10)
            local k = cot.k
            local DualT = ForwardDiff.Dual{Nothing, T, 2}

            local ode = reduced_ode(cot, gradient2!, (0.0, 1.0), 0.01, project(cot, f.x₀);
                parameters = ps)
            local pode = reduced_pode(
                cot, gradient2!, (0.0, 1.0), 0.01, project(cot, f.x₀);
                parameters = ps)
            local galerkin = reduced_ode(
                pod, gradient2!, (0.0, 1.0), 0.01, project(pod, f.x₀);
                parameters = ps)
            local sd = deim_system(T, 32, 6, 4)
            local deim = reduced_ode(sd, (0.0, 1.0), 0.01, zeros(T, 12))

            local zt = fill(T(0.3), 2k)
            local zd = dual_state(T, 2k)
            local qt = fill(T(0.3), k)
            local qd = dual_state(T, k)
            local wt = zeros(T, 2k)
            local wd = zeros(DualT, 2k)
            local ut = zeros(T, k)
            local ud = zeros(DualT, k)
            local dur = dual_time(T)
            local st = fill(T(0.1), 12)
            local sd_dual = dual_state(T, 12)

            # The callback is a type parameter, so the call is inferred and statically
            # dispatched.
            @test @inferred(equation(ode).v(wt, 0.0, zt, ps)) === wt
            @test @inferred(equation(ode).v(wd, dur, zd, ps)) === wd
            @test @inferred(equation(galerkin).v(ut, 0.0, qt, ps)) === ut
            @test @inferred(equation(pode).v(ut, 0.0, qt, qt, ps)) === ut
            @test @inferred(equation(pode).f(ut, 0.0, qt, qt, ps)) === ut
            @test @inferred(equation(pode).v(ud, dur, qd, qd, ps)) === ud
            @test @inferred(equation(pode).f(ud, dur, qd, qd, ps)) === ud
            @test @inferred(equation(deim).v(wt, 0.0, st, NullParameters())) === wt
            @test @inferred(equation(deim).v(wd, dur, sd_dual, NullParameters())) === wd

            # Coverage instrumentation allocates on its own account, so the readings are taken
            # only where the runner does not instrument the code.
            if Base.JLOptions().code_coverage == 0
                # The control shows that the barrier sees an allocation at all.
                @test field_allocations(control_field, wt, 0.0, zt, ps) > 0
                @test field_allocations(equation(ode).v, wt, 0.0, zt, ps) == 0
                @test field_allocations(equation(ode).v, wd, dur, zd, ps) == 0
                @test field_allocations(equation(galerkin).v, ut, 0.0, qt, ps) == 0

                @test pode_allocations(control_pode, wt, 0.0, qt, qt, ps) > 0
                @test pode_allocations(equation(pode).v, ut, 0.0, qt, qt, ps) == 0
                @test pode_allocations(equation(pode).f, ut, 0.0, qt, qt, ps) == 0
                @test pode_allocations(equation(pode).v, ud, dur, qd, qd, ps) == 0
                @test pode_allocations(equation(pode).f, ud, dur, qd, qd, ps) == 0

                @test field_allocations(equation(deim).v, wt, 0.0, st, NullParameters()) ==
                      0
                @test field_allocations(equation(deim).v, wd, dur, sd_dual,
                    NullParameters()) == 0
            end
        end
    end

    # ---------------------------------------------------------------------------------------
    # The exported names carry their docstrings.
    # ---------------------------------------------------------------------------------------
    @testset "the exported docstrings" begin
        local texts = (reduced_ode = string(@doc(ReducedBasisMethods.reduced_ode)),
            reduced_pode = string(@doc(ReducedBasisMethods.reduced_pode)))
        for (name, text) in pairs(texts)
            @testset "$name" begin
                @test !isempty(text)
                @test occursin("∇H!(g, t, x, params)", text)
                @test occursin("ArgumentError", text)
                @test occursin("DimensionMismatch", text)
                @test occursin("0 B", text)
                @test occursin("a callback's allocations are the caller's", text)
                @test occursin("tasks", text)         # the concurrency limit
                @test occursin("```julia", text)      # a plain example, never a jldoctest
                @test !occursin("jldoctest", text)
                for tag in TAGS
                    @test occursin(string(nameof(typeof(tag))), text)
                end
                for word in ("Vlasov", "plasma", "particle")
                    @test !occursin(word, text)
                end
            end
        end
    end
end
