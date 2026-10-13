#
# The reduced equation forms: the symplectic form `ż = J_k Aᵀ ∇H(Az)`, the Galerkin form
# `ż = Uᵀ X_H(Uz)` of a POD basis, and the split PODE variant, emitted as `GeometricEquations`
# problems.
#
# Integration checks use GeometricIntegrators' own `integrate`. The field references
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

# The subset of them the `Dual` value checks use: each is a nested-dual gradient of `H`, not
# one pass of the field, so only ten points use this more expensive reference.
const NPTS_DUAL = 10

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

"The solver tolerance: `1e-14` in `Float64`, or `max(8, 2k)·eps` in `Float32`."
solver_tolerance(::Type{Float64}, k::Int) = 1e-14
solver_tolerance(::Type{Float32}, k::Int) = max(8, 2k) * eps(Float32)

"""
The seeded snapshot and the initial state, in `T`.

`S` is the `2N × 30` stacked snapshot `[Q; P]`, which every tag accepts: a POD basis of it is
`2N × k`, and a cotangent-lift or complex basis is `N × k` with a `2N × 2k` assembly.
"""
function fixtures(::Type{T}) where {T}
    (; S = randn(Xoshiro(1), T, 2N, 30), x₀ = randn(Xoshiro(2), T, 2N))
end

"`J_n = [0 I; −I 0]`, in `T`, dense."
function symplectic_form(::Type{T}, n::Int) where {T}
    Iₙ = Matrix{T}(I, n, n)
    return [zeros(T, n, n) Iₙ; -Iₙ zeros(T, n, n)]
end

# The oscillator has `ω = 1` and `H(x) = ½xᵀx`, whose gradient is the identity. Its `∇H!` is
# elementwise, so it survives an `x` and `g` that are the same array.
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

"The 100 seeded reduced points: a direct draw in reduced coordinates."
function reduced_points(::Type{T}, n::Int) where {T}
    rng = Xoshiro(3)
    return [randn(rng, T, n) for _ in 1:NPTS]
end

"""
`z`, lifted to `ForwardDiff` states of the type Newton's `JacobianAutodiff` passes, with both
partials non-zero and varying over the components.
"""
function dual_state(z::AbstractVector{T}) where {T}
    n = length(z)
    return [ForwardDiff.Dual{Nothing, T, 2}(z[i],
                ForwardDiff.Partials((T(1 + i) / n, -T(i) / (2n)))) for i in 1:n]
end

"`ForwardDiff.value` of every element of a `Dual` state."
dual_values(w) = ForwardDiff.value.(w)

"Partial `p` of every element of a `Dual` state."
dual_partials(w, p) = [ForwardDiff.partials(x, p) for x in w]

"""
The dense reference field of checks 4–6: `L·J_N·∇H(Az)`, at whatever element type `z` carries.

`L` is `A⁺ = J_kᵀAᵀJ_N` for a symplectic basis and `Uᵀ` for a POD one. At a `Dual` state the
reference is itself `Dual`-valued — `ForwardDiff.gradient` of a `Dual` input returns the
gradient in that input's own type — so its `ForwardDiff.value` is the field's value and its
partials are the field's derivatives, and one call checks both.
"""
function reference_field(L, JN, A, z, ps)
    L * (JN * ForwardDiff.gradient(y -> hamiltonian2(y, ps), A * z))
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
    @testset "check 1: ImplicitMidpoint conserves the oscillator" begin
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
            @test all(n -> all(isfinite, sol.q[n]), eachindex(sol.q))

            local h₀ = oscillator_h(z₀)
            local worst = maximum(n -> abs(oscillator_h(sol.q[n]) - h₀) / h₀, eachindex(sol.q))
            # Only rounding (about `eps·steps`) and the solver's own tolerance remain.
            @test worst ≤ conservation_bound(T)
        end
    end

    # ---------------------------------------------------------------------------------------
    # Check 2: `ExplicitEuler` multiplies `H` by exactly `1 + dt²` a step.
    # ---------------------------------------------------------------------------------------
    @testset "check 2: ExplicitEuler multiplies H by 1 + dt² a step" begin
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
            # The horizons give distinct energy growth factors in the two element types.
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
    # Check 4: the symplectic and Galerkin fields, on their respective tags.
    # ---------------------------------------------------------------------------------------
    @testset "check 4: the symplectic and Galerkin vector fields" begin
        for T in (Float32, Float64)
            local f = fixtures(T)
            local ps = params2(T)
            local JN = symplectic_form(T, N)

            for k in KS, tag in SYMPLECTIC_TAGS

                @testset "$(nameof(typeof(tag))), k = $k, $T" begin
                    local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = k)
                    local zs = reduced_points(T, 2k)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local v = equation(prob).v
                    # The integrator calls the field with `parameters(prob)`, and the test does
                    # too: an emitter that dropped the caller's object fails here.
                    local pp = parameters(prob)
                    @test pp === ps
                    # The basis is read from the problem's own callable, never built here.
                    local A = Matrix(v.basis)
                    local Jk = symplectic_form(T, k)
                    local A⁺ = Jk' * A' * JN

                    # The symplectic inverse identity holds inside the constructed problem.
                    @test norm(A' * JN * A - Jk, Inf) ≤ sqrt(eps(T))
                    # The complex basis really carries its imaginary part, so that a formula
                    # that drops `Ψ` is a change and not a coincidence.
                    tag isa ComplexSVD && @test norm(b.Ψ) > 0

                    # The emitted field equals `A⁺ J_N ∇H(Az)`.
                    local ws = [zeros(T, length(z)) for z in zs]
                    for n in 1:NPTS
                        v(ws[n], 0.0, zs[n], pp)
                    end
                    local refs = [reference_field(A⁺, JN, A, z, ps) for z in zs]
                    @test all(ref -> norm(ref, Inf) > 0, refs)
                    @test maximum(n -> relative_error(ws[n], refs[n]), 1:NPTS) ≤
                          sqrt(eps(T))

                    # The same field at the `Dual` states Newton's `JacobianAutodiff` passes: a
                    # callable whose lift and gradient buffers are one array is wrong here, in
                    # value and in partials. The reference is `Dual`-valued, so its
                    # `ForwardDiff.value` carries the value and its partials the derivatives.
                    local DualT = ForwardDiff.Dual{Nothing, T, 2}
                    for n in 1:NPTS_DUAL
                        local zd = dual_state(zs[n])
                        local wd = zeros(DualT, 2k)
                        local refd = reference_field(A⁺, JN, A, zd, ps)
                        v(wd, 0.0, zd, pp)
                        @test norm(ForwardDiff.value.(refd), Inf) > 0
                        @test relative_error(dual_values(wd), dual_values(refd)) ≤
                              sqrt(eps(T))
                        for p in 1:2
                            @test relative_error(dual_partials(wd, p),
                                dual_partials(refd, p)) ≤ sqrt(eps(T))
                        end
                    end
                end
            end

            for k in KS, tag in POD_TAGS

                @testset "$(nameof(typeof(tag))), k = $k, $T" begin
                    local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = k)
                    local zs = reduced_points(T, k)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local v = equation(prob).v
                    local pp = parameters(prob)
                    @test pp === ps
                    local U = Matrix(v.basis)

                    # A POD basis is not a symplectic one: at `k = 10`, where the shapes match,
                    # the identity fails. That is what shows the test reads the problem's basis.
                    if k == 10
                        @test norm(U' * JN * U - symplectic_form(T, 5), Inf) > sqrt(eps(T))
                    end

                    local ws = [zeros(T, length(z)) for z in zs]
                    for n in 1:NPTS
                        v(ws[n], 0.0, zs[n], pp)
                    end
                    local refs = [reference_field(U', JN, U, z, ps) for z in zs]
                    @test all(ref -> norm(ref, Inf) > 0, refs)
                    @test maximum(n -> relative_error(ws[n], refs[n]), 1:NPTS) ≤
                          sqrt(eps(T))

                    # The Galerkin field at the `Dual` states too.
                    local DualT = ForwardDiff.Dual{Nothing, T, 2}
                    for n in 1:NPTS_DUAL
                        local zd = dual_state(zs[n])
                        local wd = zeros(DualT, k)
                        local refd = reference_field(U', JN, U, zd, ps)
                        v(wd, 0.0, zd, pp)
                        @test norm(ForwardDiff.value.(refd), Inf) > 0
                        @test relative_error(dual_values(wd), dual_values(refd)) ≤
                              sqrt(eps(T))
                        for p in 1:2
                            @test relative_error(dual_partials(wd, p),
                                dual_partials(refd, p)) ≤ sqrt(eps(T))
                        end
                    end
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
                    local zs = reduced_points(T, 2k)
                    local z₀ = project(b, f.x₀)
                    local ode = reduced_ode(b, gradient2!, (0.0, 1.0), 0.01, z₀;
                        parameters = ps)
                    local pode = reduced_pode(b, gradient2!, (0.0, 1.0), 0.01, z₀;
                        parameters = ps)
                    local reference = equation(ode).v
                    local eq = equation(pode)
                    # The integrator hands both fields `parameters(pode)`, and the test does too.
                    local pp = parameters(pode)
                    @test pp === ps

                    # `v` at all 100 points first, `f` at all 100 afterwards: an `f` that reuses
                    # the last `v`'s gradient fails at 99 of them.
                    local vs = [zeros(T, k) for _ in 1:NPTS]
                    local fs = [zeros(T, k) for _ in 1:NPTS]
                    for n in 1:NPTS
                        eq.v(vs[n], 0.0, zs[n][1:k], zs[n][(k + 1):2k], pp)
                    end
                    for n in 1:NPTS
                        eq.f(fs[n], 0.0, zs[n][1:k], zs[n][(k + 1):2k], pp)
                    end

                    local ws = [zeros(T, 2k) for _ in 1:NPTS]
                    for n in 1:NPTS
                        reference(ws[n], 0.0, zs[n], pp)
                    end
                    @test all(n -> norm(ws[n], Inf) > 0, 1:NPTS)
                    @test maximum(n -> relative_error(vs[n], ws[n][1:k]), 1:NPTS) ≤
                          sqrt(eps(T))
                    @test maximum(n -> relative_error(fs[n], ws[n][(k + 1):2k]), 1:NPTS) ≤
                          sqrt(eps(T))

                    # The two halves at the `Dual` states too, against the dense reference: a
                    # callable whose lift and gradient buffers are one array is wrong here.
                    local JN = symplectic_form(T, N)
                    local A = Matrix(reference.basis)
                    local A⁺ = symplectic_form(T, k)' * A' * JN
                    local DualT = ForwardDiff.Dual{Nothing, T, 2}
                    for n in 1:NPTS_DUAL
                        local zd = dual_state(zs[n])
                        local refd = reference_field(A⁺, JN, A, zd, ps)
                        local vd = zeros(DualT, k)
                        local fd = zeros(DualT, k)
                        eq.v(vd, 0.0, zd[1:k], zd[(k + 1):2k], pp)
                        eq.f(fd, 0.0, zd[1:k], zd[(k + 1):2k], pp)
                        @test norm(dual_values(refd), Inf) > 0
                        @test relative_error(dual_values(vd), dual_values(refd)[1:k]) ≤
                              sqrt(eps(T))
                        @test relative_error(dual_values(fd),
                            dual_values(refd)[(k + 1):2k]) ≤ sqrt(eps(T))
                        for p in 1:2
                            @test relative_error(dual_partials(vd, p),
                                dual_partials(refd, p)[1:k]) ≤ sqrt(eps(T))
                            @test relative_error(dual_partials(fd, p),
                                dual_partials(refd, p)[(k + 1):2k]) ≤ sqrt(eps(T))
                        end
                    end

                    # The initial state is split, and compared exactly. `ImplicitMidpoint` is a
                    # PODE method; `ExplicitEuler` is not. The integration uses the explicit
                    # `f_abstol` from `solver_tolerance`, rather than a default.
                    local sol = integrate(pode, ImplicitMidpoint();
                        f_abstol = solver_tolerance(T, k))
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
                # Every saved state of both halves is finite, as in check 1.
                @test all(n -> all(isfinite, sol.q[n]), eachindex(sol.q))
                @test all(n -> all(isfinite, sol.p[n]), eachindex(sol.p))
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

            # An odd reduced dimension `k = 5` has no canonical symplectic matrix of size `k × k`.
            # At `k = 10`, that matrix exists and the two forms differ; `k = 1` is the smallest.
            # Both POD tags run, so a branch only one of them reaches is still caught.
            for k in (5, 10, 1), tag in POD_TAGS

                @testset "$(nameof(typeof(tag))), k = $k, $T" begin
                    local b = ReducedBasis(
                        tag, f.S, NamedTuple(), testparamspace(); k = k)
                    local zs = reduced_points(T, k)
                    local prob = reduced_ode(
                        b, gradient2!, (0.0, 1.0), 0.01, project(b, f.x₀);
                        parameters = ps)
                    local v = equation(prob).v
                    local pp = parameters(prob)
                    local U = Matrix(v.basis)

                    local ws = [zeros(T, length(z)) for z in zs]
                    for n in 1:NPTS
                        v(ws[n], 0.0, zs[n], pp)
                    end
                    local refs = [reference_field(U', JN, U, z, ps) for z in zs]
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
                    local zs = reduced_points(T, 2k)
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

            # A state of another element type is refused, and so is an integer state: the PSD
            # emitters here, the POD ones too.
            local f32 = ReducedBasis(
                CotangentLiftSVD(), randn(Xoshiro(4), Float32, 2N, 30),
                NamedTuple(), testparamspace(); k = 10)
            @test_throws ArgumentError reduced_ode(f32, gradient2!, (0.0, 1.0), 0.01,
                zeros(Float64, 20); parameters = ps)
            @test_throws ArgumentError reduced_ode(cot, gradient2!, (0.0, 1.0), 0.01,
                zeros(Int, 2 * cot.k); parameters = ps)
            local pod32 = ReducedBasis(
                PODSVD(), randn(Xoshiro(4), Float32, 2N, 30),
                NamedTuple(), testparamspace(); k = 10)
            @test_throws ArgumentError reduced_ode(pod32, gradient2!, (0.0, 1.0), 0.01,
                zeros(Float64, pod32.k); parameters = ps)
            @test_throws ArgumentError reduced_ode(pod, gradient2!, (0.0, 1.0), 0.01,
                zeros(Int, pod.k); parameters = ps)

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
            local J2k = symplectic_form(T, k)
            local z₀ = randn(Xoshiro(21), T, 2k)

            # The system's own element type and state length are guarded.
            local other = T === Float64 ? Float32 : Float64
            @test_throws ArgumentError reduced_ode(sd, (0.0, 1.0), 0.01, zeros(other, 2k))
            @test_throws ArgumentError reduced_ode(sd, (0.0, 1.0), 0.01, zeros(Int, 2k))

            local zs = reduced_points(T, 2k)
            local prob = reduced_ode(sd, (0.0, 1.0), 0.01, z₀)
            local v = equation(prob).v
            local ws = [zeros(T, 2k) for _ in 1:NPTS]
            for i in 1:NPTS
                v(ws[i], 0.0, zs[i], NullParameters())
            end
            local refs = [deim_reference(sd, z) for z in zs]
            @test all(ref -> norm(ref, Inf) > 0, refs)
            @test maximum(i -> relative_error(ws[i], refs[i]), 1:NPTS) ≤ sqrt(eps(T))

            # The same field at the `Dual` states Newton's `JacobianAutodiff` passes:
            # `hamiltonian_gradient!` gives no guarantee for an aliased array, so a callable
            # that hands it its own lift buffer is wrong here, in value and in partials.
            local DualT = ForwardDiff.Dual{Nothing, T, 2}
            for i in 1:NPTS_DUAL
                local zd = dual_state(zs[i])
                local wd = zeros(DualT, 2k)
                local refd = J2k * ForwardDiff.gradient(sd, zd)
                v(wd, 0.0, zd, NullParameters())
                @test norm(ForwardDiff.value.(refd), Inf) > 0
                @test relative_error(dual_values(wd), dual_values(refd)) ≤ sqrt(eps(T))
                for p in 1:2
                    @test relative_error(dual_partials(wd, p), dual_partials(refd, p)) ≤
                          sqrt(eps(T))
                end
            end

            # The initial state is the reduced one, exactly; the PODE's is split. Both
            # integrations use the explicit `f_abstol` from `solver_tolerance`, as the others do.
            local sol = integrate(
                reduced_ode(sd, (0.0, 0.1), 0.01, z₀), ImplicitMidpoint();
                f_abstol = solver_tolerance(T, k))
            @test sol.q[0] == z₀
            local pode = reduced_pode(sd, (0.0, 1.0), 0.01, z₀)
            local eq = equation(pode)
            local solpode = integrate(reduced_pode(sd, (0.0, 0.1), 0.01, z₀),
                ImplicitMidpoint(); f_abstol = solver_tolerance(T, k))
            @test solpode.q[0] == z₀[1:k]
            @test solpode.p[0] == z₀[(k + 1):2k]

            # The PODE's halves are the halves of the same field, and the two halves at the
            # `Dual` states are the halves of that reference.
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

            for i in 1:NPTS_DUAL
                local zd = dual_state(zs[i])
                local refd = J2k * ForwardDiff.gradient(sd, zd)
                local vd = zeros(DualT, k)
                local fd = zeros(DualT, k)
                eq.v(vd, 0.0, zd[1:k], zd[(k + 1):2k], NullParameters())
                eq.f(fd, 0.0, zd[1:k], zd[(k + 1):2k], NullParameters())
                @test relative_error(dual_values(vd), dual_values(refd)[1:k]) ≤
                      sqrt(eps(T))
                @test relative_error(dual_values(fd), dual_values(refd)[(k + 1):2k]) ≤
                      sqrt(eps(T))
                for p in 1:2
                    @test relative_error(dual_partials(vd, p),
                        dual_partials(refd, p)[1:k]) ≤ sqrt(eps(T))
                    @test relative_error(dual_partials(fd, p),
                        dual_partials(refd, p)[(k + 1):2k]) ≤ sqrt(eps(T))
                end
            end

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
            local k = 10
            local DualT = ForwardDiff.Dual{Nothing, T, 2}
            local dur = dual_time(T)
            local ct = fill(T(0.3), 2k)

            # Coverage instrumentation allocates on its own account, so the readings are taken
            # only where the runner does not instrument the code. The control shows that the
            # barrier sees an allocation at all.
            if Base.JLOptions().code_coverage == 0
                @test field_allocations(control_field, zeros(T, 2k), 0.0, ct, ps) > 0
                @test pode_allocations(control_pode, zeros(T, 2k), 0.0, ct, ct, ps) > 0
            end

            # Every emitted field: the ODE `v` of all five tags, the PODE `v` and `f` of the
            # three symplectic ones, and the hyper-reduced system's ODE `v`, PODE `v` and `f`. The
            # callback is a type parameter, so each is inferred and statically dispatched, and
            # each reads 0 B after a warm call, for `T` and for `Dual`.
            for tag in TAGS
                local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = k)
                local n = length(project(b, f.x₀))
                local vf = equation(reduced_ode(b, gradient2!, (0.0, 1.0), 0.01,
                    project(b, f.x₀); parameters = ps)).v
                local zt = fill(T(0.3), n)
                local zd = dual_state(zt)
                local wt = zeros(T, n)
                local wd = zeros(DualT, n)

                @testset "$(nameof(typeof(tag)))" begin
                    @test @inferred(vf(wt, 0.0, zt, ps)) === wt
                    @test @inferred(vf(wd, dur, zd, ps)) === wd
                    if Base.JLOptions().code_coverage == 0
                        @test field_allocations(vf, wt, 0.0, zt, ps) == 0
                        @test field_allocations(vf, wd, dur, zd, ps) == 0
                    end
                end
            end

            for tag in SYMPLECTIC_TAGS
                local b = ReducedBasis(tag, f.S, NamedTuple(), testparamspace(); k = k)
                local eq = equation(reduced_pode(b, gradient2!, (0.0, 1.0), 0.01,
                    project(b, f.x₀); parameters = ps))
                local qt = fill(T(0.3), k)
                local qd = dual_state(qt)
                local ut = zeros(T, k)
                local ud = zeros(DualT, k)

                @testset "$(nameof(typeof(tag))) PODE" begin
                    @test @inferred(eq.v(ut, 0.0, qt, qt, ps)) === ut
                    @test @inferred(eq.f(ut, 0.0, qt, qt, ps)) === ut
                    @test @inferred(eq.v(ud, dur, qd, qd, ps)) === ud
                    @test @inferred(eq.f(ud, dur, qd, qd, ps)) === ud
                    if Base.JLOptions().code_coverage == 0
                        @test pode_allocations(eq.v, ut, 0.0, qt, qt, ps) == 0
                        @test pode_allocations(eq.f, ut, 0.0, qt, qt, ps) == 0
                        @test pode_allocations(eq.v, ud, dur, qd, qd, ps) == 0
                        @test pode_allocations(eq.f, ud, dur, qd, qd, ps) == 0
                    end
                end
            end

            @testset "the hyper-reduced system" begin
                local ksd = 6
                local sd = deim_system(T, 32, ksd, 4)
                local st = fill(T(0.1), 2ksd)
                local sd_dual = dual_state(st)
                local wt = zeros(T, 2ksd)
                local wd = zeros(DualT, 2ksd)
                local ut = zeros(T, ksd)
                local ud = zeros(DualT, ksd)
                local df = equation(reduced_ode(sd, (0.0, 1.0), 0.01, st))
                local dp = equation(reduced_pode(sd, (0.0, 1.0), 0.01, st))

                @test @inferred(df.v(wt, 0.0, st, NullParameters())) === wt
                @test @inferred(df.v(wd, dur, sd_dual, NullParameters())) === wd
                @test @inferred(dp.v(ut, 0.0, st[1:ksd], st[(ksd + 1):2ksd],
                    NullParameters())) === ut
                @test @inferred(dp.f(ut, 0.0, st[1:ksd], st[(ksd + 1):2ksd],
                    NullParameters())) === ut
                @test @inferred(dp.v(ud, dur, sd_dual[1:ksd], sd_dual[(ksd + 1):2ksd],
                    NullParameters())) === ud
                @test @inferred(dp.f(ud, dur, sd_dual[1:ksd], sd_dual[(ksd + 1):2ksd],
                    NullParameters())) === ud
                if Base.JLOptions().code_coverage == 0
                    @test field_allocations(df.v, wt, 0.0, st, NullParameters()) == 0
                    @test field_allocations(df.v, wd, dur, sd_dual, NullParameters()) == 0
                    @test pode_allocations(dp.v, ut, 0.0, st[1:ksd], st[(ksd + 1):2ksd],
                        NullParameters()) == 0
                    @test pode_allocations(dp.f, ut, 0.0, st[1:ksd], st[(ksd + 1):2ksd],
                        NullParameters()) == 0
                    @test pode_allocations(dp.v, ud, dur, sd_dual[1:ksd],
                        sd_dual[(ksd + 1):2ksd], NullParameters()) == 0
                    @test pode_allocations(dp.f, ud, dur, sd_dual[1:ksd],
                        sd_dual[(ksd + 1):2ksd], NullParameters()) == 0
                end
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
