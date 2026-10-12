# The reduced equation forms, emitted as `GeometricEquations` problems. Integration is the
# caller's: no integrator is a dependency of this package.
#
# The callable the integrator calls holds the basis (or the hyper-reduced system), the caller's
# `∇H!` as a type parameter, and its work buffers. It subtypes `Function` because
# `GeometricEquations.ODE` and `PODE` store their vector fields behind a `Callable` bound.

"""
The shape of the emitted vector field: the whole reduced state, or one half of a partitioned one.
"""
abstract type _ReducedForm end
struct _ODEForm <: _ReducedForm end
struct _VelocityForm <: _ReducedForm end
struct _ForceForm <: _ReducedForm end

"""
    _ReducedVectorField{F, BT, T, H, X, G, C} <: Function

The vector field of an emitted reduced problem, in the form `F ∈ (_ODEForm, _VelocityForm,
_ForceForm)`.

It holds the basis `BT` — a [`ReducedBasis`](@ref), or the [`SymplecticDEIM`](@ref) whose system
is already in reduced coordinates — the caller's `∇H!` as the type parameter `H`, and its work
buffers: `x = A·z` and `g = ∇H(x)` in the basis's element type `T`, which are distinct because
`∇H!` may read `x` after it writes `g`. A state of any other element type `S` — which is what
Newton's `JacobianAutodiff` passes — takes its pair from `C`, an `IdDict` keyed by `S`.

The basis is a field and not an entry of the caller's `parameters`, so `parameters` reaches
`∇H!` unchanged.
"""
struct _ReducedVectorField{F <: _ReducedForm, BT, T, H, X <: AbstractVector{T},
    G <: AbstractVector{T}, C <: IdDict{DataType, Any}} <: Function
    basis::BT
    ∇H!::H
    x::X
    g::G
    cache::C
end

function _reduced_vector_field(
        basis, ∇H!, ::Type{T}, n::Int, ::F) where {T, F <: _ReducedForm}
    x = Vector{T}(undef, n)
    g = Vector{T}(undef, n)
    cache = IdDict{DataType, Any}()
    return _ReducedVectorField{F, typeof(basis), T, typeof(∇H!), typeof(x), typeof(g),
        typeof(cache)}(basis, ∇H!, x, g, cache)
end

# The `T` pair is the callable's own; any other element type takes a pair from the per-type
# cache, so that only its first call allocates. The pair is one entry, so a lookup cannot hand
# the lift buffer back as the gradient buffer.
@inline function _work(vf::_ReducedVectorField{F, BT, T}, ::Type{S}) where {F, BT, T, S}
    S === T && return vf.x, vf.g
    return _cached_pair(vf.cache, vf.x, S)
end

@inline function _cached_pair(cache::IdDict{DataType, Any}, x::Vector{T},
        ::Type{S}) where {T, S}
    if haskey(cache, S)
        return cache[S]::Tuple{Vector{S}, Vector{S}}
    else
        pair = (similar(x, S), similar(x, S))
        return cache[S] = pair::Tuple{Vector{S}, Vector{S}}
    end
end

# `x = A·z`, blockwise over the two halves of a symplectic basis.
@inline function _lift!(x::AbstractVector,
        b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        z₁::AbstractVector, z₂::AbstractVector)
    N = size(b.Φ, 1)
    mul!(view(x, 1:N), b.Φ, z₁)
    mul!(view(x, (N + 1):2N), b.Φ, z₂)
    return x
end

@inline function _lift!(x::AbstractVector, b::ReducedBasis{<:Any, <:ComplexSVD},
        z₁::AbstractVector, z₂::AbstractVector)
    N = size(b.Φ, 1)
    mul!(view(x, 1:N), b.Φ, z₁)
    mul!(view(x, 1:N), b.Ψ, z₂, -1, 1)
    mul!(view(x, (N + 1):2N), b.Ψ, z₁)
    mul!(view(x, (N + 1):2N), b.Φ, z₂, 1, 1)
    return x
end

# A POD basis is one block, and its reduced state is not split.
@inline _lift!(x::AbstractVector, b::ReducedBasis{<:Any, <:PODAlgorithm}, z::AbstractVector) = mul!(x, b.Φ, z)

# A hyper-reduced system is already in reduced coordinates.
@inline _lift!(x::AbstractVector, ::SymplecticDEIM, z::AbstractVector) = copyto!(x, z)

@inline function _lift!(x::AbstractVector, ::SymplecticDEIM, q::AbstractVector,
        p::AbstractVector)
    k = length(q)
    copyto!(view(x, 1:k), q)
    copyto!(view(x, (k + 1):2k), p)
    return x
end

@inline function _lift!(x::AbstractVector, b::ReducedBasis{<:Any, <:SymplecticAlgorithm},
        z::AbstractVector)
    k = b.k
    return _lift!(x, b, view(z, 1:k), view(z, (k + 1):2k))
end

# The gradient at the lifted point: the caller's `∇H!`.
@inline function _gradient!(g::AbstractVector, vf::_ReducedVectorField{F, BT}, t, x,
        params) where {F, BT <: ReducedBasis}
    vf.∇H!(g, t, x, params)
    return g
end

# The reduced gradient of a hyper-reduced system, whose system's own write is the gradient.
@inline function _gradient!(g::AbstractVector, vf::_ReducedVectorField{F, BT}, t, x,
        params) where {F, BT <: SymplecticDEIM}
    vf.∇H!(g, vf.basis, x)
    return g
end

# The first and the last `k` rows of `J_k Aᵀ g`, `k = size(Φ, 2)`.
@inline function _half!(out::AbstractVector,
        b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        g::AbstractVector, ::Val{1})
    N = size(b.Φ, 1)
    return mul!(out, adjoint(b.Φ), view(g, (N + 1):2N))
end

@inline function _half!(out::AbstractVector,
        b::ReducedBasis{<:Any, <:Union{CotangentLiftSVD, CotangentLiftEVD}},
        g::AbstractVector, ::Val{2})
    N = size(b.Φ, 1)
    mul!(out, adjoint(b.Φ), view(g, 1:N))
    rmul!(out, -1)
    return out
end

@inline function _half!(out::AbstractVector, b::ReducedBasis{<:Any, <:ComplexSVD},
        g::AbstractVector, ::Val{1})
    N = size(b.Φ, 1)
    mul!(out, adjoint(b.Φ), view(g, (N + 1):2N))
    mul!(out, adjoint(b.Ψ), view(g, 1:N), -1, 1)
    return out
end

@inline function _half!(out::AbstractVector, b::ReducedBasis{<:Any, <:ComplexSVD},
        g::AbstractVector, ::Val{2})
    N = size(b.Φ, 1)
    mul!(out, adjoint(b.Φ), view(g, 1:N))
    mul!(out, adjoint(b.Ψ), view(g, (N + 1):2N), 1, 1)
    rmul!(out, -1)
    return out
end

# The reduced gradient of the hyper-reduced system, which reads no basis.
@inline function _half!(out::AbstractVector, sd::SymplecticDEIM, g::AbstractVector, ::Val{1})
    k = size(sd.A, 2) ÷ 2
    return copyto!(out, view(g, (k + 1):2k))
end

@inline function _half!(out::AbstractVector, sd::SymplecticDEIM, g::AbstractVector, ::Val{2})
    k = size(sd.A, 2) ÷ 2
    copyto!(out, view(g, 1:k))
    rmul!(out, -1)
    return out
end

"The half-size `k` of the reduced state: the basis's `k`, and the DEIM basis's column count ÷ 2."
_half_size(b::ReducedBasis) = b.k
_half_size(sd::SymplecticDEIM) = size(sd.A, 2) ÷ 2

"The lifted dimension: `2N` for a symplectic basis, `N` for a POD basis, `2k` for a DEIM system."
_lifted_dimension(b::ReducedBasis{<:Any, <:SymplecticAlgorithm}) = 2 * size(b.Φ, 1)
_lifted_dimension(b::ReducedBasis{<:Any, <:PODAlgorithm}) = size(b.Φ, 1)
_lifted_dimension(sd::SymplecticDEIM) = size(sd.A, 2)

# The whole reduced vector field: `ż = J_k Aᵀ ∇H(Az)` for a symplectic basis and
# `ż = J₂ₖ ∇H̃(z)` for a hyper-reduced system, which write their halves through the same
# dispatch, or the Galerkin form of a POD basis.
@inline function _field!(ω::AbstractVector,
        b::Union{ReducedBasis{<:Any, <:SymplecticAlgorithm}, SymplecticDEIM},
        g::AbstractVector)
    k = _half_size(b)
    _half!(view(ω, 1:k), b, g, Val(1))
    _half!(view(ω, (k + 1):2k), b, g, Val(2))
    return ω
end

@inline function _field!(ω::AbstractVector, b::ReducedBasis{<:Any, <:PODAlgorithm},
        g::AbstractVector)
    N = size(b.Φ, 1) ÷ 2
    mul!(ω, adjoint(view(b.Φ, 1:N, :)), view(g, (N + 1):2N))
    mul!(ω, adjoint(view(b.Φ, (N + 1):2N, :)), view(g, 1:N), -1, 1)
    return ω
end

@inline function (vf::_ReducedVectorField{_ODEForm})(
        ω::AbstractVector, t, z::AbstractVector,
        params)
    x, g = _work(vf, eltype(z))
    _lift!(x, vf.basis, z)
    _gradient!(g, vf, t, x, params)
    return _field!(ω, vf.basis, g)
end

@inline function (vf::_ReducedVectorField{_VelocityForm})(v::AbstractVector, t,
        q::AbstractVector, p::AbstractVector, params)
    x, g = _work(vf, eltype(q))
    _lift!(x, vf.basis, q, p)
    _gradient!(g, vf, t, x, params)
    return _half!(v, vf.basis, g, Val(1))
end

@inline function (vf::_ReducedVectorField{_ForceForm})(f::AbstractVector, t,
        q::AbstractVector, p::AbstractVector, params)
    x, g = _work(vf, eltype(q))
    _lift!(x, vf.basis, q, p)
    _gradient!(g, vf, t, x, params)
    return _half!(f, vf.basis, g, Val(2))
end

# The guards run at emission, before GeometricEquations sees anything.
function _check_reduced_state(b::ReducedBasis{<:Any, <:SymplecticAlgorithm},
        z₀::AbstractVector)
    T = eltype(b.Φ)
    eltype(z₀) === T ||
        throw(ArgumentError("the state has element type $(eltype(z₀)); the basis has $T"))
    length(z₀) == 2b.k ||
        throw(DimensionMismatch("a $(nameof(typeof(b.algorithm))) state holds two blocks of " *
                                "$(b.k); got $(length(z₀))"))
    return nothing
end

function _check_reduced_state(b::ReducedBasis{<:Any, <:PODAlgorithm}, z₀::AbstractVector)
    T = eltype(b.Φ)
    eltype(z₀) === T ||
        throw(ArgumentError("the state has element type $(eltype(z₀)); the basis has $T"))
    isodd(size(b.Φ, 1)) &&
        throw(ArgumentError("a $(nameof(typeof(b.algorithm))) basis of $(size(b.Φ, 1)) rows " *
                            "has no split into two halves; `J_N` needs an even row count"))
    length(z₀) == b.k ||
        throw(DimensionMismatch("a $(nameof(typeof(b.algorithm))) state is $(b.k) long; " *
                                "got $(length(z₀))"))
    return nothing
end

function _check_reduced_state(sd::SymplecticDEIM{T}, z₀::AbstractVector) where {T}
    eltype(z₀) === T ||
        throw(ArgumentError("the state has element type $(eltype(z₀)); the system has $T"))
    n = size(sd.A, 2)
    isodd(n) &&
        throw(ArgumentError("a DEIM basis of $n columns has no split into two halves; " *
                            "`J_2k` needs an even column count"))
    length(z₀) == n ||
        throw(DimensionMismatch("a DEIM state is $n long; got $(length(z₀))"))
    return nothing
end

"""
    reduced_ode(b::ReducedBasis, ∇H!, timespan, timestep, z₀; parameters = NullParameters())
    reduced_ode(sd::SymplecticDEIM, timespan, timestep, z₀)

The `ODEProblem` of the reduced equation that the basis `b` carries, or of the hyper-reduced
system `sd`, with the state `z` in reduced coordinates. Integration is the caller's: emit the
problem, then hand it to an integrator of the caller's own.

For `CotangentLiftSVD`, `CotangentLiftEVD` and `ComplexSVD` the vector field is

```math
\\dot z = J_k A^{\\mathsf T} ∇H(Az) , \\qquad A = `Matrix(b)` ,
```

where `J_k = [0 I_k; -I_k 0]`. For `PODSVD` and `PODEVD` it is the Galerkin form

```math
\\dot z = U^{\\mathsf T} J_N ∇H(Uz) , \\qquad U = Φ ,
```

with `J_N = [0 I_N; -I_N 0]`. A POD basis forms no `J_k`, which exists only for an even
dimension, so its `k` may be odd. A `SymplecticDEIM` emits `ż = J_2k ∇H̃(z)` through
[`hamiltonian_gradient!`](@ref), which is the gradient of its own reduced Hamiltonian.

`∇H!` is the caller's gradient of the Hamiltonian, in the argument order of a vector field:
`∇H!(g, t, x, params)` writes `∇H(x)` into `g`, and may read `x` while it writes `g`. `params` is
the `parameters` keyword, passed through unchanged. No `H` is passed and no invariant is carried,
so the caller measures `H`. The `SymplecticDEIM` method takes no `∇H!`, because its reduced
gradient is the system's own.

`z₀` is in reduced coordinates, of length `2k` for a symplectic basis — two blocks of `k`, the
second starting at `z₀[k + 1]` — and of length `k` for a POD basis. Call `project(b, x₀)` to get
it. Its element type is the basis's.

# Throws

- `ArgumentError` if `eltype(z₀)` is not the basis's element type; if a POD basis has an odd
  number of rows, where no split into two halves exists; and if a DEIM basis has an odd number of
  columns, for the same reason.
- `DimensionMismatch` if `length(z₀)` is not the length above. Without it the blockwise lift
  ignores a surplus entry.

# Cost

A vector field call allocates 0 B, after its first call for each state element type, which takes
one pair of buffers from a per-type cache. The buffers and the basis are held by the emitted
callable, and `∇H!` is a type parameter, so the call is inferred and statically dispatched. What
`∇H!` allocates is not the callable's: a callback's allocations are the caller's. Emission itself
allocates — it builds the callable, its buffers and the problem.

# Concurrency

The work buffers belong to the emitted problem, so one problem is not safe to integrate from two
tasks at once. Emit one problem per task.

# Example

```julia
oscillator!(g, t, x, params) = (g .= x)          # H(x) = ½ xᵀx
b = ReducedBasis(CotangentLiftSVD(), S, NamedTuple(), paramspace; k = 10)
z₀ = project(b, x₀)
prob = reduced_ode(b, oscillator!, (0.0, 100.0), 0.01, z₀)
sol = integrate(prob, ImplicitMidpoint())
```
"""
function reduced_ode(b::ReducedBasis, ∇H!, timespan, timestep, z₀::AbstractVector;
        parameters = NullParameters())
    _check_reduced_state(b, z₀)
    v = _reduced_vector_field(b, ∇H!, eltype(z₀), _lifted_dimension(b), _ODEForm())
    return ODEProblem(v, timespan, timestep, z₀; parameters)
end

function reduced_ode(sd::SymplecticDEIM, timespan, timestep, z₀::AbstractVector)
    _check_reduced_state(sd, z₀)
    v = _reduced_vector_field(sd, hamiltonian_gradient!, eltype(z₀), _lifted_dimension(sd),
        _ODEForm())
    return ODEProblem(v, timespan, timestep, z₀)
end

"""
    reduced_pode(b::ReducedBasis{<:Any, <:SymplecticAlgorithm}, ∇H!, timespan, timestep, z₀; parameters = NullParameters())
    reduced_pode(sd::SymplecticDEIM, timespan, timestep, z₀)

The `PODEProblem` of the split reduced equation: `q̇ = v(t, q, p)` is the first `k` rows of the
symplectic form of [`reduced_ode`](@ref) and `ṗ = f(t, q, p)` is the last `k`, at the same point
`A·[q; p]`. The initial conditions are the two halves of `z₀`, so the problem starts at
`(z₀[1:k], z₀[k + 1:2k])`.

For `CotangentLiftSVD`, `CotangentLiftEVD` and `ComplexSVD`, whose assembly is `A = Matrix(b)`,
the two halves are the two halves of

```math
\\dot z = J_k A^{\\mathsf T} ∇H(Az) , \\qquad J_k = [0 I_k; -I_k 0] .
```

A `SymplecticDEIM` splits `ż = J_2k ∇H̃(z)` the same way, through
[`hamiltonian_gradient!`](@ref). `PODSVD` and `PODEVD` have no method at all: the coordinates of
a POD basis have no `q` and `p` halves to split.

`∇H!(g, t, x, params)` is the caller's gradient of the Hamiltonian, writing `∇H(x)` into `g`,
and `parameters` is where `params` comes from — both exactly as for [`reduced_ode`](@ref). The
`SymplecticDEIM` method takes no `∇H!`, and its two halves are the halves of `J_2k ∇H̃(z)`.

# Throws

As for [`reduced_ode`](@ref): `ArgumentError` on a state whose element type is not the basis's,
and `DimensionMismatch` on a state that is not `2k` long. A `PODSVD` or `PODEVD` basis is a
`MethodError`.

# Cost

Each of `v` and `f` is a callable with its own buffers, and each allocates 0 B after its first
call for each state element type, which takes a pair from that callable's per-type cache. What
`∇H!` allocates is not the callable's: a callback's allocations are the caller's. Emission
allocates.

# Concurrency

The two callables each hold their buffers, so one problem is not safe to integrate from two tasks
at once. Emit one problem per task.

# Example

```julia
oscillator!(g, t, x, params) = (g .= x)          # H(x) = ½ xᵀx
b = ReducedBasis(CotangentLiftSVD(), S, NamedTuple(), paramspace; k = 10)
z₀ = project(b, x₀)
prob = reduced_pode(b, oscillator!, (0.0, 100.0), 0.01, z₀)
sol = integrate(prob, ImplicitMidpoint())
```
"""
function reduced_pode(
        b::ReducedBasis{<:Any, <:SymplecticAlgorithm}, ∇H!, timespan, timestep,
        z₀::AbstractVector; parameters = NullParameters())
    _check_reduced_state(b, z₀)
    k = b.k
    n = _lifted_dimension(b)
    v = _reduced_vector_field(b, ∇H!, eltype(z₀), n, _VelocityForm())
    f = _reduced_vector_field(b, ∇H!, eltype(z₀), n, _ForceForm())
    return PODEProblem(v, f, timespan, timestep, z₀[1:k], z₀[(k + 1):2k]; parameters)
end

function reduced_pode(sd::SymplecticDEIM, timespan, timestep, z₀::AbstractVector)
    _check_reduced_state(sd, z₀)
    n = size(sd.A, 2)
    k = n ÷ 2
    v = _reduced_vector_field(sd, hamiltonian_gradient!, eltype(z₀), n, _VelocityForm())
    f = _reduced_vector_field(sd, hamiltonian_gradient!, eltype(z₀), n, _ForceForm())
    return PODEProblem(v, f, timespan, timestep, z₀[1:k], z₀[(k + 1):n])
end
