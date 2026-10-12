using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedComplexityModeling: Parameter, ParameterSpace
using ReducedBasisMethods: _deim_indices
using Test

"""
Example 1 from doi.org/10.1137/090766498
"""
function s(x, μ)
    (1 - x) * cos(3π * μ * (x + 1)) * exp(-(1 + x) * μ)
end

const X = range(-1, 1, 100)
const μₜ = range(1, π, 51) # training parameters
const μᵥ = range(1, π, 101) # validation/testing parameters

"""The Example 1 training snapshots `S`, stored in `T`."""
example1_training(::Type{T}) where {T} = T[s(X[i], μₜ[j]) for i in 1:100, j in 1:51]

"""The Example 1 validation snapshots `Sᵥ`, stored in `T`."""
example1_validation(::Type{T}) where {T} = T[s(X[i], μᵥ[j]) for i in 1:100, j in 1:101]

"""The POD basis of `S` at `k = 20`, by the `ReducedBasis` route."""
function pod_basis(S; k = 20)
    basis = ReducedBasis(PODEVD(), S, NamedTuple(),
        ParameterSpace(Parameter(:μ, 1.0, 3.0, 51)); k)
    basis.Φ, basis.Λ
end

@testset "Example 1" begin
    for T in (Float32, Float64)
        S = example1_training(T)
        Ψ, Λ = pod_basis(S; k = 20)

        @test Λ[10] < 1e0
        @test Λ[15] < 1e-2
        # Λ[20] of a Float32 S sits on the round-off floor of its own EVD — 2.46e-5, against
        # 2.91e-8 in Float64 — so its threshold is a Float64 one. The Float32 basis is held to
        # the same standard by the DEIM error below, which carries the same threshold in both.
        if T === Float64
            @test Λ[20] < 1e-7
        end

        Π = get_DEIM_interpolation_matrix(Ψ)
        @test size(Π) == (100, 20)
        @test eltype(Π) === T

        Sᵥ = example1_validation(T)

        D = Ψ * inv(Π' * Ψ)
        xₑ = Π' * X

        Sₐ = zero(Sᵥ) # DEIM approximation

        for i in 1:101
            sx = xx -> s(xx, μᵥ[i])
            Sₐ[:, i] .= D * sx.(xₑ)
        end

        avg_deim_error = sum([norm(Sᵥ[:, i] - Sₐ[:, i], 2) for i in 1:101]) / 101

        @test avg_deim_error < 5e-5

        # The bound of the cited paper: the DEIM error of each validation parameter is
        # bounded by its best approximation error in the basis, amplified by the
        # interpolation matrix.
        bound = [norm(Sᵥ[:, i] - D * s.(xₑ, μᵥ[i]), 2) /
                 ((1 + sqrt(eps(T))) * opnorm(inv(Π' * Ψ), 2) *
                  norm(Sᵥ[:, i] - Ψ * (Ψ' * Sᵥ[:, i]), 2)) for i in 1:101]

        @test maximum(bound) ≤ 1
    end
end

@testset "selection" begin
    # The residual of the second column peaks at its most negative entry. Selecting the row
    # of the largest value rather than of the largest magnitude picks row 3, which the first
    # pick has already taken.
    Ψ₁ = [1 0; 0 -2; 0 1] ./ [1 sqrt(5)]
    @test _deim_indices(Ψ₁) == [1, 2]
    @test get_DEIM_interpolation_matrix(Ψ₁) == [1 0; 0 1; 0 0]

    # The first pick: the largest magnitude is in row 2, the largest value in row 3.
    Ψ₂ = [0.1 0; -1 -2; 0.2 1] ./ [norm([0.1, -1, 0.2]) sqrt(5)]
    @test _deim_indices(Ψ₂) == [2, 3]
    # At k = 1 the loop never runs, so the first pick is the whole selection.
    @test _deim_indices(Ψ₂[:, 1:1]) == [2]

    # A residual that vanishes identically: the row of the largest value is the one the
    # first pick took, and the two candidates that remain are equal, so the first of them
    # decides.
    Ψ₃ = [1 0; 0 -1; 0 -1] ./ [1 sqrt(2)]
    @test _deim_indices(Ψ₃) == [1, 2]

    p₃ = _deim_indices(Ψ₃)
    Π₃ = get_DEIM_interpolation_matrix(Ψ₃)
    @test allunique(p₃)
    @test rank(Π₃' * Ψ₃) == 2
end

@testset "edges" begin
    # Π follows the element type of Ψ.
    Ψ₃₂ = Float32[1 0; 0 -1; 0 -1] ./ Float32[1 sqrt(2)]
    @test eltype(get_DEIM_interpolation_matrix(Ψ₃₂)) === Float32
    @test _deim_indices(Ψ₃₂) == [1, 2]

    # An integer Ψ gives a Float64 Π, never an integer one.
    Ψᵢ = Int32[1 0; 0 -2; 0 1]
    @test _deim_indices(Ψᵢ) == [1, 2]
    @test eltype(get_DEIM_interpolation_matrix(Ψᵢ)) === Float64

    # k = 0 gives an N × 0 matrix.
    @test size(get_DEIM_interpolation_matrix(zeros(3, 0))) == (3, 0)

    # k > N is rejected rather than silently returning a matrix with repeated rows.
    @test_throws ArgumentError get_DEIM_interpolation_matrix(randn(Xoshiro(1), 3, 4))

    # A column whose residual is exactly zero, NaN or Inf. Each leaves the selection with no
    # order to follow: the zero and the NaN residual pick an index already taken, and the
    # Inf residual has no largest magnitude at all.
    @test_throws ArgumentError get_DEIM_interpolation_matrix([1 2; 0 0; 0 0])
    @test_throws ArgumentError get_DEIM_interpolation_matrix([NaN 0; 0 1; 0 0])
    @test_throws ArgumentError get_DEIM_interpolation_matrix([Inf 0; 0 1; 0 0])
    @test_throws ArgumentError get_DEIM_interpolation_matrix([1 0; 0 Inf; 0 0])

    # The selection is scale-invariant: the residual is compared with zero, never with a
    # cutoff. At 3.7e-8 in Float32 the largest residual is below `eps(Float32)`.
    for T in (Float32, Float64)
        Ψ, _ = pod_basis(example1_training(T); k = 20)
        p = _deim_indices(Ψ)

        @test _deim_indices(T(3.7e8) .* Ψ) == p
        @test _deim_indices(T(3.7e-8) .* Ψ) == p
    end
end
