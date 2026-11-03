using LinearAlgebra
using Random
using ReducedBasisMethods
using ReducedBasisMethods.HDF5: h5open
using ReducedComplexityModeling: Parameter, ParameterSpace
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

"""The average DEIM error of the basis `Ψ` over the validation snapshots `Sᵥ`."""
function example1_deim_error(Ψ, Sᵥ)
    Π = get_DEIM_interpolation_matrix(Ψ)
    D = Ψ * inv(Π' * Ψ)
    xₑ = Π' * X

    Sₐ = zero(Sᵥ) # DEIM approximation

    for i in 1:101
        Sₐ[:, i] .= D * s.(xₑ, μᵥ[i])
    end

    return sum([norm(Sᵥ[:, i] - Sₐ[:, i], 2) for i in 1:101]) / 101
end

@testset "Example 1" begin
    for T in (Float32, Float64)
        S = example1_training(T)
        Ψ, Λ = pod_basis(S; k = 20)

        @test Λ[10] < 1e0
        @test Λ[15] < 1e-2
        # The Float32 Gram matrix's EVD has a round-off floor of order `eps(T) * Λ[1]`.
        # The threshold accounts for that floor while retaining the 1e-7 bound
        # where it exceeds the round-off scale.
        @test Λ[20] < max(1e-7, eps(T) * Λ[1])

        Π = get_DEIM_interpolation_matrix(Ψ)
        @test size(Π) == (100, 20)
        @test eltype(Π) === T

        Sᵥ = example1_validation(T)

        D = Ψ * inv(Π' * Ψ)
        xₑ = Π' * X

        avg_deim_error = example1_deim_error(Ψ, Sᵥ)

        # The columns past the eigenvalues that the EVD resolves, those above `eps(T) * Λ[1]`,
        # are round-off, and so is the error they remove: in Float32 they start at column 17 or
        # 18, and their error depends on the BLAS. Where all 20 columns are resolved, as in
        # Float64, the bound is 5e-5; otherwise it is the error of the resolved columns alone.
        kᵣ = count(>(eps(T) * Λ[1]), Λ[1:20])
        @test avg_deim_error < (kᵣ == 20 ? 5e-5 : example1_deim_error(Ψ[:, 1:kᵣ], Sᵥ))

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
    # Each numbered case runs in both precisions: the rule is the magnitude of the residual, and
    # the element type must not change which row it picks.
    for T in (Float32, Float64)
        # The residual of the second column peaks at its most negative entry. The largest value
        # selects row 3, while the largest magnitude selects row 2.
        # The first pick is row 1, so neither selection repeats it.
        Ψ₁ = T[1 0; 0 -2; 0 1] ./ T[1 sqrt(5)]
        @test get_DEIM_indices(Ψ₁) == [1, 2]
        @test get_DEIM_interpolation_matrix(Ψ₁) == T[1 0; 0 1; 0 0]

        # The first pick: the largest magnitude is in row 2, the largest value in row 3.
        Ψ₂ = T[0.1 0; -1 -2; 0.2 1] ./ T[norm([0.1, -1, 0.2]) sqrt(5)]
        @test get_DEIM_indices(Ψ₂) == [2, 3]
        # At k = 1 the first pick is the whole selection.
        @test get_DEIM_indices(Ψ₂[:, 1:1]) == [2]

        # The residual is zero at the selected row and has equal largest magnitudes in
        # rows 2 and 3, so the tie goes to row 2.
        Ψ₃ = T[1 0; 0 -1; 0 -1] ./ T[1 sqrt(2)]
        @test get_DEIM_indices(Ψ₃) == [1, 2]

        p₃ = get_DEIM_indices(Ψ₃)
        Π₃ = get_DEIM_interpolation_matrix(Ψ₃)
        @test allunique(p₃)
        @test rank(Π₃' * Ψ₃) == 2
    end
end

@testset "edges" begin
    # Π follows the element type of Ψ.
    Ψ₃₂ = Float32[1 0; 0 -1; 0 -1] ./ Float32[1 sqrt(2)]
    @test eltype(get_DEIM_interpolation_matrix(Ψ₃₂)) === Float32
    @test get_DEIM_indices(Ψ₃₂) == [1, 2]

    # An integer Ψ gives a Float64 Π, never an integer one.
    Ψᵢ = Int32[1 0; 0 -2; 0 1]
    @test get_DEIM_indices(Ψᵢ) == [1, 2]
    @test eltype(get_DEIM_interpolation_matrix(Ψᵢ)) === Float64

    # k = 0: the empty selection, and the N × 0 matrix built from it.
    @test get_DEIM_indices(zeros(3, 0)) == Int[]
    @test size(get_DEIM_interpolation_matrix(zeros(3, 0))) == (3, 0)

    # k > N leaves too few distinct rows for the interpolation.
    @test_throws ArgumentError get_DEIM_interpolation_matrix(randn(Xoshiro(1), 3, 4))

    # A zero residual adds no independent column; a non-finite residual is not interpolable.
    @test_throws ArgumentError get_DEIM_interpolation_matrix([1 2; 0 0; 0 0])
    @test_throws ArgumentError get_DEIM_interpolation_matrix([NaN 0; 0 1; 0 0])
    @test_throws ArgumentError get_DEIM_interpolation_matrix([Inf 0; 0 1; 0 0])
    @test_throws ArgumentError get_DEIM_interpolation_matrix([1 0; 0 Inf; 0 0])

    # The validator rejects a zero, NaN or Inf first column as it does every other column.
    for T in (Float32, Float64), bad in (zero(T), T(NaN), T(Inf))

        @test_throws ArgumentError get_DEIM_interpolation_matrix(reshape(T[bad, 0, 0], 3, 1))
        @test_throws ArgumentError get_DEIM_indices(reshape(T[bad, 0, 0], 3, 1))

        # A bad first column of a two-column basis is reported as column 1, not as column 2, and
        # a zero first column is rejected before it reaches the solve, which as a singular
        # `PᵀΨ` would raise `SingularException` where the `ArgumentError` is due.
        err = try
            get_DEIM_indices(T[bad 1; 0 0; 0 1])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("column 1", sprint(showerror, err))
    end
    @test_throws ArgumentError get_DEIM_indices([0.0 1; 0 0; 0 0])

    # The selection is scale-invariant: the residual is compared with zero, never with a
    # cutoff. At 3.7e-8 in Float32 the largest residual is below `eps(Float32)`.
    for T in (Float32, Float64)
        Ψ, _ = pod_basis(example1_training(T); k = 20)
        p = get_DEIM_indices(Ψ)

        @test get_DEIM_indices(T(3.7e8) .* Ψ) == p
        @test get_DEIM_indices(T(3.7e-8) .* Ψ) == p

        # A subnormal first column is positive, so it is selected rather than refused: column 1's
        # test is the same `0 < ρ`, with no threshold.
        @test get_DEIM_indices(reshape(T[nextfloat(zero(T)), zero(T)], 2, 1)) == [1]
    end
end

@testset "indices are the index form of the interpolation matrix" begin
    # Π' * A samples the rows of A that get_DEIM_indices selects, for any basis and any finite A.
    for T in (Float32, Float64)
        rng = Xoshiro(7)
        Ψ = Matrix(qr(randn(rng, T, 50, 8)).Q)
        A = randn(rng, T, 50, 5)
        p = get_DEIM_indices(Ψ)

        @test length(p) == 8
        @test allunique(p)
        @test get_DEIM_interpolation_matrix(Ψ)' * A == A[p, :]
    end
end

"""
Saves `Π` into the new file `fpath` with the keywords `kw`, closes the file, and returns what
`load_DEIM_interpolation_matrix` reads from it with the same keywords.
"""
function deim_roundtrip(fpath, Π; kw...)
    h5open(h5 -> save_DEIM_interpolation_matrix(h5, Π; kw...), fpath, "w")
    return h5open(h5 -> load_DEIM_interpolation_matrix(h5; kw...), fpath, "r")
end

@testset "HDF5 round trip of the interpolation matrix" begin
    @test Base.isexported(ReducedBasisMethods, :save_DEIM_interpolation_matrix)
    @test Base.isexported(ReducedBasisMethods, :load_DEIM_interpolation_matrix)

    mktempdir() do dir
        fpath = joinpath(dir, "deim.h5")

        for T in (Float32, Float64)
            S = example1_training(T)
            basis = ReducedBasis(PODEVD(), S, NamedTuple(),
                ParameterSpace(Parameter(:μ, 1.0, 3.0, 51)); k = 20)
            Ψ = basis.Φ
            Π = get_DEIM_interpolation_matrix(Ψ)
            p = get_DEIM_indices(Ψ)

            # The empty keyword list reaches the default path, which is the root.
            for (kw, dataset) in (((;), "/Π"), ((; path = "deim"), "/deim/Π"),
                ((; path = "reduction/deim"), "/reduction/deim/Π"))
                loaded = deim_roundtrip(fpath, Π; kw...)
                @test loaded == Π
                @test eltype(loaded) === T
                @test loaded' * S == S[p, :]

                # The dataset has the name and the group that the file format fixes.
                @test h5open(h5 -> read(h5[dataset]), fpath, "r") == Π

                # A second save to the same path keeps HDF5's error and the first matrix.
                h5open(fpath, "r+") do h5
                    @test_throws ErrorException save_DEIM_interpolation_matrix(h5, 2 .* Π; kw...)
                    @test load_DEIM_interpolation_matrix(h5; kw...) == Π
                end
            end

            # k = 0: the N × 0 matrix keeps its size and its element type.
            Π₀ = deim_roundtrip(fpath, get_DEIM_interpolation_matrix(zeros(T, 3, 0)))
            @test size(Π₀) == (3, 0)
            @test eltype(Π₀) === T

            # A lazy wrapper reaches the file as the dense matrix it stands for.
            loaded = deim_roundtrip(fpath, Π')
            @test loaded == Matrix(Π')
            @test size(loaded) == (20, 100)
            @test eltype(loaded) === T

            # The matrix and a basis share one file, in one group or in two, whichever is
            # written first.
            for (path_b, path_Π) in (("/", "/"), ("/", "deim"), ("reduction", "/"),
                    ("reduction", "reduction"), ("reduction", "deim")),
                save_Π_first in (true, false)

                h5open(fpath, "w") do h5
                    save_Π_first && save_DEIM_interpolation_matrix(h5, Π; path = path_Π)
                    h5save(h5, basis; path = path_b)
                    save_Π_first || save_DEIM_interpolation_matrix(h5, Π; path = path_Π)
                end

                # The reads alternate, so each one follows a read of the other.
                h5open(fpath, "r") do h5
                    @test load_DEIM_interpolation_matrix(h5; path = path_Π) == Π
                    for _ in 1:2
                        b = h5load(ReducedBasis, h5; path = path_b)
                        @test b.Φ == basis.Φ
                        @test b.Λ == basis.Λ
                        @test load_DEIM_interpolation_matrix(h5; path = path_Π) == Π
                    end
                end
            end

            # A path with no dataset `Π` keeps HDF5's error, for a group that holds only a
            # basis and for a group that does not exist.
            h5open(fpath, "w") do h5
                h5save(h5, basis; path = "reduction")
                @test_throws KeyError load_DEIM_interpolation_matrix(h5)
                @test_throws KeyError load_DEIM_interpolation_matrix(h5; path = "reduction")
                @test_throws KeyError load_DEIM_interpolation_matrix(h5; path = "deim")
            end
        end
    end
end
