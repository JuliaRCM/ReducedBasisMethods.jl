using LinearAlgebra
using Test

# The comparator of `scripts/reference_spectra.jl`, on its own: the paper-data run it belongs to
# needs a 9.6 GiB snapshot, and this is the rule that decides whether that run says PASS. The
# script ends in `exit(main(...))` because it is a script, so the file is loaded up to that line.
const SCRIPT = normpath(joinpath(@__DIR__, "..", "..", "scripts", "reference_spectra.jl"))

module ReferenceSpectraScript end
Base.include_string(ReferenceSpectraScript,
    first(split(read(SCRIPT, String), "exit(main(")), SCRIPT)

@testset "Reference spectra comparator" begin
    # The bound is the Weyl bound of the reference matrix, `max(m, n)·eps(Float64)·σ₁`.
    reference = zeros(2000, 66)
    σref = Float64[4.2e4, 2.1e4, 1.05e4]
    deviation, bound = ReferenceSpectraScript.spectrum_deviation(σref, σref, reference)
    @test deviation == 0
    @test bound == 2000 * eps(Float64) * σref[1]

    # A spectrum perturbed by a relative `1e-9` deviates by `4.2e-5` absolute, more than three
    # orders of magnitude outside the `1.87e-8` bound. Divided by `σ₁` first — which the bound
    # already carries — the same deviation reads `1e-9` and passes, so a comparator that divides
    # accepts any perturbation this size. The absolute comparison rejects it.
    σin = σref .* (1 + 1e-9)
    deviated, bound = ReferenceSpectraScript.spectrum_deviation(σin, σref, reference)
    @test deviated ≈ 4.2e-5 rtol = 1e-4
    @test deviated / σref[1] ≤ bound
    @test !(deviated ≤ bound)
end
