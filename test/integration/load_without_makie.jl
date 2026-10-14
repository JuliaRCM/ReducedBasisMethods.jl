using ReducedBasisMethods
using Test

# `Makie` is a weak dependency: loading the package, in an environment that has a Makie backend
# installed, loads no Makie module, and `spectrumplot` has no method.
@testset "Load without Makie" begin
    code = """
    using ReducedBasisMethods, Test
    @test !any(m -> occursin("Makie", m.name), keys(Base.loaded_modules))
    @test_throws MethodError spectrumplot([1.0, 0.5])
    """
    @test success(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`)
end
