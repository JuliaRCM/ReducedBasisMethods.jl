using ReducedBasisMethods
using Test
using TOML

# `CairoMakie` is a weak dependency: the package resolves and loads in a fresh project that holds
# only it, with no Makie package installed, no extension loaded, and `spectrumplot` without a method.
# The subprocess load path holds only that project and the standard library.
@testset "Load without Makie" begin
    root = pkgdir(ReducedBasisMethods)
    project = TOML.parsefile(joinpath(root, "Project.toml"))
    @test project["extensions"] == Dict("ReducedBasisMethodsCairoMakieExt" => "CairoMakie")
    @test haskey(project["weakdeps"], "CairoMakie")
    @test !haskey(project["deps"], "CairoMakie")

    code = """
    import Pkg
    Pkg.activate(mktempdir(); io = devnull)
    Pkg.develop(path = $(repr(root)); io = devnull)
    using ReducedBasisMethods, Test
    @test !any(dep -> occursin("Makie", dep.name), values(Pkg.dependencies()))
    @test Base.get_extension(ReducedBasisMethods, :ReducedBasisMethodsCairoMakieExt) === nothing
    @test_throws MethodError spectrumplot([1.0, 0.5])
    """
    load_path = join(["@", "@stdlib"], Sys.iswindows() ? ';' : ':')
    @test success(addenv(`$(Base.julia_cmd()) --startup-file=no -e $code`,
        "JULIA_LOAD_PATH" => load_path))
end
