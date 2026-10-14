using ReducedBasisMethods
using Test
using TOML

# The package loads with no Makie present. `CairoMakie` is a weak dependency, so nothing about the
# ordinary load brings it in — which is what this file states from the inside of a fresh temporary
# project that holds only the package, run in a subprocess whose load path `@:@stdlib` cannot be
# satisfied out of the ambient `@v#.#` environment by a stray `using`.
@testset "Load without Makie" begin
    root = pkgdir(ReducedBasisMethods)
    project = TOML.parsefile(joinpath(root, "Project.toml"))

    # the stub is in the package and the method is not: no other dependency holds a plotter
    @test !haskey(project["deps"], "Plots")
    @test !haskey(project["deps"], "LaTeXStrings")
    @test project["extensions"] == Dict{String, String}(
        "ReducedBasisMethodsCairoMakieExt" => "CairoMakie")
    @test haskey(project["weakdeps"], "CairoMakie")
    @test !haskey(project["deps"], "CairoMakie")
    # the generic-only skeleton of `skeleton.jl` holds the rest of the `[deps]` list

    environment = mktempdir()
    code = """
    import Pkg
    Pkg.activate($(repr(environment)); io = devnull)
    Pkg.develop(path = $(repr(root)); io = devnull)
    using ReducedBasisMethods

    dependencies = Pkg.dependencies()
    uuids = keys(dependencies)
    names = [dependencies[uuid].name for uuid in uuids]
    println("MAKIE=", join(sort([name for name in names if occursin("Makie", name)]), ","))
    println("MAKIEWEAKDEP=",
        Base.UUID("13f3f980-e62b-5c42-98c6-ff1f3baf88f0") in uuids)
    println("EXTENSION=", Base.get_extension(
        ReducedBasisMethods, :ReducedBasisMethodsCairoMakieExt) === nothing)
    try
        ReducedBasisMethods.spectrumplot(Float64[1.0, 0.5])
        println("STUB=no method missing")
    catch error
        println("STUB=", error isa MethodError)
    end
    """
    output = String(read(setenv(
        `$(Base.julia_cmd()) --startup-file=no --color=no -e $code`,
        "JULIA_LOAD_PATH" => "@:@stdlib")))
    reported = Dict(strip.(split(l, '='; limit = 2))
    for l in split(output, '\n') if '=' in l)

    @test reported["MAKIE"] == ""
    @test reported["MAKIEWEAKDEP"] == "false"
    @test reported["EXTENSION"] == "true"
    @test reported["STUB"] == "true"
end
