using ReducedBasisMethods
using Test
using TOML

"Every file under `dir` whose text matches `pattern`."
function files_matching(pattern, dir)
    files = [joinpath(root, name) for (root, _, names) in walkdir(dir) for name in names]
    return filter(file -> occursin(pattern, read(file, String)), files)
end

@testset "Docs and changelog" begin
    root = pkgdir(ReducedBasisMethods)

    @testset "no physical word in the code or the documentation" begin
        # The package knows no model. `CHANGELOG.md` and `reference/` may name one.
        for dir in ("src", "ext", "docs/src")
            @test isempty(files_matching(r"vlasov|plasma|particle"i, joinpath(root, dir)))
        end
        @test !occursin(r"vlasov|plasma|particle"i,
            read(joinpath(root, "docs", "make.jl"), String))
    end

    @testset "the docs environment holds no Weave and no Plots" begin
        @test isempty(intersect(["Plots", "Weave"],
            keys(TOML.parsefile(joinpath(root, "docs", "Project.toml"))["deps"])))
    end

    @testset "at least one `jldoctest` in the package" begin
        # A build with zero doctests passes vacuously.
        @test !isempty(files_matching("```jldoctest", joinpath(root, "src"))) ||
              !isempty(files_matching("```jldoctest", joinpath(root, "docs", "src")))
    end

    @testset "the changelog names the extension and the function" begin
        changelog = read(joinpath(root, "CHANGELOG.md"), String)
        @test occursin("ReducedBasisMethodsCairoMakieExt", changelog)
        @test occursin("spectrumplot", changelog)
    end
end
