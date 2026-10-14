using ReducedBasisMethods
using Test
using TOML

# The words a generic package must not write, and the search over a tree that holds them, are
# defined once for the package and used twice: `docs/make.jl` greps the site it builds, this file
# greps the sources. Including it here keeps the two from drifting apart.
include(joinpath(pkgdir(ReducedBasisMethods), "docs", "grep_physical_words.jl"))

"How many files under `dir` contain at least one `jldoctest` block."
function jldoctest_blocks(dir)
    count(file -> occursin("```jldoctest", read(file, String)),
        [joinpath(root, name) for (root, _, names) in walkdir(dir) for name in names])
end

@testset "Docs and changelog" begin
    root = pkgdir(ReducedBasisMethods)
    make = read(joinpath(root, "docs", "make.jl"), String)

    @testset "no physical word in the sources or the build's configuration" begin
        # check 5, on the sources, the manual and the build's own configuration: `sitename`,
        # `authors` and the page titles are the words that reach the site and appear nowhere else,
        # and `docs/make.jl` is where they are written. `CHANGELOG.md` and `reference/` may name
        # the model, and are left out on purpose.
        for path in ("src", "docs/src", "docs/make.jl")
            @test isempty(forbidden_files(joinpath(root, path)))
        end
        # What the build itself renders is grepped by the build, once the site exists: the grep of
        # its `build` runs after `makedocs` and fails the Documentation workflow, so the assertion
        # is that the build holds that grep rather than a repeat of it here.
        @test occursin(r"""forbidden_files\(joinpath\(@__DIR__, "build"\)\)""", make)
        @test occursin(r"""isempty\(forbidden\)\s*\|\|\s*error\(""", make)
    end

    @testset "the docs environment holds no Weave and no Plots" begin
        # Neither the docs build nor the manual uses Weave or Plots, so neither package
        # belongs in the docs environment.
        @test isempty(intersect(["Plots", "Weave"],
            keys(TOML.parsefile(joinpath(root, "docs", "Project.toml"))["deps"])))
        @test !occursin("Weave", make)
        @test !occursin("Plots", make)
        @test occursin("doctest = true", make)
    end

    @testset "at least one `jldoctest` in the package" begin
        # A build with zero doctests passes vacuously, so it is the count that is asserted.
        n = jldoctest_blocks(joinpath(root, "src")) +
            jldoctest_blocks(joinpath(root, "docs", "src"))
        @test n ≥ 1
    end

    @testset "the changelog names the extension and the function" begin
        # Search only the unreleased section, ending at the next version heading.
        changelog = read(joinpath(root, "CHANGELOG.md"), String)
        head = findfirst("## [Unreleased]", changelog)
        @test head !== nothing
        rest = SubString(changelog, nextind(changelog, last(head)))
        tail = findfirst(r"^## \["m, rest)
        unreleased = changelog[first(head):(tail === nothing ? lastindex(changelog) :
                                            last(head) + first(tail) - 1)]
        @test occursin("ReducedBasisMethodsCairoMakieExt", unreleased)
        @test occursin("spectrumplot", unreleased)
    end

    @testset "the test environment takes `CairoMakie` from the package alone" begin
        # `CairoMakie` is a weak dependency, so the test environment states its own entry and takes
        # the bound from the package.
        test_project = TOML.parsefile(joinpath(root, "test", "Project.toml"))
        @test haskey(test_project["deps"], "CairoMakie")
        @test !haskey(get(test_project, "compat", Dict{String, Any}()), "CairoMakie")
    end
end
