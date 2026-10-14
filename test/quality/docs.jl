using ReducedBasisMethods
using Test
using TOML

# The words a generic package must not write: ReducedBasisMethods knows no model, so a physical
# word in its sources or its documentation is a physical example that does not belong to it (T7.15,
# section 3.6 of the plan).
const FORBIDDEN = r"vlasov|plasma|particle"i

"Every file of the tree, recursively, in sorted order."
function walkfiles(dir)
    isdir(dir) || return String[]
    files = String[]
    for (root, _, names) in walkdir(dir)
        append!(files, joinpath.(root, names))
    end
    return sort!(files)
end

"Every file under `dir` whose text holds one of `FORBIDDEN`, as paths relative to `dir`."
function forbidden_files(dir)
    return String[relpath(p, dir)
                  for p in walkfiles(dir)
                  if isfile(p) && occursin(FORBIDDEN, String(read(p)))]
end

"How many `jldoctest` blocks the tree under `dir` holds."
function jldoctest_blocks(dir)
    count(p -> occursin("```jldoctest", String(read(p))), walkfiles(dir))
end

@testset "Docs and changelog" begin
    root = pkgdir(ReducedBasisMethods)

    @testset "no physical word in `src`, `docs/src` or the built docs" begin
        # check 5, on the sources and the manual. `CHANGELOG.md` and `reference/` may name the
        # model, and are left out on purpose.
        for dir in ("src", "docs/src")
            @test isempty(forbidden_files(joinpath(root, dir)))
        end
        # the built site is searched where it exists: `docs/build` is a build artefact, so a run
        # that has built no docs searches nothing and the build itself is the evidence
        build = joinpath(root, "docs", "build")
        isdir(build) && @test isempty(forbidden_files(build))
    end

    @testset "the docs environment holds no Weave and no Plots" begin
        # `docs/make.jl` no longer loads `Weave` and no page uses either, so the docs environment
        # does not resolve them.
        @test isempty(intersect(["Plots", "Weave"],
            keys(TOML.parsefile(joinpath(root, "docs", "Project.toml"))["deps"])))
        make = read(joinpath(root, "docs", "make.jl"), String)
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
        # check 6: one entry under the unreleased heading. `[Unreleased]` holds the entries of the
        # parts that precede this one as well, so the section is cut at the next version heading.
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
