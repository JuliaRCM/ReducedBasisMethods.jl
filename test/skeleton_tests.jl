using Aqua
using TOML

@testset "Generic-only skeleton" begin
    project = TOML.parsefile(joinpath(pkgdir(ReducedBasisMethods), "Project.toml"))
    deps = keys(project["deps"])

    # no model package, and no integrator, in [deps]
    for name in ("VlasovMethods", "ParticleMethods", "PoissonSolvers",
        "GeometricIntegrators", "GeometricIntegratorsBase")
        @test name ∉ deps
    end

    # none of the ten unused [deps]
    for name in ("TypedTables", "Optimisers", "Zygote", "LinearMaps", "RecursiveArrayTools",
        "Plots", "Distances", "LaTeXStrings", "Parameters", "Random")
        @test name ∉ deps
    end

    # the integrator is a test-only dependency
    @test haskey(project["extras"], "GeometricIntegratorsBase")
    @test "GeometricIntegratorsBase" ∈ project["targets"]["test"]

    Aqua.test_stale_deps(ReducedBasisMethods)
    Aqua.test_undefined_exports(ReducedBasisMethods)
end
