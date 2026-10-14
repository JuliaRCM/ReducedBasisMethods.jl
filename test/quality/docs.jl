using ReducedBasisMethods
using Test

@testset "no physical word in the code or the documentation" begin
    # The package knows no model. `CHANGELOG.md` and `reference/` may name one.
    root = pkgdir(ReducedBasisMethods)
    files = [joinpath(path, file) for dir in ("src", "ext", "docs/src")
             for (path, _, names) in walkdir(joinpath(root, dir)) for file in names]
    for file in [files; joinpath(root, "docs", "make.jl")]
        @test !occursin(r"vlasov|particle"i, read(file, String))
    end
end
