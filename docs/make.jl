using ReducedBasisMethods
using Documenter

include("grep_physical_words.jl")

makedocs(;
    modules = [ReducedBasisMethods],
    authors = "Tobias M. Blickhan, Michael Kraus, Tomasz M. Tyranoski",
    repo = "https://github.com/JuliaRCM/ReducedBasisMethods.jl/blob/{commit}{path}#L{line}",
    sitename = "ReducedBasisMethods.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliarcm.github.io/ReducedBasisMethods.jl",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Library" => "library.md"
    ],
    doctest = true
)

# The sources, the manual and the build's own `sitename`, `authors` and page titles carry no
# physical word, so one in the site was added by the build itself: it fails here rather than
# publishing a page that names a model. `test/quality/docs.jl` greps the sources with the same
# words.
forbidden = forbidden_files(joinpath(@__DIR__, "build"))
isempty(forbidden) ||
    error("the built documentation names a model: ", join(forbidden, ", "))

deploydocs(;
    repo = "github.com/JuliaRCM/ReducedBasisMethods.jl",
    devurl = "latest",
    devbranch = "main"
)
