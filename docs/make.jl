using ReducedBasisMethods
using Documenter
using Weave


# weave("src/poisson.jmd",
#          out_path = "src",
#          doctype = "github")


makedocs(;
    modules=[ReducedBasisMethods],
    authors="Tobias M. Blickhan, Michael Kraus, Tomasz M. Tyranoski",
    repo="https://github.com/JuliaGNI/ReducedBasisMethods.jl/blob/{commit}{path}#L{line}",
    sitename="ReducedBasisMethods.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliagni.github.io/ReducedBasisMethods.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Library" => "library.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/ReducedBasisMethods.jl",
)
