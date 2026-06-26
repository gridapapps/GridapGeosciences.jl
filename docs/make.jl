using Documenter

include("tutorials.jl")

makedocs(;
    sitename = "GridapGeosciences.jl",
    authors = "Tamara A. Tambyah, Alberto F. Martin, Santiago Badia, David Lee",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://gridap.github.io/GridapGeosciences.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => tutorial_pages,
    ],
    warnonly = true,
    clean = true,
)

deploydocs(;
    repo = "github.com/gridap/GridapGeosciences.jl",
    devbranch = "atlas_discrete_models",
)
