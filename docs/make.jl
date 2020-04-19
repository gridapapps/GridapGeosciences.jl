using Documenter, GridapGeosciences

makedocs(;
    modules=[GridapGeosciences],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/santiagobadia/GridapGeosciences.jl/blob/{commit}{path}#L{line}",
    sitename="GridapGeosciences.jl",
    authors="Santiago Badia",
    assets=String[],
)

deploydocs(;
    repo="github.com/santiagobadia/GridapGeosciences.jl",
)
