using BayesBase
using Documenter

DocMeta.setdocmeta!(BayesBase, :DocTestSetup, :(using BayesBase); recursive=true)

makedocs(;
    modules=[BayesBase],
    authors="Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    sitename="BayesBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://reactivebayes.github.io/BayesBase.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/ReactiveBayes/BayesBase.jl", devbranch="main", forcepush=true)
