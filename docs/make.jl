# The documentation site of BayesBase. Build it with `make docs`, or `julia --project=docs docs/make.jl`.
using Documenter
using BayesBase

DocMeta.setdocmeta!(BayesBase, :DocTestSetup, :(using BayesBase); recursive = true)

makedocs(
    modules = [BayesBase],
    authors = "Bagaev Dmitry <bvdmitri@gmail.com> and contributors",
    sitename = "BayesBase.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    pages = ["Home" => "index.md"],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://reactivebayes.github.io/BayesBase.jl",
        edit_link = "main",
    ),
)

deploydocs(
    repo = "github.com/ReactiveBayes/BayesBase.jl.git",
    devbranch = "main",
    forcepush = true,
    # A preview for every pull request but CompatHelper's, whose documentation nobody reviews;
    # `DocPreviewCleanup.yml` deletes it when the pull request closes.
    push_preview = !startswith(get(ENV, "GITHUB_HEAD_REF", ""), "compathelper/"),
)
