@testitem "quality:aqua" tags = [:quality] begin
    using Aqua
    # BayesBase extends functions of Distributions for Distributions' own types, which Aqua reports
    # as piracy, and some ambiguities among those methods are expected and never reached in
    # practice; the expectation protocol is held to both checks in its own test item.
    Aqua.test_all(
        BayesBase;
        ambiguities = false,
        piracies = false,
        deps_compat = (; check_extras = true, check_weakdeps = true),
    )
end

@testitem "quality:jet" tags = [:quality] begin
    # JET follows Julia's compiler closely, and its releases for Julia 1.10 and 1.11 cannot analyse
    # every package; it runs from 1.12 on.
    if VERSION >= v"1.12"
        using JET
        JET.test_package(BayesBase; target_modules = (BayesBase,))
    end
end

@testitem "quality:doctests" tags = [:quality] begin
    using Documenter
    DocMeta.setdocmeta!(BayesBase, :DocTestSetup, :(using BayesBase); recursive = true)
    doctest(BayesBase; manual = false)
end
