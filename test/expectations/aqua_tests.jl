@testitem "the expectation protocol has no ambiguities and no piracy" tags = [:expectations, :quality] begin
    using Aqua, Test

    # `runtests.jl` runs Aqua without these two checks for the whole package, because other
    # parts of BayesBase extend Distributions; the protocol is held to both
    protocol = (
        BayesBase.expectation,
        BayesBase.prepare_expectation,
        BayesBase.supports,
        BayesBase.which_backend,
        BayesBase.haslabel,
        BayesBase.isexact,
        BayesBase.isstochastic,
        BayesBase.combine_label,
        BayesBase.withrng,
        BayesBase.getrng,
        BayesBase.register_backend!,
    )
    callable(method) = Base.unwrap_unionall(method.sig).parameters[1]
    inprotocol(method) =
        any(f -> callable(method) === typeof(f), protocol) ||
        callable(method) <: BayesBase.Logpdf ||
        callable(method) <: Type{<:BayesBase.FirstOf}

    ambiguities = filter(
        ((a, b),) -> inprotocol(a) || inprotocol(b), Test.detect_ambiguities(BayesBase)
    )
    @test isempty(ambiguities)

    piracies = filter(inprotocol, Aqua.Piracy.hunt(BayesBase))
    @test isempty(piracies)
end
