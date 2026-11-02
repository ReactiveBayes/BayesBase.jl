@testitem "the expectation protocol has no ambiguities and no piracy" tags = [:expectations, :quality] begin
    using Aqua, Test

    # `runtests.jl` runs Aqua without these two checks for the whole package, because other
    # parts of BayesBase extend Distributions; the protocol is held to both
    protocol = (
        BayesBase.expectation,
        BayesBase.prepare_expectation,
        BayesBase.supports_expectation,
        BayesBase.which_expectation_backend,
        BayesBase.has_expectation_label,
        BayesBase.is_expectation_exact,
        BayesBase.is_expectation_stochastic,
        BayesBase.combine_expectation_label,
        BayesBase.with_expectation_rng,
        BayesBase.get_expectation_rng,
        BayesBase.register_expectation_backend!,
    )
    callable(method) = Base.unwrap_unionall(method.sig).parameters[1]
    inprotocol(method) =
        any(f -> callable(method) === typeof(f), protocol) ||
        callable(method) <: Type{<:BayesBase.ExpectationFirstOf}

    ambiguities = filter(
        ((a, b),) -> inprotocol(a) || inprotocol(b), Test.detect_ambiguities(BayesBase)
    )
    @test isempty(ambiguities)

    piracies = filter(inprotocol, Aqua.Piracy.hunt(BayesBase))
    @test isempty(piracies)
end
