@testitem "register_expectation_backend!: a registry of backends with priorities" tags = [:expectations] setup = [ExpectationToys] begin
    registry = P.ExpectationBackendRegistry()
    @test isempty(registry)
    P.register_expectation_backend!(ToyGrid(); registry)
    P.register_expectation_backend!(ToyExact(); registry, priority = 3)
    @test length(registry) == 2
    # registering again replaces the priority instead of adding a duplicate
    P.register_expectation_backend!(ToyGrid(); registry, priority = 1)
    @test length(registry) == 2
    @test P.expectation_backend_priority(ToyGrid(); registry) == 1
    @test P.expectation_backend_priority(ToyExact(); registry) == 3
    @test P.expectation_backend_priority(ToyExact(); registry = P.ExpectationBackendRegistry()) === nothing
end

@testitem "ExpectationDynamicRouter: exact first, then not stochastic, then priority, then name" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    sampler = ToySampler(100, StableRNG(1))
    q = Normal(0.0, 1.0)

    registry = P.ExpectationBackendRegistry()
    foreach(b -> P.register_expectation_backend!(b; registry), (sampler, ToyGrid(), ToyExact()))
    router = P.ExpectationDynamicRouter(registry)
    @test P.which_expectation_backend(router, identity, q) === ToyExact()  # exact
    @test P.which_expectation_backend(router, log, q) === ToyGrid()        # not stochastic
    @test P.which_expectation_backend(router, log, Gamma(1, 1)) === nothing
    @test P.expectation(router, identity, q) === 0.0
    @test P.is_expectation_exact(router, identity, q) === true
    @test P.is_expectation_exact(router, log, q) === false
    @test P.supports_expectation(router, log, Gamma(1, 1)) === false
    @test_throws P.UnsupportedExpectationError P.expectation(router, log, Gamma(1, 1))

    # priority beats name, name beats nothing; the order of registration never matters
    struct Alpha end
    struct Bravo end
    P.supports_expectation(::Alpha, ::Type, ::Type{<:Normal}) = true
    P.supports_expectation(::Bravo, ::Type, ::Type{<:Normal}) = true
    forward, backward = P.ExpectationBackendRegistry(), P.ExpectationBackendRegistry()
    foreach(b -> P.register_expectation_backend!(b; registry = forward), (Alpha(), Bravo()))
    foreach(b -> P.register_expectation_backend!(b; registry = backward), (Bravo(), Alpha()))
    @test P.which_expectation_backend(P.ExpectationDynamicRouter(forward), log, q) === Alpha()
    @test P.which_expectation_backend(P.ExpectationDynamicRouter(backward), log, q) === Alpha()
    P.register_expectation_backend!(Bravo(); registry = backward, priority = 1)
    @test P.which_expectation_backend(P.ExpectationDynamicRouter(backward), log, q) === Bravo()
end

@testitem "ExpectationDynamicRouter: the global registry is the default" tags = [:expectations] setup = [ExpectationToys] begin
    @test P.ExpectationDynamicRouter().registry === P.ExpectationDynamicRouter().registry
    @test P.ExpectationDynamicRouter().registry isa P.ExpectationBackendRegistry
end

@testitem "ExpectationFirstOf(ExpectationDynamicRouter) freezes the choice into a type-stable router" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    sampler = ToySampler(100, StableRNG(1))
    registry = P.ExpectationBackendRegistry()
    foreach(b -> P.register_expectation_backend!(b; registry), (sampler, ToyGrid(), ToyExact()))
    frozen = P.ExpectationFirstOf(P.ExpectationDynamicRouter(registry))
    @test frozen isa P.ExpectationFirstOf
    # not stochastic before stochastic, exact preferred per pair
    @test frozen.backends === (ToyExact(), ToyGrid(), sampler)
    @test frozen.prefer === (P.ExpectationExact(),)

    q = Normal(0.0, 1.0)
    dynamic = P.ExpectationDynamicRouter(registry)
    for f in (identity, abs2, log)
        @test P.which_expectation_backend(frozen, f, q) === P.which_expectation_backend(dynamic, f, q)
    end
    @test @inferred(P.expectation(frozen, identity, q)) === 0.0
end
