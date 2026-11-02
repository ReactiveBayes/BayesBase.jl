@testitem "register_backend!: a registry of backends with priorities" tags = [:expectations] setup = [ExpectationToys] begin
    registry = P.BackendRegistry()
    @test isempty(registry)
    P.register_backend!(ToyGrid(); registry)
    P.register_backend!(ToyExact(); registry, priority = 3)
    @test length(registry) == 2
    # registering again replaces the priority instead of adding a duplicate
    P.register_backend!(ToyGrid(); registry, priority = 1)
    @test length(registry) == 2
end

@testitem "DynamicRouter: exact first, then not stochastic, then priority, then name" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    sampler = ToySampler(100, StableRNG(1))
    q = Normal(0.0, 1.0)

    registry = P.BackendRegistry()
    foreach(b -> P.register_backend!(b; registry), (sampler, ToyGrid(), ToyExact()))
    router = P.DynamicRouter(registry)
    @test P.which_backend(router, identity, q) === ToyExact()  # exact
    @test P.which_backend(router, log, q) === ToyGrid()        # not stochastic
    @test P.which_backend(router, log, Gamma(1, 1)) === nothing
    @test P.expectation(router, identity, q) === 0.0
    @test P.isexact(router, identity, q) === true
    @test P.isexact(router, log, q) === false
    @test P.supports(router, log, Gamma(1, 1)) === false
    @test_throws P.NoBackendError P.expectation(router, log, Gamma(1, 1))

    # priority beats name, name beats nothing; the order of registration never matters
    struct Alpha end
    struct Bravo end
    P.supports(::Alpha, ::Type, ::Type{<:Normal}) = true
    P.supports(::Bravo, ::Type, ::Type{<:Normal}) = true
    forward, backward = P.BackendRegistry(), P.BackendRegistry()
    foreach(b -> P.register_backend!(b; registry = forward), (Alpha(), Bravo()))
    foreach(b -> P.register_backend!(b; registry = backward), (Bravo(), Alpha()))
    @test P.which_backend(P.DynamicRouter(forward), log, q) === Alpha()
    @test P.which_backend(P.DynamicRouter(backward), log, q) === Alpha()
    P.register_backend!(Bravo(); registry = backward, priority = 1)
    @test P.which_backend(P.DynamicRouter(backward), log, q) === Bravo()
end

@testitem "DynamicRouter: the global registry is the default" tags = [:expectations] setup = [ExpectationToys] begin
    @test P.DynamicRouter().registry === P.DynamicRouter().registry
    @test P.DynamicRouter().registry isa P.BackendRegistry
end

@testitem "FirstOf(DynamicRouter) freezes the choice into a type-stable router" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    sampler = ToySampler(100, StableRNG(1))
    registry = P.BackendRegistry()
    foreach(b -> P.register_backend!(b; registry), (sampler, ToyGrid(), ToyExact()))
    frozen = P.FirstOf(P.DynamicRouter(registry))
    @test frozen isa P.FirstOf
    # not stochastic before stochastic, exact preferred per pair
    @test frozen.backends === (ToyExact(), ToyGrid(), sampler)
    @test frozen.prefer === (P.Exact(),)

    q = Normal(0.0, 1.0)
    dynamic = P.DynamicRouter(registry)
    for f in (identity, abs2, log)
        @test P.which_backend(frozen, f, q) === P.which_backend(dynamic, f, q)
    end
    @test @inferred(P.expectation(frozen, identity, q)) === 0.0
end
