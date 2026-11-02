@testitem "ExpectationFirstOf: construction" tags = [:expectations] setup = [ExpectationToys] begin
    router = P.ExpectationFirstOf(ToyExact(), ToyGrid())
    @test router.backends === (ToyExact(), ToyGrid())
    @test router.prefer === ()
    @test router.exclude === ()
    @test P.ExpectationFirstOf(ToyGrid(); prefer = P.ExpectationExact()).prefer === (P.ExpectationExact(),)
    @test P.ExpectationFirstOf(ToyGrid(); exclude = (P.ExpectationStochastic(),)).exclude === (P.ExpectationStochastic(),)
end

@testitem "ExpectationFirstOf: the first backend that supports the pair" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    router = P.ExpectationFirstOf(ToyNothing(), ToyExact(), ToyGrid())
    q = Normal(1.0, 2.0)
    @test P.which_expectation_backend(router, identity, q) === ToyExact()
    @test P.which_expectation_backend(router, log, q) === ToyGrid()
    @test P.which_expectation_backend(router, log, Gamma(1, 1)) === nothing
    @test P.supports_expectation(router, identity, q) === true
    @test P.supports_expectation(router, log, Gamma(1, 1)) === false

    @test P.expectation(router, identity, q) === 1.0 # exactly, from the closed form
    @test P.expectation(router, x -> x^4, q) ≈ 73.0
    @test_throws P.UnsupportedExpectationError P.expectation(router, log, Gamma(1, 1))
end

@testitem "ExpectationFirstOf: prefer and exclude select by label" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    q = Normal(0.0, 1.0)
    sampler = ToySampler(100, StableRNG(1))
    @test P.which_expectation_backend(P.ExpectationFirstOf(ToyGrid(), ToyExact()), identity, q) === ToyGrid()
    @test P.which_expectation_backend(
        P.ExpectationFirstOf(ToyGrid(), ToyExact(); prefer = P.ExpectationExact()), identity, q
    ) === ToyExact()
    # a preference that nobody meets falls back to the order
    @test P.which_expectation_backend(P.ExpectationFirstOf(ToyGrid(), ToyExact(); prefer = P.ExpectationExact()), log, q) ===
        ToyGrid()
    @test P.which_expectation_backend(P.ExpectationFirstOf(sampler, ToyGrid()), log, q) === sampler
    @test P.which_expectation_backend(P.ExpectationFirstOf(sampler, ToyGrid(); exclude = P.ExpectationStochastic()), log, q) ===
        ToyGrid()
    @test P.which_expectation_backend(P.ExpectationFirstOf(sampler; exclude = P.ExpectationStochastic()), log, q) === nothing
end

@testitem "ExpectationFirstOf: the choice is made by inference" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, JET

    router = P.ExpectationFirstOf(ToyNothing(), ToyExact(), ToyGrid())
    q = Normal(1.0, 2.0)
    @test folds_to(
        P.which_expectation_backend, (typeof(router), typeof(identity), typeof(q)), ToyExact()
    )
    @test folds_to(P.supports_expectation, (typeof(router), typeof(log), Gamma{Float64}), false)
    @test @inferred(P.expectation(router, identity, q)) === 1.0
    @test @inferred(P.expectation(router, abs2, q)) == 5.0
    @test_opt P.expectation(router, identity, q)
    @test_opt P.expectation(router, x -> x^4, q)
end

@testitem "ExpectationFirstOf: labels come from the chosen backend" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    router = P.ExpectationFirstOf(ToyExact(), ToyGrid(), ToySampler(10))
    q = Normal(0.0, 1.0)
    @test P.is_expectation_exact(router, identity, q) === true
    @test P.is_expectation_exact(router, log, q) === false
    @test P.is_expectation_exact(router, log, Gamma(1, 1)) === false
    @test folds_to(P.is_expectation_exact, (typeof(router), typeof(identity), typeof(q)), true)
    @test folds_to(P.is_expectation_exact, (typeof(router), typeof(log), typeof(q)), false)

    # backend-level labels are conservative: true only if every backend has the label
    @test P.has_expectation_label(P.ExpectationFirstOf(ToyExact(), ToyExact()), P.ExpectationExact()) === true
    @test P.has_expectation_label(router, P.ExpectationExact()) === false
    @test P.is_expectation_stochastic(router) === false
    @test P.is_expectation_stochastic(P.ExpectationFirstOf(ToySampler(10))) === true
end

@testitem "ExpectationFirstOf: with_expectation_rng reaches every backend, nested routers too" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using StableRNGs

    rng, other = StableRNG(1), StableRNG(2)
    router = P.ExpectationFirstOf(ToyExact(), P.ExpectationFirstOf(ToySampler(10), ToySampler(20, other)))
    filled = P.with_expectation_rng(router, rng)
    @test P.get_expectation_rng(filled.backends[2].backends[1]) === rng
    @test P.get_expectation_rng(filled.backends[2].backends[2]) === other
    @test filled.backends[1] === ToyExact()
    @test allocations(P.with_expectation_rng, router, rng) == 0
end

@testitem "ExpectationFirstOf: prepared expectations keep exact closed forms first" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using Distributions

    router = P.ExpectationFirstOf(ToyExact(), ToyGrid())
    q = Normal(1.0, 2.0)
    prep = P.prepare_expectation(router, q)
    # `identity` routes to the closed form, any other integrand to the prepared point set
    @test P.expectation(prep, identity) === 1.0
    @test P.expectation(prep, x -> x^4) ≈ 73.0
    @test allocations(P.expectation, prep, abs2) == 0
    @test @inferred(P.expectation(prep, identity)) === 1.0

    # nothing supports every integrand: only the closed forms remain
    closed = P.prepare_expectation(P.ExpectationFirstOf(ToyExact()), q)
    @test P.expectation(closed, identity) === 1.0
    @test_throws P.UnsupportedExpectationError P.expectation(closed, x -> x^4)
end

@testitem "ExpectationFirstOf: products route each component on its own" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    # a point-set backend for Gamma only, in Gauss–Legendre fashion via quantiles
    struct GammaGrid end
    P.supports_expectation(::GammaGrid, ::Type, ::Type{<:Gamma}) = true
    P.prepare_expectation(::GammaGrid, q::Gamma) =
        P.ExpectationPointSet(quantile.(q, (0.25, 0.75)), (0.5, 0.5))

    router = P.ExpectationFirstOf(ToyExact(), ToyGrid(), GammaGrid())
    joint = FactorizedJoint((PointMass(2.0), Gamma(2.0, 1.0)))
    @test P.supports_expectation(ToyGrid(), Any, typeof(joint)) === false
    @test P.supports_expectation(router, Any, typeof(joint)) === true
    @test P.is_expectation_exact(router, log, joint) === false
    prep = P.prepare_expectation(router, joint)
    @test P.expectation(prep, ((y, λ),) -> y * λ) ≈
        2 * sum(quantile.(Gamma(2.0, 1.0), (0.25, 0.75))) / 2
    @test P.expectation(router, ((y, λ),) -> y, joint) ≈ 2.0
    @test allocations(P.expectation, prep, ((y, λ),) -> y * λ) == 0
end
