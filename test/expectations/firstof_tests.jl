@testitem "FirstOf: construction" tags = [:expectations] setup = [ExpectationToys] begin
    router = P.FirstOf(ToyExact(), ToyGrid())
    @test router.backends === (ToyExact(), ToyGrid())
    @test router.prefer === ()
    @test router.exclude === ()
    @test P.FirstOf(ToyGrid(); prefer = P.Exact()).prefer === (P.Exact(),)
    @test P.FirstOf(ToyGrid(); exclude = (P.Stochastic(),)).exclude === (P.Stochastic(),)
end

@testitem "FirstOf: the first backend that supports the pair" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    router = P.FirstOf(ToyNothing(), ToyExact(), ToyGrid())
    q = Normal(1.0, 2.0)
    @test P.which_backend(router, identity, q) === ToyExact()
    @test P.which_backend(router, log, q) === ToyGrid()
    @test P.which_backend(router, log, Gamma(1, 1)) === nothing
    @test P.supports(router, identity, q) === true
    @test P.supports(router, log, Gamma(1, 1)) === false

    @test P.expectation(router, identity, q) === 1.0 # exactly, from the closed form
    @test P.expectation(router, x -> x^4, q) ≈ 73.0
    @test_throws P.NoBackendError P.expectation(router, log, Gamma(1, 1))
end

@testitem "FirstOf: prefer and exclude select by label" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    q = Normal(0.0, 1.0)
    sampler = ToySampler(100, StableRNG(1))
    @test P.which_backend(P.FirstOf(ToyGrid(), ToyExact()), identity, q) === ToyGrid()
    @test P.which_backend(
        P.FirstOf(ToyGrid(), ToyExact(); prefer = P.Exact()), identity, q
    ) === ToyExact()
    # a preference that nobody meets falls back to the order
    @test P.which_backend(P.FirstOf(ToyGrid(), ToyExact(); prefer = P.Exact()), log, q) ===
        ToyGrid()
    @test P.which_backend(P.FirstOf(sampler, ToyGrid()), log, q) === sampler
    @test P.which_backend(P.FirstOf(sampler, ToyGrid(); exclude = P.Stochastic()), log, q) ===
        ToyGrid()
    @test P.which_backend(P.FirstOf(sampler; exclude = P.Stochastic()), log, q) === nothing
end

@testitem "FirstOf: the choice is made by inference" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, JET

    router = P.FirstOf(ToyNothing(), ToyExact(), ToyGrid())
    q = Normal(1.0, 2.0)
    @test folds_to(
        P.which_backend, (typeof(router), typeof(identity), typeof(q)), ToyExact()
    )
    @test folds_to(P.supports, (typeof(router), typeof(log), Gamma{Float64}), false)
    @test @inferred(P.expectation(router, identity, q)) === 1.0
    @test @inferred(P.expectation(router, abs2, q)) == 5.0
    @test_opt P.expectation(router, identity, q)
    @test_opt P.expectation(router, x -> x^4, q)
end

@testitem "FirstOf: labels come from the chosen backend" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    router = P.FirstOf(ToyExact(), ToyGrid(), ToySampler(10))
    q = Normal(0.0, 1.0)
    @test P.isexact(router, identity, q) === true
    @test P.isexact(router, log, q) === false
    @test P.isexact(router, log, Gamma(1, 1)) === false
    @test folds_to(P.isexact, (typeof(router), typeof(identity), typeof(q)), true)
    @test folds_to(P.isexact, (typeof(router), typeof(log), typeof(q)), false)

    # backend-level labels are conservative: true only if every backend has the label
    @test P.haslabel(P.FirstOf(ToyExact(), ToyExact()), P.Exact()) === true
    @test P.haslabel(router, P.Exact()) === false
    @test P.isstochastic(router) === false
    @test P.isstochastic(P.FirstOf(ToySampler(10))) === true
end

@testitem "FirstOf: withrng reaches every backend, nested routers too" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using StableRNGs

    rng, other = StableRNG(1), StableRNG(2)
    router = P.FirstOf(ToyExact(), P.FirstOf(ToySampler(10), ToySampler(20, other)))
    filled = P.withrng(router, rng)
    @test P.getrng(filled.backends[2].backends[1]) === rng
    @test P.getrng(filled.backends[2].backends[2]) === other
    @test filled.backends[1] === ToyExact()
    @test allocations(P.withrng, router, rng) == 0
end

@testitem "FirstOf: prepared expectations keep exact closed forms first" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using Distributions

    router = P.FirstOf(ToyExact(), ToyGrid())
    q = Normal(1.0, 2.0)
    prep = P.prepare_expectation(router, q)
    # `identity` routes to the closed form, any other integrand to the prepared point set
    @test P.expectation(prep, identity) === 1.0
    @test P.expectation(prep, x -> x^4) ≈ 73.0
    @test allocations(P.expectation, prep, abs2) == 0
    @test @inferred(P.expectation(prep, identity)) === 1.0

    # nothing supports every integrand: only the closed forms remain
    closed = P.prepare_expectation(P.FirstOf(ToyExact()), q)
    @test P.expectation(closed, identity) === 1.0
    @test_throws P.NoBackendError P.expectation(closed, x -> x^4)
end

@testitem "FirstOf: products route each component on its own" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    # a point-set backend for Gamma only, in Gauss–Legendre fashion via quantiles
    struct GammaGrid end
    P.supports(::GammaGrid, ::Type, ::Type{<:Gamma}) = true
    P.prepare_expectation(::GammaGrid, q::Gamma) =
        P.PointSet(quantile.(q, (0.25, 0.75)), (0.5, 0.5))

    router = P.FirstOf(ToyExact(), ToyGrid(), GammaGrid())
    joint = FactorizedJoint((PointMass(2.0), Gamma(2.0, 1.0)))
    @test P.supports(ToyGrid(), Any, typeof(joint)) === false
    @test P.supports(router, Any, typeof(joint)) === true
    @test P.isexact(router, log, joint) === false
    prep = P.prepare_expectation(router, joint)
    @test P.expectation(prep, ((y, λ),) -> y * λ) ≈
        2 * sum(quantile.(Gamma(2.0, 1.0), (0.25, 0.75))) / 2
    @test P.expectation(router, ((y, λ),) -> y, joint) ≈ 2.0
    @test allocations(P.expectation, prep, ((y, λ),) -> y * λ) == 0
end
