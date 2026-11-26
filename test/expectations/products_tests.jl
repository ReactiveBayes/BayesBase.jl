@testitem "FactorizedJoint: supported when every component is, for any integrand" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    joint = FactorizedJoint((Categorical([0.3, 0.7]), Normal(0.0, 1.0), PointMass(2.0)))
    @test P.supports_expectation(ToyGrid(), Any, typeof(joint)) === true
    @test P.supports_expectation(ToyGrid(), log, joint) === true
    @test P.supports_expectation(
        ToyGrid(), Any, FactorizedJoint{Tuple{Normal{Float64}, Gamma{Float64}}}
    ) === false
    @test P.supports_expectation(ToyExact(), identity, FactorizedJoint((Normal(0.0, 1.0),))) === false
    @test folds_to(P.supports_expectation, (ToyGrid, Type{Any}, Type{typeof(joint)}), true)
end

@testitem "FactorizedJoint: labels combine over components" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    discrete = FactorizedJoint((Categorical([0.3, 0.7]), PointMass(2.0)))
    mixed = FactorizedJoint((Categorical([0.3, 0.7]), Normal(0.0, 1.0)))
    @test P.is_expectation_exact(ToyGrid(), log, discrete) === true
    @test P.is_expectation_exact(ToyGrid(), log, mixed) === false
    @test folds_to(P.is_expectation_exact, (ToyGrid, typeof(log), typeof(discrete)), true)
    @test folds_to(P.is_expectation_exact, (ToyGrid, typeof(log), typeof(mixed)), false)
end

@testitem "FactorizedJoint: the integrand receives one value per component" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    z, x, y = Categorical([0.3, 0.7]), Normal(1.0, 2.0), PointMass(2.0)
    joint = FactorizedJoint((z, x, y))
    # E[z · x² + y] = E[z] E[x²] + y = 1.7 · 5 + 2
    f((zi, xi, yi)) = zi * xi^2 + yi
    @test P.expectation(ToyGrid(), f, joint) ≈ 1.7 * 5 + 2
    prep = P.prepare_expectation(ToyGrid(), joint)
    @test prep isa P.ExpectationProductPointSet
    @test P.expectation(prep, f) ≈ 1.7 * 5 + 2
    @test length(prep) == 2 * 3 * 1
end

@testitem "FactorizedJoint: evaluation does not allocate and passes duals" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using Distributions, ForwardDiff

    joint = FactorizedJoint((Categorical([0.3, 0.7]), Normal(1.0, 2.0), PointMass(2.0)))
    prep = P.prepare_expectation(ToyGrid(), joint)
    f((zi, xi, yi)) = zi * xi^2 + yi
    @test allocations(P.expectation, prep, f) == 0

    # d/dθ E[θ z x²] = E[z] E[x²]
    derivative(p, θ) =
        ForwardDiff.derivative(t -> P.expectation(p, ((zi, xi, yi),) -> t * zi * xi^2), θ)
    @test derivative(prep, 1.0) ≈ 1.7 * 5
    @test allocations(derivative, prep, 1.0) == 0
end

@testitem "FactorizedJoint: preparations that are not point sets give nested expectations" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # sigma points that are not an `ExpectationPointSet`: exact for E[x] and E[x²] under a Normal
    struct Sigma end
    struct SigmaSet{T}
        m::T
        s::T
    end
    P.supports_expectation(::Sigma, ::Type, ::Type{<:Normal}) = true
    P.prepare_expectation(::Sigma, q::Normal) = SigmaSet(mean(q), std(q))
    P.expectation(set::SigmaSet, f::F) where {F} = (f(set.m - set.s) + f(set.m + set.s)) / 2

    joint = FactorizedJoint((Normal(1.0, 1.0), Normal(2.0, 0.5)))
    f(x) = x[1]^2 * x[2] # E = (1 + 1) * 2
    @test P.supports_expectation(Sigma(), f, joint) === true
    @test P.prepare_expectation(Sigma(), joint) isa P.ExpectationNestedProduct
    @test P.expectation(Sigma(), f, joint) ≈ 4.0

    # a router uses the first backend that supports each component, whatever its preparation
    router = P.ExpectationFirstOf(Sigma(), ToyGrid())
    @test P.expectation(router, f, joint) ≈ 4.0
    mixed = FactorizedJoint((Normal(1.0, 1.0), Categorical([0.5, 0.5])))
    @test P.expectation(router, x -> x[1]^2 * x[2], mixed) ≈ 2 * 1.5
    @test allocations(P.expectation, P.prepare_expectation(router, mixed), f) == 0

    # a backend that computes every integrand itself, without a preparation
    sampler = ToySampler(1, Random.Xoshiro(1))
    @test P.expectation(sampler, x -> x[1] + x[2], FactorizedJoint((Normal(3.0, 0.0), Normal(4.0, 0.0)))) == 7.0
end

@testitem "FactorizedJoint: an unsupported product throws UnsupportedExpectationError" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # ToyExact has closed forms for two integrands only, not every integrand
    joint = FactorizedJoint((Normal(0.0, 1.0),))
    @test P.supports_expectation(ToyExact(), identity, joint) === false
    @test_throws P.UnsupportedExpectationError P.expectation(ToyExact(), identity, joint)
    @test_throws P.UnsupportedExpectationError P.expectation(P.ExpectationFirstOf(ToyExact()), identity, joint)
    @test_throws P.UnsupportedExpectationError P.expectation(P.prepare_expectation(P.ExpectationFirstOf(ToyExact()), joint), identity)
    @test_throws P.UnsupportedExpectationError P.expectation(P.ExpectationDynamicRouter(P.ExpectationBackendRegistry()), identity, joint)

    # components in a vector are not supported
    vector = FactorizedJoint([Normal(0.0, 1.0), Normal(1.0, 1.0)])
    @test P.supports_expectation(ToyGrid(), identity, vector) === false
    @test_throws P.UnsupportedExpectationError P.expectation(ToyGrid(), identity, vector)
    @test_throws P.UnsupportedExpectationError P.expectation(P.ExpectationFirstOf(ToyGrid()), identity, vector)
end

@testitem "FactorizedJoint: an empty product is the integrand at ()" tags = [:expectations] setup = [ExpectationToys] begin
    empty = FactorizedJoint(())
    @test P.supports_expectation(ToyGrid(), Any, typeof(empty)) === true
    @test P.expectation(ToyGrid(), x -> 3.0, empty) == 3.0
    @test P.is_expectation_exact(ToyGrid(), sin, empty) === true
    @test P.is_expectation_stochastic(ToySampler(1), sin, empty) === false
end
