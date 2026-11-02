@testitem "FactorizedJoint: supported when every component is, for any integrand" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    joint = FactorizedJoint((Categorical([0.3, 0.7]), Normal(0.0, 1.0), PointMass(2.0)))
    @test P.supports(ToyGrid(), Any, typeof(joint)) === true
    @test P.supports(ToyGrid(), log, joint) === true
    @test P.supports(
        ToyGrid(), Any, FactorizedJoint{Tuple{Normal{Float64}, Gamma{Float64}}}
    ) === false
    @test P.supports(ToyExact(), identity, FactorizedJoint((Normal(0.0, 1.0),))) === false
    @test folds_to(P.supports, (ToyGrid, Type{Any}, Type{typeof(joint)}), true)
end

@testitem "FactorizedJoint: labels combine over components" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    discrete = FactorizedJoint((Categorical([0.3, 0.7]), PointMass(2.0)))
    mixed = FactorizedJoint((Categorical([0.3, 0.7]), Normal(0.0, 1.0)))
    @test P.isexact(ToyGrid(), log, discrete) === true
    @test P.isexact(ToyGrid(), log, mixed) === false
    @test folds_to(P.isexact, (ToyGrid, typeof(log), typeof(discrete)), true)
    @test folds_to(P.isexact, (ToyGrid, typeof(log), typeof(mixed)), false)
end

@testitem "FactorizedJoint: the integrand receives one value per component" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    z, x, y = Categorical([0.3, 0.7]), Normal(1.0, 2.0), PointMass(2.0)
    joint = FactorizedJoint((z, x, y))
    # E[z · x² + y] = E[z] E[x²] + y = 1.7 · 5 + 2
    f((zi, xi, yi)) = zi * xi^2 + yi
    @test P.expectation(ToyGrid(), f, joint) ≈ 1.7 * 5 + 2
    prep = P.prepare_expectation(ToyGrid(), joint)
    @test prep isa P.ProductPointSet
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

@testitem "FactorizedJoint: a component without a point set is an error" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # ToyExact prepares a lazy `Prepared`, which cannot be part of a product rule
    @test_throws ArgumentError P.prepare_expectation(
        ToyExact(), FactorizedJoint((Normal(0.0, 1.0),))
    )
end
