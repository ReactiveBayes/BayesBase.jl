@testitem "expectation_score is the gradient of logpdf in params" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, ForwardDiff

    cases = [
        (Normal(1.0, 2.0), 0.3),
        (LogNormal(0.2, 0.7), 1.3),
        (Gamma(2.5, 1.5), 0.7),
        (Exponential(2.0), 0.4),
        (Beta(2.0, 3.0), 0.3),
    ]
    for (q, x) in cases
        family = typeof(q).name.wrapper
        expected = ForwardDiff.gradient(θ -> logpdf(family(θ...), x), collect(Distributions.params(q)))
        @test P.expectation_score(q, x) ≈ expected atol = 1.0e-12
        @test length(@inferred(P.expectation_score(q, x))) == length(Distributions.params(q))
    end

    # a product concatenates the scores of its components at a tuple
    joint = FactorizedJoint((Normal(1.0, 2.0), Gamma(2.5, 1.5)))
    @test P.expectation_score(joint, (0.3, 0.7)) ≈
        vcat(P.expectation_score(Normal(1.0, 2.0), 0.3), P.expectation_score(Gamma(2.5, 1.5), 0.7))

    # a distribution without a score
    @test_throws ArgumentError P.expectation_score(Poisson(2.0), 1)
end

@testitem "expectation_gradient through a point set is the gradient of the expectation" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, ForwardDiff

    # the three-point rule of `ToyGrid` is exact for polynomials up to degree five in x, and the
    # score of a Normal is a polynomial of degree two
    q = Normal(1.0, 2.0)
    exact(θ) = θ[1]^2 + θ[2]^2 # E[x²]
    @test P.expectation_gradient(ToyGrid(), abs2, q) ≈ ForwardDiff.gradient(exact, [1.0, 2.0])
    @test P.expectation_gradient(ToyGrid(), identity, q) ≈ [1.0, 0.0] atol = 1.0e-12

    # with a preparation for q
    prepared = P.prepare_expectation(ToyGrid(), q)
    @test P.expectation(prepared, P.ExpectationScoreProduct(abs2, q)) ≈ [2.0, 4.0]

    # support: a backend that supports every integrand under a family supports every gradient there
    @test P.supports_expectation_gradient(ToyGrid(), abs2, q) === true
    @test P.supports_expectation_gradient(ToyGrid(), Any, Normal{Float64}) === true
    @test P.supports_expectation_gradient(ToyExact(), abs2, q) === false
    @test folds_to(P.supports_expectation_gradient, (ToyGrid, typeof(abs2), Normal{Float64}), true)
end

@testitem "routers choose a closed-form gradient before a numerical one" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    struct GradientClosedForm end

    P.@supports_expectation_gradient function P.expectation_gradient(::GradientClosedForm, ::typeof(abs2), q::Normal)
        return [2 * mean(q), 2 * std(q)]
    end

    q = Normal(1.0, 2.0)
    first = P.ExpectationFirstOf(GradientClosedForm(), ToyGrid())
    @test P.which_expectation_backend(first, P.ExpectationScoreProduct{typeof(abs2), typeof(q)}, typeof(q)) === GradientClosedForm()
    @test P.which_expectation_backend(first, P.ExpectationScoreProduct{typeof(identity), typeof(q)}, typeof(q)) === ToyGrid()
    @test P.expectation_gradient(first, abs2, q) == [2.0, 4.0]
    @test P.expectation_gradient(first, identity, q) ≈ [1.0, 0.0] atol = 1.0e-12

    registry = P.ExpectationBackendRegistry()
    P.register_expectation_backend!(ToyGrid(); registry)
    P.register_expectation_backend!(GradientClosedForm(); registry, priority = -1)
    dynamic = P.ExpectationDynamicRouter(registry)
    @test P.expectation_gradient(dynamic, abs2, q) ≈ [2.0, 4.0]
end

@testitem "expectation_gradient under a product of distributions" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # E[x₁² + 3 x₂] = μ₁² + σ₁² + 3 μ₂, with respect to (μ₁, σ₁, μ₂, σ₂)
    q = FactorizedJoint((Normal(1.0, 2.0), Normal(0.5, 1.0)))
    f = x -> x[1]^2 + 3 * x[2]
    @test P.supports_expectation_gradient(ToyGrid(), f, q) === true
    @test P.expectation_gradient(ToyGrid(), f, q) ≈ [2.0, 4.0, 3.0, 0.0] atol = 1.0e-12
    @test P.expectation_gradient(P.ExpectationFirstOf(ToyGrid()), f, q) ≈ [2.0, 4.0, 3.0, 0.0] atol = 1.0e-12
end

@testitem "unsupported gradients and distributions without a score" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    q = Normal(1.0, 2.0)
    @test_throws P.UnsupportedExpectationError P.expectation_gradient(ToyExact(), abs2, q)
    @test_throws P.UnsupportedExpectationError P.expectation_gradient(P.ExpectationFirstOf(ToyExact()), log, q)
    message = sprint(showerror, P.UnsupportedExpectationError(ToyExact(), P.ExpectationScoreProduct(abs2, q), q))
    @test occursin("cannot compute the gradient of the expectation of `abs2`", message)

    # a backend that supports every integrand under a Categorical, which has no score, does not
    # support gradients there, and called directly it throws
    categorical = Categorical([0.2, 0.8])
    @test P.has_expectation_score(categorical) === false
    @test P.supports_expectation_gradient(ToyGrid(), identity, categorical) === false
    @test P.supports_expectation_gradient(P.ExpectationFirstOf(ToyGrid()), identity, categorical) === false
    @test_throws ArgumentError P.expectation_gradient(ToyGrid(), identity, categorical)
    @test_throws P.UnsupportedExpectationError P.expectation_gradient(P.ExpectationFirstOf(ToyGrid()), identity, categorical)
end

@testitem "has_expectation_score: the families with a score, and products of them" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    for q in (Normal(0.0, 1.0), LogNormal(0.0, 1.0), Gamma(2.0, 1.0), Exponential(1.0), Beta(2.0, 3.0))
        @test P.has_expectation_score(q) === true
    end
    @test P.has_expectation_score(Poisson(1.0)) === false
    @test P.has_expectation_score(FactorizedJoint((Normal(0.0, 1.0), Gamma(2.0, 1.0)))) === true
    @test P.has_expectation_score(FactorizedJoint((Normal(0.0, 1.0), Categorical([0.5, 0.5])))) === false
    @test folds_to(P.has_expectation_score, (Type{FactorizedJoint{Tuple{Normal{Float64}, Gamma{Float64}}}},), true)
end

@testitem "routers pass over a numerical backend for a family without a score" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # a closed-form gradient under a Categorical, which has no score, after a backend that
    # supports every integrand there
    struct CategoricalGradient end
    P.@supports_expectation_gradient P.expectation_gradient(::CategoricalGradient, ::typeof(identity), q::Categorical) = [1.0, 2.0]

    q = Categorical([0.2, 0.8])
    first = P.ExpectationFirstOf(ToyGrid(), CategoricalGradient())
    @test P.supports_expectation_gradient(first, identity, q) === true
    @test P.expectation_gradient(first, identity, q) == [1.0, 2.0]

    registry = P.ExpectationBackendRegistry()
    P.register_expectation_backend!(ToyGrid(); registry, priority = 10)
    P.register_expectation_backend!(CategoricalGradient(); registry)
    @test P.expectation_gradient(P.ExpectationDynamicRouter(registry), identity, q) == [1.0, 2.0]
    @test P.expectation_gradient(P.ExpectationFirstOf(P.ExpectationDynamicRouter(registry)), identity, q) == [1.0, 2.0]
    @test folds_to(P.supports_expectation_gradient, (typeof(first), typeof(identity), typeof(q)), true)
end
