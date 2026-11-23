@testitem "@supports_expectation defines the method and declares its support" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    struct MacroMoments end

    P.@supports_expectation function P.expectation(::MacroMoments, ::typeof(identity), q::Gamma)
        return shape(q) * scale(q)
    end

    # the short form, `where` parameters with bounds and a return type
    P.@supports_expectation (P.expectation(::MacroMoments, f::F, q::Normal{T})::T) where {F <: typeof(abs2), T <: Real} = mean(q)^2 + var(q)

    @test P.expectation(MacroMoments(), identity, Gamma(2.0, 3.0)) == 6.0
    @test P.expectation(MacroMoments(), abs2, Normal(1.0, 2.0)) == 5.0
    @test P.supports_expectation(MacroMoments(), identity, Gamma(2.0, 3.0)) === true
    @test P.supports_expectation(MacroMoments(), abs2, Normal(1.0, 2.0)) === true
    @test P.supports_expectation(MacroMoments(), abs2, Normal(1.0f0, 2.0f0)) === true
    @test P.supports_expectation(MacroMoments(), log, Gamma(2.0, 3.0)) === false
    @test P.supports_expectation(MacroMoments(), identity, Normal(1.0, 2.0)) === false
    @test P.supports_expectation(MacroMoments(), Any, Gamma{Float64}) === false

    # decided by inference
    @test folds_to(P.supports_expectation, (MacroMoments, typeof(identity), Gamma{Float64}), true)
    @test folds_to(P.supports_expectation, (MacroMoments, typeof(log), Gamma{Float64}), false)

    # an integrand without a type declares every integrand
    struct MacroEverything end
    P.@supports_expectation P.expectation(::MacroEverything, f, q::Categorical) = sum(probs(q) .* f.(1:ncategories(q)))
    @test P.supports_expectation(MacroEverything(), Any, Categorical{Float64, Vector{Float64}}) === true
    @test P.expectation(MacroEverything(), abs2, Categorical([0.5, 0.5])) == 2.5
end

@testitem "@supports_expectation_gradient is reached directly and through routers" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    struct MacroGradient end

    P.@supports_expectation_gradient function P.expectation_gradient(::MacroGradient, ::typeof(identity), q::Gamma{T}) where {T}
        return [scale(q), shape(q)]
    end

    q = Gamma(2.0, 3.0)
    @test P.expectation_gradient(MacroGradient(), identity, q) == [3.0, 2.0]
    @test P.expectation_gradient(P.ExpectationFirstOf(MacroGradient()), identity, q) == [3.0, 2.0]
    @test P.expectation(MacroGradient(), P.ExpectationScoreProduct(identity, q), q) == [3.0, 2.0]
    @test P.supports_expectation_gradient(MacroGradient(), identity, q) === true
    @test P.supports_expectation_gradient(MacroGradient(), log, q) === false
    @test P.supports_expectation(MacroGradient(), identity, q) === false # only the gradient
    @test folds_to(P.supports_expectation_gradient, (MacroGradient, typeof(identity), Gamma{Float64}), true)
end

@testitem "a docstring before the macros documents the method" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    struct MacroDocumented end

    "The closed form of `E[sin(x)]` under a centred Normal."
    P.@supports_expectation P.expectation(::MacroDocumented, ::typeof(sin), q::Normal) = 0.0

    # The docstrings of `expectation` defined in this module, read from the registry because what
    # `@doc` returns for a function with several docstrings differs between Julia versions
    documented = Base.Docs.meta(@__MODULE__)[Base.Docs.Binding(P, :expectation)]
    signature = Tuple{MacroDocumented, typeof(sin), Normal}
    @test haskey(documented.docs, signature)
    @test occursin("E[sin(x)]", only(documented.docs[signature].text))
end

@testitem "the macros work where only they are imported" tags = [:expectations] begin
    module MacroUser
    using Distributions
    import BayesBase: @supports_expectation, @supports_expectation_gradient, expectation, expectation_gradient

    struct Backend end

    @supports_expectation expectation(::Backend, ::typeof(cos), q::Normal) = exp(-var(q) / 2) * cos(mean(q))
    @supports_expectation_gradient expectation_gradient(::Backend, ::typeof(cos), q::Normal) = [0.0, 0.0]
    end

    using BayesBase, Distributions
    @test BayesBase.supports_expectation(MacroUser.Backend(), cos, Normal(0.0, 1.0)) === true
    @test BayesBase.supports_expectation_gradient(MacroUser.Backend(), cos, Normal(0.0, 1.0)) === true
    @test BayesBase.expectation(MacroUser.Backend(), cos, Normal(0.0, 1.0)) ≈ exp(-1 / 2)
end

@testitem "the macros reject what they cannot read" tags = [:expectations] begin
    using BayesBase

    expand(expression) = macroexpand(@__MODULE__, expression)
    rejected = [
        :(BayesBase.@supports_expectation mean(::B, f, q) = 1),
        :(BayesBase.@supports_expectation expectation(::B, f) = 1),
        :(BayesBase.@supports_expectation expectation(::B, f, q, r) = 1),
        :(BayesBase.@supports_expectation expectation(::B, f, q; k = 1) = 1),
        :(BayesBase.@supports_expectation expectation(::B, f, q = 1) = 1),
        :(BayesBase.@supports_expectation struct B end),
        :(BayesBase.@supports_expectation_gradient expectation(::B, f, q) = 1),
    ]
    for expression in rejected
        @test_throws ArgumentError expand(expression)
    end
end
