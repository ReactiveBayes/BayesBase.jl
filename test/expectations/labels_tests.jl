@testitem "labels are false by default" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.ExpectationExact() isa P.ExpectationLabel
    @test P.ExpectationStochastic() isa P.ExpectationLabel
    @test P.has_expectation_label(ToyNothing(), P.ExpectationExact()) === false
    @test P.has_expectation_label(ToyNothing(), P.ExpectationStochastic()) === false
    @test P.has_expectation_label(ToyNothing(), Any, Normal, P.ExpectationExact()) === false
end

@testitem "a backend-level label applies to every supported pair" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.has_expectation_label(ToyExact(), P.ExpectationExact()) === true
    @test P.has_expectation_label(ToyExact(), typeof(identity), Normal, P.ExpectationExact()) === true
    @test P.is_expectation_exact(ToyExact(), identity, Normal(0, 1)) === true
    @test P.is_expectation_stochastic(ToySampler(10), identity, Normal(0, 1)) === true
    @test P.is_expectation_stochastic(ToySampler(10)) === true
    @test P.is_expectation_stochastic(ToyExact()) === false
end

@testitem "a label needs support: unsupported pairs have no labels" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # ToyExact is exact as a backend, but it does not support `log`
    @test P.is_expectation_exact(ToyExact(), log, Normal(0, 1)) === false
    @test P.is_expectation_stochastic(ToySampler(10), identity, Gamma(1, 1)) === false
end

@testitem "per-pair labels override the backend-level default" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.is_expectation_exact(ToyGrid(), identity, Normal(0, 1)) === false
    @test P.is_expectation_exact(ToyGrid(), identity, Categorical([0.5, 0.5])) === true
    @test P.is_expectation_exact(ToyGrid(), identity, PointMass(1.0)) === true
end

@testitem "labels fold to constants" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test folds_to(P.is_expectation_exact, (ToyExact, typeof(identity), Normal{Float64}), true)
    @test folds_to(P.is_expectation_exact, (ToyExact, typeof(log), Normal{Float64}), false)
    @test folds_to(
        P.is_expectation_exact, (ToyGrid, typeof(log), Categorical{Float64, Vector{Float64}}), true
    )
    @test folds_to(
        P.is_expectation_stochastic, (ToySampler{Nothing}, typeof(log), Normal{Float64}), true
    )
end

@testitem "combine_expectation_label: exact needs all parts, stochastic any part" tags = [:expectations] setup = [ExpectationToys] begin

    @test P.combine_expectation_label(P.ExpectationExact(), true, true) === true
    @test P.combine_expectation_label(P.ExpectationExact(), true, false) === false
    @test P.combine_expectation_label(P.ExpectationStochastic(), false, false) === false
    @test P.combine_expectation_label(P.ExpectationStochastic(), true, false) === true

    # a label defined elsewhere is combined conservatively, as "all parts"
    struct Differentiable <: P.ExpectationLabel end
    @test P.combine_expectation_label(Differentiable(), true, false) === false
    @test P.combine_expectation_label(Differentiable(), true, true) === true
end
