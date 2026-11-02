@testitem "labels are false by default" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.Exact() isa P.ExpectationLabel
    @test P.Stochastic() isa P.ExpectationLabel
    @test P.haslabel(ToyNothing(), P.Exact()) === false
    @test P.haslabel(ToyNothing(), P.Stochastic()) === false
    @test P.haslabel(ToyNothing(), Any, Normal, P.Exact()) === false
end

@testitem "a backend-level label applies to every supported pair" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.haslabel(ToyExact(), P.Exact()) === true
    @test P.haslabel(ToyExact(), typeof(identity), Normal, P.Exact()) === true
    @test P.isexact(ToyExact(), identity, Normal(0, 1)) === true
    @test P.isstochastic(ToySampler(10), identity, Normal(0, 1)) === true
    @test P.isstochastic(ToySampler(10)) === true
    @test P.isstochastic(ToyExact()) === false
end

@testitem "a label needs support: unsupported pairs have no labels" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # ToyExact is exact as a backend, but it does not support `log`
    @test P.isexact(ToyExact(), log, Normal(0, 1)) === false
    @test P.isstochastic(ToySampler(10), identity, Gamma(1, 1)) === false
end

@testitem "per-pair labels override the backend-level default" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.isexact(ToyGrid(), identity, Normal(0, 1)) === false
    @test P.isexact(ToyGrid(), identity, Categorical([0.5, 0.5])) === true
    @test P.isexact(ToyGrid(), identity, PointMass(1.0)) === true
end

@testitem "labels fold to constants" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test folds_to(P.isexact, (ToyExact, typeof(identity), Normal{Float64}), true)
    @test folds_to(P.isexact, (ToyExact, typeof(log), Normal{Float64}), false)
    @test folds_to(
        P.isexact, (ToyGrid, typeof(log), Categorical{Float64, Vector{Float64}}), true
    )
    @test folds_to(
        P.isstochastic, (ToySampler{Nothing}, typeof(log), Normal{Float64}), true
    )
end

@testitem "combine_label: exact needs all parts, stochastic any part" tags = [:expectations] setup = [ExpectationToys] begin

    @test P.combine_label(P.Exact(), true, true) === true
    @test P.combine_label(P.Exact(), true, false) === false
    @test P.combine_label(P.Stochastic(), false, false) === false
    @test P.combine_label(P.Stochastic(), true, false) === true

    # a label defined elsewhere is combined conservatively, as "all parts"
    struct Differentiable <: P.ExpectationLabel end
    @test P.combine_label(Differentiable(), true, false) === false
    @test P.combine_label(Differentiable(), true, true) === true
end
