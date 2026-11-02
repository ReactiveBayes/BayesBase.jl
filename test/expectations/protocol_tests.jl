@testitem "supports: false by default, instances forward to types" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.supports(ToyNothing(), Any, Normal) === false
    @test P.supports(ToyNothing(), log, Normal(0, 1)) === false
    @test P.supports(ToyExact(), identity, Normal(0, 1)) === true
    @test P.supports(ToyExact(), typeof(identity), Normal) === true
    @test P.supports(ToyExact(), log, Normal(0, 1)) === false

    # `Any` as the integrand type asks whether every integrand is supported
    @test P.supports(ToyGrid(), Any, Normal) === true
    @test P.supports(ToyExact(), Any, Normal) === false

    # decided by inference
    @test folds_to(P.supports, (ToyExact, typeof(identity), Normal{Float64}), true)
    @test folds_to(P.supports, (ToyExact, typeof(log), Normal{Float64}), false)
end

@testitem "which_backend of a plain backend is itself or nothing" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test P.which_backend(ToyExact(), identity, Normal(0, 1)) === ToyExact()
    @test P.which_backend(ToyExact(), log, Normal(0, 1)) === nothing
    @test P.which_backend(ToyGrid(), Any, Normal) === ToyGrid()
    @test @inferred(P.which_backend(ToyExact(), identity, Normal(0, 1))) === ToyExact()
end

@testitem "expectation and prepare_expectation of a closed-form backend" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    q = Normal(1.0, 2.0)
    @test P.expectation(ToyExact(), identity, q) == 1.0
    @test P.expectation(ToyExact(), abs2, q) == 5.0

    # the default preparation wraps the backend and the distribution and forwards
    prep = P.prepare_expectation(ToyExact(), q)
    @test prep isa P.Prepared
    @test P.expectation(prep, identity) == 1.0
    @test P.expectation(prep, abs2) == 5.0
end

@testitem "expectation of a point-set backend goes through its preparation" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    q = Normal(1.0, 2.0)
    # three-point Gauss–Hermite is exact for polynomials up to degree 5
    @test P.expectation(ToyGrid(), x -> x^4, q) ≈ 1 + 6 * 4 + 3 * 16
    @test P.expectation(ToyGrid(), identity, Categorical([0.2, 0.8])) ≈ 1.8
    @test P.expectation(P.prepare_expectation(ToyGrid(), q), abs2) ≈ 5.0
end

@testitem "an unsupported pair throws NoBackendError, not a stack overflow" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    @test_throws P.NoBackendError P.expectation(ToyNothing(), identity, Normal(0, 1))
    @test_throws P.NoBackendError P.expectation(ToyExact(), log, Normal(0, 1))
    @test_throws P.NoBackendError P.expectation(
        P.prepare_expectation(ToyNothing(), Normal(0, 1)), identity
    )

    message = sprint(showerror, P.NoBackendError(ToyNothing(), log, Normal(0, 1)))
    @test occursin("ToyNothing", message)
    @test occursin("log", message)
    @test occursin("Normal", message)
end
