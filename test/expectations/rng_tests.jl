@testitem "with_expectation_rng and get_expectation_rng: identity and nothing by default" tags = [:expectations] setup = [ExpectationToys] begin
    using StableRNGs

    rng = StableRNG(1)
    @test P.with_expectation_rng(ToyExact(), rng) === ToyExact()
    @test P.get_expectation_rng(ToyExact()) === nothing
end

@testitem "with_expectation_rng fills an unset RNG and keeps a set one" tags = [:expectations] setup = [ExpectationToys] begin
    using StableRNGs

    rng, other = StableRNG(1), StableRNG(2)
    @test P.get_expectation_rng(P.with_expectation_rng(ToySampler(10), rng)) === rng
    @test P.get_expectation_rng(P.with_expectation_rng(ToySampler(10, other), rng)) === other
end

@testitem "a backend with an RNG computes; without one it says what to do" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    sampler = P.with_expectation_rng(ToySampler(10_000), StableRNG(1))
    @test P.expectation(sampler, identity, Normal(3.0, 1.0)) ≈ 3.0 atol = 0.05
    @test_throws ArgumentError P.expectation(ToySampler(10), identity, Normal(0, 1))
end
