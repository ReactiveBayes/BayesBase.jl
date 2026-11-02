@testitem "withrng and getrng: identity and nothing by default" tags = [:expectations] setup = [ExpectationToys] begin
    using StableRNGs

    rng = StableRNG(1)
    @test P.withrng(ToyExact(), rng) === ToyExact()
    @test P.getrng(ToyExact()) === nothing
end

@testitem "withrng fills an unset RNG and keeps a set one" tags = [:expectations] setup = [ExpectationToys] begin
    using StableRNGs

    rng, other = StableRNG(1), StableRNG(2)
    @test P.getrng(P.withrng(ToySampler(10), rng)) === rng
    @test P.getrng(P.withrng(ToySampler(10, other), rng)) === other
end

@testitem "a backend with an RNG computes; without one it says what to do" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, StableRNGs

    sampler = P.withrng(ToySampler(10_000), StableRNG(1))
    @test P.expectation(sampler, identity, Normal(3.0, 1.0)) ≈ 3.0 atol = 0.05
    @test_throws ArgumentError P.expectation(ToySampler(10), identity, Normal(0, 1))
end
