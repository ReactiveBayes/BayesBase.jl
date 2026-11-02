# The expectation of a log-density, `E_q[log p(x)]`, has no name of its own in the protocol: its
# integrand is `Base.Fix1(logpdf, p)`, on which a closed form dispatches.

@testitem "a closed form of E[log p] dispatches on Base.Fix1(logpdf, p)" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    # E_{N(m, s²)}[log N(x | μ, σ²)] = -log(2πσ²)/2 - ((m - μ)² + s²) / (2σ²)
    struct ToyLogNormalDensity end
    const LogNormalDensity = Base.Fix1{typeof(logpdf), <:Normal}
    P.supports_expectation(::ToyLogNormalDensity, ::Type{<:LogNormalDensity}, ::Type{<:Normal}) = true
    function P.expectation(::ToyLogNormalDensity, f::LogNormalDensity, q::Normal)
        μ, σ = mean(f.x), std(f.x)
        return -log(2π * σ^2) / 2 - ((mean(q) - μ)^2 + var(q)) / (2 * σ^2)
    end

    p, q = Normal(1.0, 2.0), Normal(0.5, 0.5)
    integrand = Base.Fix1(logpdf, p)
    @test integrand(0.3) == logpdf(p, 0.3)
    expected = -log(2π * 4.0) / 2 - (0.25 + 0.25) / 8
    @test P.expectation(ToyLogNormalDensity(), integrand, q) ≈ expected
    # Through a router, the closed form comes first; any other integrand goes to the grid.
    router = P.ExpectationFirstOf(ToyLogNormalDensity(), ToyGrid())
    @test P.which_expectation_backend(router, integrand, q) === ToyLogNormalDensity()
    @test P.which_expectation_backend(router, abs2, q) === ToyGrid()
    @test P.expectation(router, integrand, q) ≈ expected
end
