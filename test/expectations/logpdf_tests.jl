@testitem "Logpdf evaluates the log-density of a distribution" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions

    integrand = P.Logpdf(Normal(1.0, 2.0))
    @test integrand(0.5) == logpdf(Normal(1.0, 2.0), 0.5)
    @test integrand.dist == Normal(1.0, 2.0)
    @test @inferred(integrand(0.5)) isa Float64
end

@testitem "Logpdf works for anything with a logpdf, not only distributions" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, BayesBase

    product = BayesBase.ProductOf(Normal(0.0, 1.0), Normal(1.0, 1.0))
    @test P.Logpdf(product)(0.3) ≈
        logpdf(Normal(0.0, 1.0), 0.3) + logpdf(Normal(1.0, 1.0), 0.3)

    struct OnlyLogpdf end
    BayesBase.logpdf(::OnlyLogpdf, x) = -x^2
    @test P.Logpdf(OnlyLogpdf())(2.0) == -4.0
end

@testitem "Logpdf passes dual numbers through" tags = [:expectations] setup = [ExpectationToys] begin
    using Distributions, ForwardDiff

    # d/dx log N(x | 1, 2²) = -(x - 1) / 4
    @test ForwardDiff.derivative(P.Logpdf(Normal(1.0, 2.0)), 3.0) ≈ -0.5
end
