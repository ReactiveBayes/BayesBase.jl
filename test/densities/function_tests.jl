
@testitem "ContinuousUnivariateLogPdf: Constructor" begin
    import DomainSets: FullSpace

    f = (x) -> -x^2
    d1 = ContinuousUnivariateLogPdf(f)
    d2 = ContinuousUnivariateLogPdf(FullSpace(), f)

    @test typeof(d1) === typeof(d2)
    @test eltype(d1) === Float64
    @test eltype(d2) === Float64
    @test paramfloattype(d1) === Float64
    @test samplefloattype(d1) === Float64
    @test paramfloattype(d2) === Float64
    @test samplefloattype(d2) === Float64

    @test_throws AssertionError ContinuousUnivariateLogPdf(FullSpace()^2, f)
end

@testitem "ContinuousUnivariateLogPdf: Intentional errors" begin
    dist = ContinuousUnivariateLogPdf((x) -> x)
    @test_throws ErrorException mean(dist)
    @test_throws ErrorException median(dist)
    @test_throws ErrorException mode(dist)
    @test_throws ErrorException var(dist)
    @test_throws ErrorException std(dist)
    @test_throws ErrorException cov(dist)
    @test_throws ErrorException invcov(dist)
    @test_throws ErrorException entropy(dist)
    @test_throws ErrorException precision(dist)
end

@testitem "ContinuousUnivariateLogPdf: pdf/logpdf" begin
    import DomainSets: FullSpace, HalfLine

    d1 = ContinuousUnivariateLogPdf(FullSpace(), (x) -> -x^2)

    f32_points1 = range(Float32(-10.0), Float32(10.0); length=50)
    f64_points1 = range(-10.0, 10.0; length=50)
    bf_points1 = range(BigFloat(-10.0), BigFloat(10.0); length=50)
    points1 = vcat(f32_points1, f64_points1, bf_points1)

    @test all(map(p -> -p^2 == d1(p), points1))
    @test all(map(p -> -p^2 == logpdf(d1, p), points1))
    @test all(map(p -> exp(-p^2) == pdf(d1, p), points1))
    @test all(map(p -> -p^2 == d1([p]), points1))
    @test all(map(p -> -p^2 == logpdf(d1, [p]), points1))
    @test all(map(p -> exp(-p^2) == pdf(d1, [p]), points1))

    d2 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> -x^4)

    f32_points2 = range(Float32(0.0), Float32(10.0); length=50)
    f64_points2 = range(0.0, 10.0; length=50)
    bf_points2 = range(BigFloat(0.0), BigFloat(10.0); length=50)
    points2 = vcat(f32_points2, f64_points2, bf_points2)

    @test all(map(p -> -p^4 == d2(p), points2))
    @test all(map(p -> -p^4 == logpdf(d2, p), points2))
    @test all(map(p -> exp(-p^4) == pdf(d2, p), points2))
    @test all(map(p -> -p^4 == d2([p]), points2))
    @test all(map(p -> -p^4 == logpdf(d2, [p]), points2))
    @test all(map(p -> exp(-p^4) == pdf(d2, [p]), points2))

    @test_throws AssertionError d2(-1.0)
    @test_throws AssertionError logpdf(d2, -1.0)
    @test_throws AssertionError pdf(d2, -1.0)
    @test_throws AssertionError d2([-1.0])
    @test_throws AssertionError logpdf(d2, [-1.0])
    @test_throws AssertionError pdf(d2, [-1.0])

    @test_throws AssertionError d2(Float32(-1.0))
    @test_throws AssertionError logpdf(d2, Float32(-1.0))
    @test_throws AssertionError pdf(d2, Float32(-1.0))
    @test_throws AssertionError d2([Float32(-1.0)])
    @test_throws AssertionError logpdf(d2, [Float32(-1.0)])
    @test_throws AssertionError pdf(d2, [Float32(-1.0)])

    @test_throws AssertionError d2(BigFloat(-1.0))
    @test_throws AssertionError logpdf(d2, BigFloat(-1.0))
    @test_throws AssertionError pdf(d2, BigFloat(-1.0))
    @test_throws AssertionError d2([BigFloat(-1.0)])
    @test_throws AssertionError logpdf(d2, [BigFloat(-1.0)])
    @test_throws AssertionError pdf(d2, [BigFloat(-1.0)])

    d3 = ContinuousUnivariateLogPdf(FullSpace(Float32), (x) -> -x^2)

    @test all(map(p -> -p^2 == d3(p), points1))
    @test all(map(p -> -p^2 == logpdf(d3, p), points1))
    @test all(map(p -> exp(-p^2) == pdf(d3, p), points1))
    @test all(map(p -> -p^2 == d3([p]), points1))
    @test all(map(p -> -p^2 == logpdf(d3, [p]), points1))
    @test all(map(p -> exp(-p^2) == pdf(d3, [p]), points1))

    d4 = ContinuousUnivariateLogPdf(FullSpace(BigFloat), (x) -> -x^2)

    @test all(map(p -> -p^2 == d4(p), points1))
    @test all(map(p -> -p^2 == logpdf(d4, p), points1))
    @test all(map(p -> exp(-p^2) == pdf(d4, p), points1))
    @test all(map(p -> -p^2 == d4([p]), points1))
    @test all(map(p -> -p^2 == logpdf(d4, [p]), points1))
    @test all(map(p -> exp(-p^2) == pdf(d4, [p]), points1))

    d5 = ContinuousUnivariateLogPdf(HalfLine{Float32}(), (x) -> -x^2)

    @test all(map(p -> -p^2 == d5(p), points2))
    @test all(map(p -> -p^2 == logpdf(d5, p), points2))
    @test all(map(p -> exp(-p^2) == pdf(d5, p), points2))
    @test all(map(p -> -p^2 == d5([p]), points2))
    @test all(map(p -> -p^2 == logpdf(d5, [p]), points2))
    @test all(map(p -> exp(-p^2) == pdf(d5, [p]), points2))

    d6 = ContinuousUnivariateLogPdf(HalfLine{BigFloat}(), (x) -> -x^2)

    @test all(map(p -> -p^2 == d6(p), points2))
    @test all(map(p -> -p^2 == logpdf(d6, p), points2))
    @test all(map(p -> exp(-p^2) == pdf(d6, p), points2))
    @test all(map(p -> -p^2 == d6([p]), points2))
    @test all(map(p -> -p^2 == logpdf(d6, [p]), points2))
    @test all(map(p -> exp(-p^2) == pdf(d6, [p]), points2))
end

@testitem "ContinuousUnivariateLogPdf: test domain in logpdf" begin
    import DomainSets: FullSpace, HalfLine

    d1 = ContinuousUnivariateLogPdf(FullSpace(), (x) -> -x^2)
    d2 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> -x^4)

    # This also throws a warning in stdout
    @test_throws AssertionError logpdf(d1, [1.0, 1.0])
    @test_throws AssertionError logpdf(d2, [1.0, 1.0])
end

@testitem "ContinuousUnivariateLogPdf: support" begin
    import DomainSets: FullSpace, HalfLine

    d1 = ContinuousUnivariateLogPdf(FullSpace(), (x) -> 1.0)
    @test 1.0 ∈ support(d1)
    @test -1.0 ∈ support(d1)

    d2 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> 1.0)
    @test 1.0 ∈ support(d2)
    @test -1.0 ∉ support(d2)
end

@testitem "ContinuousUnivariateLogPdf: vague" begin
    d = vague(ContinuousUnivariateLogPdf)

    @test typeof(d) <: ContinuousUnivariateLogPdf
    @test d(rand()) ≈ 0
end

@testitem "ContinuousUnivariateLogPdf: prod" begin
    import DomainSets: FullSpace, HalfLine

    dist = ContinuousUnivariateLogPdf(FullSpace(), (x) -> 2.0 * -x^2)
    d2 = ContinuousUnivariateLogPdf(FullSpace(), (x) -> 3.0 * -x^2)

    product = prod(GenericProd(), dist, d2)
    pt1 = ContinuousUnivariateLogPdf(FullSpace(), (x) -> logpdf(dist, x) + logpdf(d2, x))

    @test variate_form(typeof(product)) === variate_form(typeof(dist))
    @test variate_form(typeof(product)) === variate_form(typeof(d2))
    @test value_support(typeof(product)) === value_support(typeof(dist))
    @test value_support(typeof(product)) === value_support(typeof(d2))
    @test support(product) === support(dist)
    @test support(product) === support(d2)

    for x in rand(10)
        @test isapprox(logpdf(product, x), logpdf(pt1, x))
        @test isapprox(pdf(product, x), pdf(pt1, x))
    end

    result = ContinuousUnivariateLogPdf(HalfLine(), (x) -> 2.0 * -x^2)
    d4 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> 3.0 * -x^2)

    pr2 = prod(GenericProd(), result, d4)
    pt2 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> logpdf(result, x) + logpdf(d4, x))

    @test variate_form(typeof(pr2)) === variate_form(typeof(result))
    @test variate_form(typeof(pr2)) === variate_form(typeof(d4))
    @test value_support(typeof(pr2)) === value_support(typeof(result))
    @test value_support(typeof(pr2)) === value_support(typeof(d4))
    @test support(pr2) === support(result)
    @test support(pr2) === support(d4)

    for x in rand(10)
        @test isapprox(logpdf(pr2, x), logpdf(pt2, x))
        @test isapprox(pdf(pr2, x), pdf(pt2, x))
    end

    d5 = ContinuousUnivariateLogPdf(FullSpace(), (x) -> 2.0 * -x^2)
    d6 = ContinuousUnivariateLogPdf(HalfLine(), (x) -> 2.0 * -x^2)

    @test logpdf(prod(GenericProd(), d5, d6), 1.0) ≈ -4.0
    @test_throws AssertionError logpdf(prod(GenericProd(), d5, d6), -1.0) # supports are different
end

@testitem "ContinuousUnivariateLogPdf: vectorised-prod" begin
    import DomainSets: FullSpace

    f = (x) -> 2.0 * -x^2
    dist = ContinuousUnivariateLogPdf(FullSpace(), f)
    result = ContinuousUnivariateLogPdf(FullSpace(), (x) -> 3 * f(x))
    product = prod(GenericProd(), prod(GenericProd(), dist, dist), dist)

    @test product isa LinearizedProductOf

    @test variate_form(typeof(product)) === variate_form(typeof(dist))
    @test variate_form(typeof(product)) === variate_form(typeof(result))
    @test value_support(typeof(product)) === value_support(typeof(dist))
    @test value_support(typeof(product)) === value_support(typeof(result))
    @test support(product) === support(dist)
    @test support(product) === support(result)

    for x in rand(10)
        @test logpdf(product, x) ≈ logpdf(result, x)
        @test pdf(product, x) ≈ pdf(result, x)
    end

    # Test internal side-effects
    another_product = prod(GenericProd(), product, dist)

    for x in rand(10)
        @test logpdf(product, x) ≈ logpdf(result, x)
        @test pdf(product, x) ≈ pdf(result, x)

        @test logpdf(another_product, x) ≈ (logpdf(product, x) + logpdf(dist, x))
        @test pdf(another_product, x) ≈ (pdf(product, x) * pdf(dist, x))
    end
end

@testitem "ContinuousUnivariateLogPdf: convert" begin
    import DomainSets: FullSpace

    d = FullSpace()
    l = (x) -> 1.0

    c = convert(ContinuousUnivariateLogPdf, d, l)
    @test typeof(c) <: ContinuousUnivariateLogPdf

    c2 = convert(ContinuousUnivariateLogPdf, c)
    @test typeof(c2) <: ContinuousUnivariateLogPdf
end

@testitem "ContinuousMultivariateLogPdf: Constructor" begin
    import DomainSets: FullSpace

    f = (x) -> -x'x
    dist = ContinuousMultivariateLogPdf(2, f)
    d2 = ContinuousMultivariateLogPdf(FullSpace()^2, f)

    @test typeof(dist) === typeof(d2)
    @test paramfloattype(dist) === Float64
    @test samplefloattype(dist) === Float64
    @test paramfloattype(d2) === Float64
    @test samplefloattype(d2) === Float64

    @test_throws AssertionError ContinuousMultivariateLogPdf(FullSpace(), f)
    @test_throws MethodError ContinuousMultivariateLogPdf(f)
end

@testitem "ContinuousMultivariateLogPdf: Intentional errors" begin
    dist = ContinuousMultivariateLogPdf(2, (x) -> -x'x)
    @test_throws ErrorException mean(dist)
    @test_throws ErrorException median(dist)
    @test_throws ErrorException mode(dist)
    @test_throws ErrorException var(dist)
    @test_throws ErrorException std(dist)
    @test_throws ErrorException cov(dist)
    @test_throws ErrorException invcov(dist)
    @test_throws ErrorException entropy(dist)
    @test_throws ErrorException precision(dist)
end

@testitem "ContinuousMultivariateLogPdf: pdf/logpdf" begin
    import DomainSets: FullSpace, HalfLine

    dist = ContinuousMultivariateLogPdf(FullSpace()^2, (x) -> -x'x)

    f32_points1 = range(Float32(-10.0), Float32(10.0); length=5)
    f64_points1 = range(-10.0, 10.0; length=5)
    bf_points1 = range(BigFloat(-10.0), BigFloat(10.0); length=5)

    points1 = vcat(
        vec(map(collect, Iterators.product(f32_points1, f32_points1))),
        vec(map(collect, Iterators.product(f64_points1, f64_points1))),
        vec(map(collect, Iterators.product(bf_points1, bf_points1))),
    )

    @test all(map(p -> -p'p == dist(p), points1))
    @test all(map(p -> -p'p == logpdf(dist, p), points1))
    @test all(map(p -> exp(-p'p) == pdf(dist, p), points1))

    d2 = ContinuousMultivariateLogPdf(HalfLine()^2, (x) -> -x'x / 4)

    f32_points2 = range(Float32(0.0), Float32(10.0); length=5)
    f64_points2 = range(0.0, 10.0; length=5)
    bf_points2 = range(BigFloat(0.0), BigFloat(10.0); length=5)

    points2 = vcat(
        vec(map(collect, Iterators.product(f32_points2, f32_points2))),
        vec(map(collect, Iterators.product(f64_points2, f64_points2))),
        vec(map(collect, Iterators.product(bf_points2, bf_points2))),
    )

    @test all(map(p -> -p'p / 4 == d2(p), points2))
    @test all(map(p -> -p'p / 4 == logpdf(d2, p), points2))
    @test all(map(p -> exp(-p'p / 4) == pdf(d2, p), points2))
end

@testitem "ContinuousMultivariateLogPdf: test domain in logpdf" begin
    import DomainSets: FullSpace, HalfLine

    for dim in (2, 3, 4)
        dist = ContinuousMultivariateLogPdf(FullSpace()^dim, (x) -> -x'x)
        d2 = ContinuousMultivariateLogPdf(HalfLine()^dim, (x) -> -x'x)

        # This also throws a warning in stdout
        @test_logs (:warn, r".*incompatible combination.*") @test_throws AssertionError logpdf(
            dist, ones(dim + 1)
        )
        @test_logs (:warn, r".*incompatible combination.*") @test_throws AssertionError logpdf(
            d2, ones(dim + 1)
        )
    end
end

@testitem "ContinuousMultivariateLogPdf: vague" begin
    for k in 2:5
        d = vague(ContinuousMultivariateLogPdf, k)

        @test typeof(d) <: ContinuousMultivariateLogPdf
        @test d(rand(k)) ≈ 0
    end
end

@testitem "ContinuousMultivariateLogPdf: prod" begin
    import DomainSets: FullSpace, HalfLine

    dist = ContinuousMultivariateLogPdf(FullSpace()^2, (x) -> 2.0 * -x'x)
    d2 = ContinuousMultivariateLogPdf(FullSpace()^2, (x) -> 3.0 * -x'x)

    pr1 = prod(GenericProd(), dist, d2)
    pt1 = ContinuousMultivariateLogPdf(
        FullSpace()^2, (x) -> logpdf(dist, x) + logpdf(d2, x)
    )

    @test variate_form(typeof(pr1)) === variate_form(typeof(dist))
    @test variate_form(typeof(pr1)) === variate_form(typeof(d2))
    @test value_support(typeof(pr1)) === value_support(typeof(dist))
    @test value_support(typeof(pr1)) === value_support(typeof(d2))
    @test support(pr1) === support(dist)
    @test support(pr1) === support(d2)

    for x in [randn(2) for _ in 1:10]
        @test isapprox(logpdf(pr1, x), logpdf(pt1, x))
        @test isapprox(pdf(pr1, x), pdf(pt1, x))
    end

    result = ContinuousMultivariateLogPdf(HalfLine()^2, (x) -> 2.0 * -x'x)
    d4 = ContinuousMultivariateLogPdf(HalfLine()^2, (x) -> 3.0 * -x'x)

    pr2 = prod(GenericProd(), result, d4)
    pt2 = ContinuousMultivariateLogPdf(
        HalfLine()^2, (x) -> logpdf(result, x) + logpdf(d4, x)
    )

    @test variate_form(typeof(pr2)) === variate_form(typeof(result))
    @test variate_form(typeof(pr2)) === variate_form(typeof(d4))
    @test value_support(typeof(pr2)) === value_support(typeof(result))
    @test value_support(typeof(pr2)) === value_support(typeof(d4))
    @test support(pr2) === support(result)
    @test support(pr2) === support(d4)

    for x in [rand(2) for _ in 1:10]
        @test isapprox(logpdf(pr2, x), logpdf(pt2, x))
        @test isapprox(pdf(pr2, x), pdf(pt2, x))
    end

    d5 = ContinuousMultivariateLogPdf(FullSpace()^2, (x) -> 2.0 * -x'x)
    d6 = ContinuousMultivariateLogPdf(HalfLine()^2, (x) -> 2.0 * -x'x)
    @test_throws AssertionError logpdf(prod(GenericProd(), d5, d6), [1.0, -1.0]) # domains are incompatible
end

@testitem "ContinuousMultivariateLogPdf: vectorised-prod" begin
    import DomainSets: FullSpace, HalfLine

    f = (x) -> 2.0 * -x'x
    dist = ContinuousMultivariateLogPdf(FullSpace()^2, f)
    result = ContinuousMultivariateLogPdf(FullSpace()^2, (x) -> 3 * f(x))

    product = prod(GenericProd(), prod(GenericProd(), dist, dist), dist)

    @test product isa LinearizedProductOf
    @test variate_form(typeof(product)) === variate_form(typeof(dist))
    @test variate_form(typeof(product)) === variate_form(typeof(result))
    @test value_support(typeof(product)) === value_support(typeof(dist))
    @test value_support(typeof(product)) === value_support(typeof(result))
    @test support(product) === support(dist)
    @test support(product) === support(result)

    for x in [rand(2) for _ in 1:10]
        @test pdf(product, x) ≈ pdf(result, x)
        @test logpdf(product, x) ≈ logpdf(result, x)
    end

    # Test internal side-effects
    another_product = prod(GenericProd(), product, dist)

    for x in [rand(2) for _ in 1:10]
        @test logpdf(product, x) ≈ logpdf(result, x)
        @test pdf(product, x) ≈ pdf(result, x)

        @test logpdf(another_product, x) ≈ (logpdf(product, x) + logpdf(dist, x))
        @test pdf(another_product, x) ≈ (pdf(product, x) * pdf(dist, x))
    end
end

@testitem "ContinuousMultivariateLogPdf: convert" begin
    import DomainSets: FullSpace, HalfLine

    for k in 2:5
        d = FullSpace()^k
        l = (x) -> 0

        c = convert(ContinuousMultivariateLogPdf, d, l)
        @test typeof(c) <: ContinuousMultivariateLogPdf

        c2 = convert(ContinuousMultivariateLogPdf, c)
        @test typeof(c2) <: ContinuousMultivariateLogPdf
    end
end
