
@testitem "Mixture Distribution: creation" begin
    using Distributions

    component1 = Normal(0.1, 0.3)
    component2 = Normal(2, 2.6)
    w = [0.3, 0.7]

    dist = MixtureDistribution([component1, component2], w)

    @test mean(dist) ≈ 0.3 * 0.1 + 0.7 * 2
    @test var(dist) ≈ 0.3 * (0.3^2 + 0.1^2) + 0.7 * (2.6^2 + 2^2) - mean(dist)^2

    @test weights(dist) == w
    @test components(dist) == [component1, component2]
    @test component(dist, 2) == component2

    @test_throws ErrorException MixtureDistribution(
        [component1, component2], [0.4, 0.4, 0.2]
    )
    @test_throws AssertionError MixtureDistribution([component1, component2], [0.4, 0.4])
    @test_throws AssertionError MixtureDistribution([component1, component2], [-0.5, 1.5])
end

@testitem "Mixture Distribution: parameters - different distributions" begin
    using Distributions

    component1 = Beta(0.1, 0.3)
    component2 = Normal(2, 2.6)
    w = [0.3, 0.7]

    dist = MixtureDistribution([component1, component2], w)

    @test mean(dist) ≈ 0.3 * (0.1 / (0.1 + 0.3)) + 0.7 * 2
    @test var(dist) ≈
        0.3 * ((0.1 * 0.3) / ((0.1 + 0.3)^2 * (0.1 + 0.3 + 1)) + (0.1 / (0.1 + 0.3))^2) +
          0.7 * (2.6^2 + 2^2) - mean(dist)^2
end

# resolves issue #253
@testitem "Mixture Distribution: creation with arbitrary objects" begin
    component1 = 0.1
    component2 = 2
    w = [0.7, 0.3]

    dist = MixtureDistribution([component1, component2], w)

    @test weights(dist) == w
    @test components(dist) == [component1, component2]
    @test component(dist, 2) == component2
end

@testitem "Mixture Distribution: moments with 3 components" begin
    using Distributions

    component1 = Normal(0.3, sqrt(inv(2)))
    component2 = Normal(0.5, sqrt(1.0))
    component3 = Normal(0.7, sqrt(inv(2)))
    w = [0.1, 0.3, 0.6]

    dist = MixtureDistribution([component1, component2, component3], w)

    @test mean(dist) ≈ 0.3 * 0.1 + 0.5 * 0.3 + 0.7 * 0.6
    @test var(dist) ≈
        0.1 * (1 / 2 + 0.3^2) + 0.3 * (1.0 + 0.5^2) + 0.6 * (0.7^2 + 1 / 2) - mean(dist)^2
end

@testitem "Mixture Distribution: pdf" begin
    using Distributions

    component1 = Beta(0.1, 0.3)
    component2 = Normal(2, 2.6)
    w = [0.3, 0.7]

    dist = MixtureDistribution([component1, component2], w)
    @test pdf(dist, 0.5) ≈ 0.3 * pdf(component1, 0.5) + 0.7 * pdf(component2, 0.5)
    @test pdf(dist, 0) ≈ 0.3 * pdf(component1, 0) + 0.7 * pdf(component2, 0)
    @test pdf(dist, 5) ≈ 0.3 * pdf(component1, 5) + 0.7 * pdf(component2, 5)
end

@testitem "Mixture Distribution: prod custom normal" begin
    using LinearAlgebra

    struct NormalMeanVariance
        mean
        var
    end

    function BayesBase.default_prod_rule(
        ::Type{NormalMeanVariance}, ::Type{NormalMeanVariance}
    )
        return PreserveTypeProd(NormalMeanVariance)
    end

    BayesBase.mean(d::NormalMeanVariance) = d.mean
    BayesBase.var(d::NormalMeanVariance) = d.var

    function Base.:(==)(left::NormalMeanVariance, right::NormalMeanVariance)
        return (left.mean == right.mean) && (left.var == right.var)
    end

    function Base.prod(
        ::PreserveTypeProd{NormalMeanVariance},
        left::NormalMeanVariance,
        right::NormalMeanVariance,
    )
        μ = (mean(left) * var(right) + mean(right) * var(left)) / (var(right) + var(left))
        v = (var(left) * var(right)) / (var(left) + var(right))
        return NormalMeanVariance(μ, v)
    end

    function BayesBase.compute_logscale(
        ::NormalMeanVariance, left::NormalMeanVariance, right::NormalMeanVariance
    )
        m_left, v_left = mean_var(left)
        m_right, v_right = mean_var(right)
        v = v_left + v_right
        m = m_left - m_right
        return -(logdet(v) + log(2π)) / 2 - m^2 / v / 2
    end

    component1 = NormalMeanVariance(3, 1)
    component2 = NormalMeanVariance(2, 4)
    w = [0.3, 0.7]
    dist = MixtureDistribution([component1, component2], w)

    for new_dist in
        (prod(GenericProd(), dist, component1), prod(GenericProd(), component1, dist))
        sf1 = 0.3 * sqrt(1 / (2π * (1 + 1)))
        sf2 = 0.7 * sqrt(1 / (2π * (1 + 4))) * exp(-(3 - 2)^2 / 5 / 2)
        p = sf1 / (sf1 + sf2)

        @test component(new_dist, 1) == NormalMeanVariance(3, 1 / 2)
        @test component(new_dist, 2) == NormalMeanVariance(2.8, 0.8)
        @test weights(new_dist) ≈ [p, 1 - p]
    end
end

@testitem "Mixture Distribution: Multivariate Normal" begin
    using Distributions

    component1 = MvNormal([1.0, 1.0], [1.0 0.5; 0.5 1.0])
    component2 = MvNormal([2.0, 2.0], [1.0 0.5; 0.5 1.0])
    w = [0.3, 0.7]

    dist = MixtureDistribution([component1, component2], w)

    mean_dist = mean(dist)
    @test mean_dist ≈ 0.3 * [1.0, 1.0] + 0.7 * [2.0, 2.0]
    @test var(dist) ≈
        0.3 * ([1.0, 1.0] + [1.0, 1.0])+0.7 * ([1.0, 1.0] + [4.0, 4.0]) - mean_dist .^ 2
    @test cov(dist) ≈
        0.3 * (
        [1.0 0.5; 0.5 1.0] + [1.0, 1.0]*[1.0, 1.0]'
    )+0.7 * ([1.0 0.5; 0.5 1.0] + [2.0, 2.0]*[2.0, 2.0]') - mean_dist*mean_dist'
end
