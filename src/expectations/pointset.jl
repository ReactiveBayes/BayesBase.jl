"""
    PointSet(points, weights)

A prepared expectation that is a weighted sum, `Σᵢ wᵢ f(xᵢ)`: quadrature, cubature, sigma
points, exact sums over a finite support, or samples with equal weights. `points` and `weights`
are anything indexable from one, tuples included. Point sets are the parts of which product
rules over a [`FactorizedJoint`](@ref) are made.

The sum starts from its first term, so its type follows the integrand: dual numbers pass
through without the point set knowing about them.
"""
struct PointSet{P, W}
    points::P
    weights::W

    function PointSet(points::P, weights::W) where {P, W}
        length(points) == length(weights) || throw(
            DimensionMismatch("$(length(points)) points but $(length(weights)) weights")
        )
        isempty(weights) && throw(ArgumentError("a point set needs at least one point"))
        return new{P, W}(points, weights)
    end
end

Base.length(points::PointSet) = length(points.weights)

function expectation(set::PointSet, f::F) where {F}
    points, weights = set.points, set.weights
    result = weights[1] * f(points[1])
    for i in 2:length(weights)
        @inbounds result += weights[i] * f(points[i])
    end
    return result
end
