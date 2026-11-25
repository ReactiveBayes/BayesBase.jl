"""
    ExpectationPointSet(points, weights)

A prepared expectation that is a weighted sum, `Σᵢ wᵢ f(xᵢ)`. Quadrature and cubature rules,
sigma points, exact sums over a finite support and samples with equal weights all have this
form. `points` and `weights` can be any collections indexed from one, including tuples. Product
rules over a [`FactorizedJoint`](@ref) are built from the point sets of the components, see
[`BayesBase.ExpectationProductPointSet`](@ref).

The sum starts from its first term, so its type follows the values of the integrand. For
example, an integrand that returns dual numbers needs no special handling.
"""
struct ExpectationPointSet{P, W}
    points::P
    weights::W

    function ExpectationPointSet(points::P, weights::W) where {P, W}
        length(points) == length(weights) || throw(
            DimensionMismatch("$(length(points)) points but $(length(weights)) weights")
        )
        isempty(weights) && throw(ArgumentError("a point set needs at least one point"))
        return new{P, W}(points, weights)
    end
end

Base.length(points::ExpectationPointSet) = length(points.weights)

function expectation(set::ExpectationPointSet, f::F) where {F}
    points, weights = set.points, set.weights
    result = weights[1] * f(points[1])
    for i in 2:length(weights)
        @inbounds result += weights[i] * f(points[i])
    end
    return result
end
