using StatsFuns: softmax
using LinearAlgebra

export MixtureDistribution

"""
    MixtureDistribution(components, weights)

A custom mixture distribution implementation, parameterized by:
* `C` type family of the mixture
* `CT` the type for the weights

This implementation solves:
* [Distributions.jl Issue 1669](https://github.com/JuliaStats/Distributions.jl/issues/1669)
* [ReactiveMP.jl Issue 253](https://github.com/biaslab/ReactiveMP.jl/issues/253)

"""
struct MixtureDistribution{C,CT<:Real}
    components::Vector{C}
    weights::Vector{CT}

    function MixtureDistribution(cs::Vector{C}, w::Vector{CT}) where {C,CT}
        length(cs) == length(w) ||
            error("The number of components does not match the length of prior.")
        @assert all(>=(0), w) "weight vector contains negative entries."
        @assert sum(w) ≈ 1 "weight vector is not normalized."
        return new{C,CT}(cs, w)
    end
end

BayesBase.components(mixture::MixtureDistribution) = mixture.components
BayesBase.component(mixture::MixtureDistribution, k::Int) = mixture.components[k]
BayesBase.weights(mixture::MixtureDistribution) = mixture.weights

function BayesBase.mean(mixture::MixtureDistribution)
    return dot(weights(mixture), mean.(components(mixture)))
end

function BayesBase.var(mixture::MixtureDistribution)
    accumulated =
        mapreduce(+, zip(weights(mixture), components(mixture))) do (weight, component)
            return (weight * (var(component) + mean(component)^2))
        end
    return accumulated - mean(mixture)^2
end

function BayesBase.logpdf(mixture::MixtureDistribution, x)
    return log(pdf(mixture, x))
end

function BayesBase.pdf(mixture::MixtureDistribution, x)
    return mapreduce(+, zip(weights(mixture), components(mixture))) do (weight, component)
        return (weight * pdf(component, x))
    end
end

function BayesBase.default_prod_rule(::Type{<:MixtureDistribution}, ::Type{T}) where {T}
    return PreserveTypeProd(MixtureDistribution)
end

function Base.prod(
    ::PreserveTypeProd{MixtureDistribution}, left::MixtureDistribution, right::Any
)

    # get weights and components
    w = weights(left)
    dists = components(left)

    # get new distributions
    dists_new = map(dist -> prod(default_prod_rule(dist, right), dist, right), dists)

    # get scales
    logscales = map(
        (dist, dist_new) -> compute_logscale(dist_new, dist, right), dists, dists_new
    )

    # compute updated weights
    logscales_new = log.(w) + logscales

    # return mixture distribution
    return MixtureDistribution(dists_new, softmax(logscales_new))
end

function BayesBase.compute_logscale(
    new_dist::MixtureDistribution, left_dist::MixtureDistribution, right_dist::Any
)

    # get prior weights and components
    w = left_dist.weights
    dists = left_dist.components

    # get new distributions
    dists_new = map(dist -> prod(ProdAnalytical(), dist, right_dist), dists)

    # get scales
    logscales = map(
        (dist, dist_new) -> compute_logscale(dist_new, dist, right_dist), dists, dists_new
    )

    # compute updated weights
    logscales_new = log.(w) + logscales

    return logsumexp(logscales_new)
end

function BayesBase.compute_logscale(
    new_dist::MixtureDistribution, left_dist::Any, right_dist::MixtureDistribution
)
    return compute_logscale(new_dist, right_dist, left_dist)
end