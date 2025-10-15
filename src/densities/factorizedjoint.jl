export FactorizedJoint

"""
    FactorizedJoint(components)

`FactorizedJoint` represents a joint distribution of independent random variables. 
Use `component()` function or square-brackets indexing to access the marginal distribution for individual variables.
Use `components()` function to get a tuple of multipliers.
"""
struct FactorizedJoint{T}
    multipliers::T
end

BayesBase.components(joint::FactorizedJoint) = joint.multipliers

Base.@propagate_inbounds function BayesBase.component(joint::FactorizedJoint, i::Int)
    return getindex(joint, i)
end
Base.@propagate_inbounds function Base.getindex(joint::FactorizedJoint, i::Int)
    return getindex(components(joint), i)
end

Base.length(joint::FactorizedJoint) = length(joint.multipliers)

function Base.isapprox(x::FactorizedJoint, y::FactorizedJoint; kwargs...)
    return length(x) === length(y) && all(
        tuple -> isapprox(tuple[1], tuple[2]; kwargs...), zip(components(x), components(y))
    )
end

BayesBase.entropy(joint::FactorizedJoint) = mapreduce(entropy, +, components(joint))

function BayesBase.paramfloattype(joint::FactorizedJoint)
    return BayesBase.paramfloattype(BayesBase.components(joint))
end

function BayesBase.convert_paramfloattype(::Type{T}, joint::FactorizedJoint) where {T}
    return FactorizedJoint(
        map(e -> BayesBase.convert_paramfloattype(T, e), BayesBase.components(joint))
    )
end
