# Independent products of distributions of mixed kinds, represented by `FactorizedJoint`. The
# integrand receives a tuple with one value per component. A backend supports a product when it
# supports every component for any integrand. The product is computed as nested expectations over
# the preparations of the components, and as a product rule when every preparation is a point set.

"""
    ExpectationProductPointSet(sets)

The product rule over a tuple of [`ExpectationPointSet`](@ref)s: every combination of points,
weighted by the product of their weights. The integrand receives a tuple of points. It is the
preparation of a [`FactorizedJoint`](@ref) whose components all have a point set. Otherwise the
preparation computes nested expectations over the components' preparations.
"""
struct ExpectationProductPointSet{S <: Tuple}
    sets::S
end

Base.length(product::ExpectationProductPointSet) = prod(map(length, product.sets))

function expectation(product::ExpectationProductPointSet, f::F) where {F}
    sets = product.sets
    indices = CartesianIndices(map(set -> Base.OneTo(length(set)), sets))
    result = _weight(sets, first(indices)) * f(_point(sets, first(indices)))
    for (k, index) in enumerate(indices)
        k == 1 && continue
        result += _weight(sets, index) * f(_point(sets, index))
    end
    return result
end

_point(sets, index) = map((set, i) -> @inbounds(set.points[i]), sets, Tuple(index))
_weight(sets, index) = prod(map((set, i) -> @inbounds(set.weights[i]), sets, Tuple(index)))

# Nested expectations, `E₁[x₁ ↦ E₂[x₂ ↦ … f((x₁, x₂, …))]]`, over the preparations of the
# components. Any preparation works, such as sigma points or a closed form that takes every integrand.
struct ExpectationNestedProduct{S <: Tuple}
    parts::S
end

function expectation(product::ExpectationNestedProduct, f::F) where {F}
    return nested_expectation(product.parts, (), f)
end

nested_expectation(::Tuple{}, values, f::F) where {F} = f(values)
function nested_expectation(parts::Tuple, values, f::F) where {F}
    rest = Base.tail(parts)
    return expectation(first(parts), x -> nested_expectation(rest, (values..., x), f))
end

# The product rule when every component has a point set, nested expectations otherwise
product_rule(parts::Tuple{Vararg{ExpectationPointSet}}) = ExpectationProductPointSet(parts)
product_rule(parts::Tuple) = ExpectationNestedProduct(parts)

# The part of a component's preparation that the product uses; routers unwrap their own
product_part(prepared) = prepared

prepare_expectation(backend, q::FactorizedJoint{<:Tuple}) = prepare_product(backend, q)

# A product that is not supported is prepared lazily, so that its evaluation throws an
# `UnsupportedExpectationError` naming the integrand
function prepare_product(backend, q::FactorizedJoint{T}) where {T}
    supports_all(backend, T) || return PreparedExpectation(backend, q)
    return product_rule(map(c -> product_part(prepare_expectation(backend, c)), components(q)))
end

function supports_expectation(backend, ::Type, ::Type{FactorizedJoint{T}}) where {T <: Tuple}
    return supports_all(backend, T)
end

function has_expectation_label(
        backend, ::Type, ::Type{FactorizedJoint{T}}, label::ExpectationLabel
    ) where {T <: Tuple}
    return haslabel_all(backend, T, label)
end

# Recursions over the component types, so that inference decides them. An empty product is the
# integrand at `()`: supported, exact and not stochastic.
supports_all(backend, ::Type{Tuple{}}) = true
function supports_all(backend, ::Type{T}) where {T <: Tuple}
    return supports_expectation(backend, Any, Base.tuple_type_head(T)) &&
        supports_all(backend, Base.tuple_type_tail(T))
end

haslabel_all(backend, ::Type{Tuple{}}, label::ExpectationLabel) = true
haslabel_all(backend, ::Type{Tuple{}}, ::ExpectationStochastic) = false
haslabel_all(backend, ::Type{Tuple{H}}, label) where {H} = has_expectation_label(backend, Any, H, label)
function haslabel_all(backend, ::Type{T}, label) where {T <: Tuple}
    return combine_expectation_label(
        label,
        has_expectation_label(backend, Any, Base.tuple_type_head(T), label),
        haslabel_all(backend, Base.tuple_type_tail(T), label),
    )
end
