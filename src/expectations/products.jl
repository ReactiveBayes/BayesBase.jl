# Independent products of distributions of mixed kinds, represented by `FactorizedJoint`. The integrand receives a tuple with one value per component.
# A backend supports a product when it supports every component for any integrand, and the
# product is computed as a product rule over the point sets of the components.

"""
    ProductPointSet(sets)

The product rule over a tuple of [`PointSet`](@ref)s: every combination of points, weighted
by the product of their weights. The integrand receives a tuple of points.
"""
struct ProductPointSet{S <: Tuple}
    sets::S
end

Base.length(product::ProductPointSet) = prod(map(length, product.sets))

function expectation(product::ProductPointSet, f::F) where {F}
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

# The point set behind a preparation; a product rule needs one for every component
pointset(set::PointSet) = set
function pointset(prepared)
    return throw(
        ArgumentError(
            "a product rule needs a point set for every component, but the preparation is a $(typeof(prepared))",
        ),
    )
end

prepare_expectation(backend, q::FactorizedJoint) = prepare_product(backend, q)

function prepare_product(backend, q)
    return ProductPointSet(
        map(c -> pointset(prepare_expectation(backend, c)), components(q))
    )
end

function supports(backend, ::Type, ::Type{FactorizedJoint{T}}) where {T <: Tuple}
    return supports_all(backend, T)
end

function haslabel(
        backend, ::Type, ::Type{FactorizedJoint{T}}, label::ExpectationLabel
    ) where {T <: Tuple}
    return haslabel_all(backend, T, label)
end

# Recursions over the component types, so that inference decides them
supports_all(backend, ::Type{Tuple{}}) = true
function supports_all(backend, ::Type{T}) where {T <: Tuple}
    return supports(backend, Any, Base.tuple_type_head(T)) &&
        supports_all(backend, Base.tuple_type_tail(T))
end

haslabel_all(backend, ::Type{Tuple{H}}, label) where {H} = haslabel(backend, Any, H, label)
function haslabel_all(backend, ::Type{T}, label) where {T <: Tuple}
    return combine_label(
        label,
        haslabel(backend, Any, Base.tuple_type_head(T), label),
        haslabel_all(backend, Base.tuple_type_tail(T), label),
    )
end
