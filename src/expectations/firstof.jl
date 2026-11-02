"""
    ExpectationFirstOf(backends...; prefer = (), exclude = ())

A router, itself a backend: each expectation goes to the first of `backends` that supports the
pair `(f, q)` and has none of the labels in `exclude`. With `prefer`, the first backend that
also has all the labels in `prefer` wins, when there is one. The choice depends on types only,
so inference makes it and nothing is decided at run time.

A product of distributions ([`FactorizedJoint`](@ref)) is routed
component by component.
"""
struct ExpectationFirstOf{B <: Tuple, P <: Tuple, X <: Tuple}
    backends::B
    prefer::P
    exclude::X
end

function ExpectationFirstOf(backends...; prefer = (), exclude = ())
    return ExpectationFirstOf(backends, aslabels(prefer), aslabels(exclude))
end

aslabels(label::ExpectationLabel) = (label,)
aslabels(labels::Tuple) = labels

# The position of the chosen backend, or zero
@inline function choice(router::ExpectationFirstOf, ::Type{F}, ::Type{Q}) where {F, Q}
    preferred = find(router.backends, F, Q, router.prefer, router.exclude)
    return if preferred == 0 && !isempty(router.prefer)
        find(router.backends, F, Q, (), router.exclude)
    else
        preferred
    end
end

@inline find(::Tuple{}, ::Type, ::Type, need, avoid) = 0
@inline function find(backends::Tuple, ::Type{F}, ::Type{Q}, need, avoid) where {F, Q}
    backend = first(backends)
    admissible =
        supports_expectation(backend, F, Q) &&
        hasall(backend, F, Q, need) &&
        !hasany(backend, F, Q, avoid)
    admissible && return 1
    rest = find(Base.tail(backends), F, Q, need, avoid)
    return rest == 0 ? 0 : rest + 1
end

@inline hasall(backend, ::Type, ::Type, ::Tuple{}) = true
@inline hasall(backend, ::Type{F}, ::Type{Q}, labels::Tuple) where {F, Q} =
    has_expectation_label(backend, F, Q, first(labels)) && hasall(backend, F, Q, Base.tail(labels))

@inline hasany(backend, ::Type, ::Type, ::Tuple{}) = false
@inline hasany(backend, ::Type{F}, ::Type{Q}, labels::Tuple) where {F, Q} =
    has_expectation_label(backend, F, Q, first(labels)) || hasany(backend, F, Q, Base.tail(labels))

function which_expectation_backend(router::ExpectationFirstOf, ::Type{F}, ::Type{Q}) where {F, Q}
    i = choice(router, F, Q)
    return i == 0 ? nothing : router.backends[i]
end

supports_expectation(router::ExpectationFirstOf, ::Type{F}, ::Type{Q}) where {F, Q} = choice(router, F, Q) != 0

function has_expectation_label(
        router::ExpectationFirstOf, ::Type{F}, ::Type{Q}, label::ExpectationLabel
    ) where {F, Q}
    i = choice(router, F, Q)
    return i == 0 ? false : has_expectation_label(router.backends[i], F, Q, label)
end

# For every pair: only if every backend has the label, which is conservative
function has_expectation_label(router::ExpectationFirstOf, label::ExpectationLabel)
    return all(backend -> has_expectation_label(backend, label), router.backends)
end

function expectation(router::ExpectationFirstOf, f::F, q) where {F}
    backend = which_expectation_backend(router, typeof(f), typeof(q))
    isnothing(backend) && throw(UnsupportedExpectationError(router, f, q))
    return expectation(backend, f, q)
end

function with_expectation_rng(router::ExpectationFirstOf, rng)
    return ExpectationFirstOf(withrngs(router.backends, rng), router.prefer, router.exclude)
end

# A recursion instead of `map` with a closure, which allocates for nested routers on Julia 1.10
withrngs(::Tuple{}, rng) = ()
function withrngs(backends::Tuple, rng)
    return (with_expectation_rng(first(backends), rng), withrngs(Base.tail(backends), rng)...)
end

# A prepared `ExpectationFirstOf`: the preparation of the backend chosen for every integrand (`F =
# Any`), and the router, so that an integrand with a closed form still goes to it
struct PreparedExpectationFirstOf{R, Q, G}
    router::R
    q::Q
    general::G
end

function prepare_expectation(router::ExpectationFirstOf, q)
    general = which_expectation_backend(router, Any, typeof(q))
    return PreparedExpectationFirstOf(
        router, q, isnothing(general) ? nothing : prepare_expectation(general, q)
    )
end

function expectation(prepared::PreparedExpectationFirstOf, f::F) where {F}
    i = choice(prepared.router, typeof(f), typeof(prepared.q))
    i == 0 && throw(UnsupportedExpectationError(prepared.router, f, prepared.q))
    i == choice(prepared.router, Any, typeof(prepared.q)) &&
        return expectation(prepared.general, f)
    return expectation(prepared.router.backends[i], f, prepared.q)
end

expectation_pointset(prepared::PreparedExpectationFirstOf) = expectation_pointset(prepared.general)

# Products: each component goes to its own backend
function supports_expectation(router::ExpectationFirstOf, ::Type, ::Type{FactorizedJoint{T}}) where {T <: Tuple}
    return supports_all(router, T)
end
function has_expectation_label(
        router::ExpectationFirstOf, ::Type, ::Type{FactorizedJoint{T}}, label::ExpectationLabel
    ) where {T <: Tuple}
    return haslabel_all(router, T, label)
end
function which_expectation_backend(router::ExpectationFirstOf, ::Type, ::Type{FactorizedJoint{T}}) where {T <: Tuple}
    return supports_all(router, T) ? router : nothing
end
prepare_expectation(router::ExpectationFirstOf, q::FactorizedJoint) = prepare_product(router, q)
function expectation(router::ExpectationFirstOf, f::F, q::FactorizedJoint) where {F}
    return expectation(prepare_expectation(router, q), f)
end
