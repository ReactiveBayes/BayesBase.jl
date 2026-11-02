"""
    ExpectationBackendRegistry()

Backends registered for automatic choice by an [`ExpectationDynamicRouter`](@ref), with priorities.
"""
struct ExpectationBackendRegistry
    backends::Vector{Any}
    priorities::Vector{Int}
end

ExpectationBackendRegistry() = ExpectationBackendRegistry(Any[], Int[])

Base.length(registry::ExpectationBackendRegistry) = length(registry.backends)
Base.isempty(registry::ExpectationBackendRegistry) = isempty(registry.backends)

const EXPECTATION_BACKEND_REGISTRY = ExpectationBackendRegistry()

"""
    register_expectation_backend!(backend; priority = 0, registry = the global registry)

Make `backend` available to [`ExpectationDynamicRouter`](@ref)s. Registering a backend again changes
its priority. Packages call this in `__init__`.
"""
function register_expectation_backend!(
        backend; priority::Integer = 0, registry::ExpectationBackendRegistry = EXPECTATION_BACKEND_REGISTRY
    )
    i = findfirst(isequal(backend), registry.backends)
    if isnothing(i)
        push!(registry.backends, backend)
        push!(registry.priorities, priority)
    else
        registry.priorities[i] = priority
    end
    return registry
end

"""
    ExpectationDynamicRouter(registry = the global registry)

A router that chooses among registered backends at run time. Ranking: exact for the pair
first, then not stochastic, then higher priority, then the name of the backend's type. Never
the order of registration, so the result does not depend on which package was loaded first.

Type-unstable by design, and opt-in: never a default and never used inside library code.
`ExpectationFirstOf(ExpectationDynamicRouter())` freezes the current choice into a type-stable
router.
"""
struct ExpectationDynamicRouter{R}
    registry::R
end

ExpectationDynamicRouter() = ExpectationDynamicRouter(EXPECTATION_BACKEND_REGISTRY)

backendname(backend) = string(nameof(typeof(backend)))

function which_expectation_backend(router::ExpectationDynamicRouter, ::Type{F}, ::Type{Q}) where {F, Q}
    candidates = [
        (b, p) for (b, p) in zip(router.registry.backends, router.registry.priorities) if
            supports_expectation(b, F, Q)
    ]
    isempty(candidates) && return nothing
    rank((b, p)) =
        (!has_expectation_label(b, F, Q, ExpectationExact()), has_expectation_label(b, F, Q, ExpectationStochastic()), -p, backendname(b))
    return first(argmin(rank, candidates))
end

function supports_expectation(router::ExpectationDynamicRouter, ::Type{F}, ::Type{Q}) where {F, Q}
    return !isnothing(which_expectation_backend(router, F, Q))
end

function has_expectation_label(
        router::ExpectationDynamicRouter, ::Type{F}, ::Type{Q}, label::ExpectationLabel
    ) where {F, Q}
    backend = which_expectation_backend(router, F, Q)
    return !isnothing(backend) && has_expectation_label(backend, F, Q, label)
end

function expectation(router::ExpectationDynamicRouter, f::F, q) where {F}
    backend = which_expectation_backend(router, typeof(f), typeof(q))
    isnothing(backend) && throw(UnsupportedExpectationError(router, f, q))
    return expectation(backend, f, q)
end

prepare_expectation(router::ExpectationDynamicRouter, q) = prepare_expectation(ExpectationFirstOf(router), q)

"""
    ExpectationFirstOf(router::ExpectationDynamicRouter; prefer = ExpectationExact(), exclude = ())

Freeze the backends registered now into a type-stable [`ExpectationFirstOf`](@ref): ordered by not
stochastic first, then priority, then name, and preferring exact backends pair by pair.
"""
function ExpectationFirstOf(router::ExpectationDynamicRouter; prefer = ExpectationExact(), exclude = ())
    registry = router.registry
    order = sortperm(
        collect(zip(registry.backends, registry.priorities));
        by = ((b, p),) -> (is_expectation_stochastic(b), -p, backendname(b)),
    )
    return ExpectationFirstOf(registry.backends[order]...; prefer, exclude)
end

# Products: each component goes to its own backend
function supports_expectation(
        router::ExpectationDynamicRouter, ::Type, ::Type{FactorizedJoint{T}}
    ) where {T <: Tuple}
    return supports_all(router, T)
end
function has_expectation_label(
        router::ExpectationDynamicRouter, ::Type, ::Type{FactorizedJoint{T}}, label::ExpectationLabel
    ) where {T <: Tuple}
    return haslabel_all(router, T, label)
end
function which_expectation_backend(
        router::ExpectationDynamicRouter, ::Type, ::Type{FactorizedJoint{T}}
    ) where {T <: Tuple}
    return supports_all(router, T) ? router : nothing
end
prepare_expectation(router::ExpectationDynamicRouter, q::FactorizedJoint) = prepare_product(router, q)
function expectation(router::ExpectationDynamicRouter, f::F, q::FactorizedJoint) where {F}
    return expectation(prepare_expectation(router, q), f)
end
