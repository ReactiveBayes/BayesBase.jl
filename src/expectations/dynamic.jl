"""
    ExpectationBackendRegistry()

The backends, each with a priority, among which an [`ExpectationDynamicRouter`](@ref) chooses.
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
its priority. Packages call this in `__init__`. A dynamic router cannot be registered, because it
would choose among the backends that include itself.
"""
function register_expectation_backend!(
        backend; priority::Integer = 0, registry::ExpectationBackendRegistry = EXPECTATION_BACKEND_REGISTRY
    )
    backend isa ExpectationDynamicRouter &&
        throw(ArgumentError("a dynamic router cannot be registered as a backend of a registry"))
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
    expectation_backend_priority(backend; registry = the global registry) -> Union{Int, Nothing}

The priority `backend` was registered with, or `nothing` if it is not registered. A negative
priority marks a last resort: a backend meant for the cases where no other applies. An
inference report can use it to say where such a backend was used. An [`ExpectationFirstOf`](@ref)
frozen from a dynamic router keeps the order that the priorities gave but not the priorities
themselves, so this function looks them up in the registry.
"""
function expectation_backend_priority(backend; registry::ExpectationBackendRegistry = EXPECTATION_BACKEND_REGISTRY)
    i = findfirst(isequal(backend), registry.backends)
    return isnothing(i) ? nothing : registry.priorities[i]
end

"""
    ExpectationDynamicRouter(registry = the global registry)

A router that chooses among the registered backends at run time. Among the backends that
support the pair, it prefers one that is exact for the pair, then one that is not stochastic,
then one with a higher priority, and finally orders by the name of the backend's type. Backends
with different names therefore do not depend on the order in which packages were loaded. Only
backends that tie on all of these, such as two backends of the same type, keep the order in which
they were registered.

The router is type-unstable. `ExpectationFirstOf(ExpectationDynamicRouter())` freezes the current
choice into a type-stable router, which is what an inference engine should use, as RxInfer's
fallback schemes do by default. `with_expectation_rng` freezes it too, so that the generator
reaches the backends.
"""
struct ExpectationDynamicRouter{R}
    registry::R
end

ExpectationDynamicRouter() = ExpectationDynamicRouter(EXPECTATION_BACKEND_REGISTRY)

backendname(backend) = string(nameof(typeof(backend)))

function which_expectation_backend(router::ExpectationDynamicRouter, ::Type{F}, ::Type{Q}) where {F, Q}
    candidates = [
        (b, p) for (b, p) in zip(router.registry.backends, router.registry.priorities) if
            admits(b, F, Q)
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

# The registered backends are not known until run time, so the generator goes to a frozen router
with_expectation_rng(router::ExpectationDynamicRouter, rng) = with_expectation_rng(ExpectationFirstOf(router), rng)

"""
    ExpectationFirstOf(router::ExpectationDynamicRouter; prefer = ExpectationExact(), exclude = ())

Freeze the backends registered now into a type-stable [`ExpectationFirstOf`](@ref). The backends
are ordered with the non-stochastic ones first, then by decreasing priority, then by name. With
the default `prefer`, an exact backend is chosen for a pair when there is one.
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
prepare_expectation(router::ExpectationDynamicRouter, q::FactorizedJoint{<:Tuple}) = prepare_product(router, q)
function expectation(router::ExpectationDynamicRouter, f::F, q::FactorizedJoint{T}) where {F, T <: Tuple}
    supports_all(router, T) || throw(UnsupportedExpectationError(router, f, q))
    return expectation(prepare_expectation(router, q), f)
end
