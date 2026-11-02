"""
    BackendRegistry()

Backends registered for automatic choice by a [`DynamicRouter`](@ref), with priorities.
"""
struct BackendRegistry
    backends::Vector{Any}
    priorities::Vector{Int}
end

BackendRegistry() = BackendRegistry(Any[], Int[])

Base.length(registry::BackendRegistry) = length(registry.backends)
Base.isempty(registry::BackendRegistry) = isempty(registry.backends)

const GLOBAL_REGISTRY = BackendRegistry()

"""
    register_backend!(backend; priority = 0, registry = the global registry)

Make `backend` available to [`DynamicRouter`](@ref)s. Registering a backend again changes its
priority. Packages call this in `__init__`.
"""
function register_backend!(
        backend; priority::Integer = 0, registry::BackendRegistry = GLOBAL_REGISTRY
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
    DynamicRouter(registry = the global registry)

A router that chooses among registered backends at run time. Ranking: exact for the pair
first, then not stochastic, then higher priority, then the name of the backend's type. Never
the order of registration, so the result does not depend on which package was loaded first.

Type-unstable by design, and opt-in: never a default and never used inside library code.
`FirstOf(DynamicRouter())` freezes the current choice into a type-stable router.
"""
struct DynamicRouter{R}
    registry::R
end

DynamicRouter() = DynamicRouter(GLOBAL_REGISTRY)

backendname(backend) = string(nameof(typeof(backend)))

function which_backend(router::DynamicRouter, ::Type{F}, ::Type{Q}) where {F, Q}
    candidates = [
        (b, p) for (b, p) in zip(router.registry.backends, router.registry.priorities) if
            supports(b, F, Q)
    ]
    isempty(candidates) && return nothing
    rank((b, p)) =
        (!haslabel(b, F, Q, Exact()), haslabel(b, F, Q, Stochastic()), -p, backendname(b))
    return first(argmin(rank, candidates))
end

function supports(router::DynamicRouter, ::Type{F}, ::Type{Q}) where {F, Q}
    return !isnothing(which_backend(router, F, Q))
end

function haslabel(
        router::DynamicRouter, ::Type{F}, ::Type{Q}, label::ExpectationLabel
    ) where {F, Q}
    backend = which_backend(router, F, Q)
    return !isnothing(backend) && haslabel(backend, F, Q, label)
end

function expectation(router::DynamicRouter, f, q)
    backend = which_backend(router, typeof(f), typeof(q))
    isnothing(backend) && throw(NoBackendError(router, f, q))
    return expectation(backend, f, q)
end

prepare_expectation(router::DynamicRouter, q) = prepare_expectation(FirstOf(router), q)

"""
    FirstOf(router::DynamicRouter; prefer = Exact(), exclude = ())

Freeze the backends registered now into a type-stable [`FirstOf`](@ref): ordered by not
stochastic first, then priority, then name, and preferring exact backends pair by pair.
"""
function FirstOf(router::DynamicRouter; prefer = Exact(), exclude = ())
    registry = router.registry
    order = sortperm(
        collect(zip(registry.backends, registry.priorities));
        by = ((b, p),) -> (isstochastic(b), -p, backendname(b)),
    )
    return FirstOf(registry.backends[order]...; prefer, exclude)
end

# Products: each component goes to its own backend
function supports(
        router::DynamicRouter, ::Type, ::Type{FactorizedJoint{T}}
    ) where {T <: Tuple}
    return supports_all(router, T)
end
function haslabel(
        router::DynamicRouter, ::Type, ::Type{FactorizedJoint{T}}, label::ExpectationLabel
    ) where {T <: Tuple}
    return haslabel_all(router, T, label)
end
function which_backend(
        router::DynamicRouter, ::Type, ::Type{FactorizedJoint{T}}
    ) where {T <: Tuple}
    return supports_all(router, T) ? router : nothing
end
prepare_expectation(router::DynamicRouter, q::FactorizedJoint) = prepare_product(router, q)
function expectation(router::DynamicRouter, f, q::FactorizedJoint)
    return expectation(prepare_expectation(router, q), f)
end
