# The expectation protocol: one way to ask for `E_q[f]`, answered by backends defined in other
# packages (closed forms, quadrature, sampling). A backend says what it supports and which
# labels hold (`ExpectationExact`, `ExpectationStochastic`); routers (`ExpectationFirstOf`,
# `ExpectationDynamicRouter`) choose among backends. Nothing here is exported: the names are public
# and used qualified, such as `BayesBase.expectation`, so they never clash with names exported by
# other packages.

"""
    expectation(backend, f, q)

The expectation `E_q[f]` of the integrand `f` under the distribution `q`, computed by `backend`.

A backend implements either a closed form, a method of `expectation` for specific types of
`f` and `q`, or a preparation, a method of [`prepare_expectation`](@ref) for specific types of
`q` that returns an object with its own `expectation(prepared, f)`. The fallback goes through
the preparation.

    expectation(prepared, f)

The expectation of `f` from an object returned by [`prepare_expectation`](@ref).
"""
function expectation end

"""
    prepare_expectation(backend, q)

Prepare the expectations of many integrands under `q`: nodes, weights and other work that does
not depend on the integrand are done once. The result is used as `expectation(prepared, f)`,
for any `f`, including integrands whose values are dual numbers.

The default returns a [`PreparedExpectation`](@ref), which forwards every call to the backend.
"""
function prepare_expectation end

"""
    PreparedExpectation(backend, q)

The default result of [`prepare_expectation`](@ref): no work is done in advance, and
`expectation(prepared, f)` calls `expectation(backend, f, q)`.
"""
struct PreparedExpectation{B, Q}
    backend::B
    q::Q
end

prepare_expectation(backend, q) = PreparedExpectation(backend, q)

expectation(prepared::PreparedExpectation, f::F) where {F} = expectation(prepared.backend, f, prepared.q)

expectation(backend, f::F, q) where {F} = expectation_from(prepare_expectation(backend, q), f)

# A backend with neither a closed form for `(f, q)` nor a preparation for `q` would otherwise
# send `expectation` and the default preparation back and forth forever
expectation_from(prepared, f::F) where {F} = expectation(prepared, f)
function expectation_from(prepared::PreparedExpectation, f::F) where {F}
    return throw(UnsupportedExpectationError(prepared.backend, f, prepared.q))
end

"""
    UnsupportedExpectationError(backend, f, q)

Thrown when `backend` cannot compute the expectation of `f` under `q`.
"""
struct UnsupportedExpectationError{B, F, Q} <: Exception
    backend::B
    f::F
    q::Q
end

function Base.showerror(io::IO, error::UnsupportedExpectationError)
    print(
        io,
        "UnsupportedExpectationError: `",
        error.backend,
        "` cannot compute the expectation of `",
        error.f,
    )
    return print(io, "` under `", error.q, "`")
end

"""
    supports_expectation(backend, F::Type, Q::Type)
    supports_expectation(backend, f, q)

Whether `backend` computes expectations of integrands of type `F` under distributions of type
`Q`. Backend owners add methods for the type-level form; it is `false` by default. `F = Any`
asks whether every integrand is supported, which is what a numerical method declares with
`supports_expectation(::MyBackend, ::Type, ::Type{<:MyFamily}) = true`.
"""
supports_expectation(backend, ::Type, ::Type) = false
supports_expectation(backend, f, q) = supports_expectation(backend, typeof(f), typeof(q))

"""
    which_expectation_backend(backend, F::Type, Q::Type)
    which_expectation_backend(backend, f, q)

The backend that computes the expectation, or `nothing`. A router returns its choice, any
other backend itself when it supports the pair.
"""
function which_expectation_backend(backend, ::Type{F}, ::Type{Q}) where {F, Q}
    return supports_expectation(backend, F, Q) ? backend : nothing
end
which_expectation_backend(backend, f, q) = which_expectation_backend(backend, typeof(f), typeof(q))
