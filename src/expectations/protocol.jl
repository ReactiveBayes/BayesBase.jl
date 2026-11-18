# The expectation protocol: one way to ask for `E_q[f]`, answered by backends defined in other
# packages (closed forms, quadrature, sampling). A backend says what it supports and which
# labels hold (`ExpectationExact`, `ExpectationStochastic`); routers (`ExpectationFirstOf`,
# `ExpectationDynamicRouter`) choose among backends. Nothing here is exported: the names are public
# and used qualified, such as `BayesBase.expectation`, so they never clash with names exported by
# other packages.

"""
    expectation(backend, f, q)

The expectation `E_q[f]` of the integrand `f` under the distribution `q`, computed by `backend`.

A backend with a closed form adds a method for the specific types of `f` and `q` it covers.
A numerical backend adds a method of [`prepare_expectation`](@ref) for the types of `q` it
supports instead, and the fallback method computes
`expectation(prepare_expectation(backend, q), f)`.
When the backend has neither, the fallback throws an [`UnsupportedExpectationError`](@ref).

    expectation(prepared, f)

The expectation of `f` from an object returned by [`prepare_expectation`](@ref).
"""
function expectation end

"""
    prepare_expectation(backend, q)

Do the work for expectations under `q` that does not depend on the integrand, such as
computing nodes and weights, and return an object `prepared`. Then `expectation(prepared, f)`
computes the expectation of any `f`, including integrands whose values are dual numbers.

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
`Q`. Backends add methods to the type form, which is `false` by default. `F = Any` asks whether
every integrand is supported. A numerical method supports every integrand and declares it with
`supports_expectation(::MyBackend, ::Type, ::Type{<:MyFamily}) = true`.
"""
supports_expectation(backend, ::Type, ::Type) = false
supports_expectation(backend, f, q) = supports_expectation(backend, typeof(f), typeof(q))

"""
    which_expectation_backend(backend, F::Type, Q::Type)
    which_expectation_backend(backend, f, q)

The backend that computes the expectation, or `nothing` if there is none. A router returns the
backend it chooses, or itself for a product of distributions. Any other backend returns itself
if it supports the pair.
"""
function which_expectation_backend(backend, ::Type{F}, ::Type{Q}) where {F, Q}
    return supports_expectation(backend, F, Q) ? backend : nothing
end
which_expectation_backend(backend, f, q) = which_expectation_backend(backend, typeof(f), typeof(q))
