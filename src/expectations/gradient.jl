# Gradients of expectations with respect to the parameters of the distribution. By the score
# identity, ∇_θ E_q[f] = E_q[f ∇_θ log q], a gradient is the expectation of the integrand
# `ExpectationScoreProduct(f, q)`, so the routers, the registry, the labels, preparations and products
# of the protocol serve gradients without code of their own. A closed form is a method of
# `expectation` for that integrand's type, and a numerical backend sums the integrand at its
# points wherever `expectation_score` has a method for the distribution.

"""
    expectation_score(q, x)

The score of `q` at `x`: the gradient of `log q(x; θ)` with respect to the parameters `θ` of
`q`. The parametrization is the one the type of `q` defines: `params(q)` for a distribution of
Distributions.jl, and the natural parameters for an `ExponentialFamilyDistribution` of
ExponentialFamily.jl. BayesBase has methods for `Normal`, `LogNormal`, `Gamma`, `Exponential`,
`Beta` and a [`FactorizedJoint`](@ref) of those, whose score concatenates the scores of its
components at the tuple `x`. Packages add methods for the types they define.

The default throws an `ArgumentError`. A backend with a closed form for the gradient does not
need the score.

```jldoctest
julia> using Distributions

julia> BayesBase.expectation_score(Normal(1.0, 2.0), 4.0)  # with respect to μ and σ
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 0.75
 0.625
```
"""
function expectation_score(q, x)
    return throw(
        ArgumentError(
            "BayesBase.expectation_score has no method for $(typeof(q)), so the gradient of an expectation under it needs a backend with a closed form for that gradient",
        ),
    )
end

function expectation_score(q::Normal, x)
    μ, σ = params(q)
    z = (x - μ) / σ
    return SA[z / σ, (z^2 - 1) / σ]
end

function expectation_score(q::LogNormal, x)
    μ, σ = params(q)
    z = (log(x) - μ) / σ
    return SA[z / σ, (z^2 - 1) / σ]
end

function expectation_score(q::Gamma, x)
    α, θ = params(q)
    return SA[log(x) - log(θ) - digamma(α), (x / θ - α) / θ]
end

function expectation_score(q::Exponential, x)
    θ = scale(q)
    return SA[(x / θ - 1) / θ]
end

function expectation_score(q::Beta, x)
    α, β = params(q)
    ψ = digamma(α + β)
    return SA[log(x) - digamma(α) + ψ, log1p(-x) - digamma(β) + ψ]
end

function expectation_score(q::FactorizedJoint, x::Tuple)
    return reduce(vcat, map(expectation_score, components(q), x))
end

"""
    has_expectation_score(Q::Type)
    has_expectation_score(q)

Whether [`BayesBase.expectation_score`](@ref) has a method for distributions of type `Q`. It is
`false` by default, `true` for the families with a score in BayesBase, and for a
[`FactorizedJoint`](@ref) whose components all have one. A package that adds a method of
`expectation_score` also adds a method of this function, which makes numerical backends
compute gradients under its type, see [`BayesBase.supports_expectation_gradient`](@ref).

```jldoctest
julia> using Distributions

julia> BayesBase.has_expectation_score(Normal(0.0, 1.0))
true

julia> BayesBase.has_expectation_score(Categorical([0.5, 0.5]))
false
```
"""
has_expectation_score(::Type) = false
has_expectation_score(q) = has_expectation_score(typeof(q))
has_expectation_score(::Type{<:Union{Normal, LogNormal, Gamma, Exponential, Beta}}) = true
has_expectation_score(::Type{FactorizedJoint{T}}) where {T <: Tuple} = all_have_scores(T)

all_have_scores(::Type{Tuple{}}) = true
function all_have_scores(::Type{T}) where {T <: Tuple}
    return has_expectation_score(Base.tuple_type_head(T)) && all_have_scores(Base.tuple_type_tail(T))
end

"""
    ExpectationScoreProduct(f, q)

The integrand `x -> f(x) * expectation_score(q, x)`, whose expectation under `q` is the gradient of
the expectation of `f` with respect to the parameters of `q`, see
[`BayesBase.expectation_gradient`](@ref). A backend with a closed form for the gradient adds a
method of `expectation` for this type, for example with
[`BayesBase.@supports_expectation_gradient`](@ref). A numerical backend evaluates it at its points
like any other integrand, which needs [`BayesBase.expectation_score`](@ref) for `q`.

With a preparation for `q`, the gradient is `expectation(prepared, ExpectationScoreProduct(f, q))`.
"""
struct ExpectationScoreProduct{F, Q}
    f::F
    q::Q
end

(product::ExpectationScoreProduct)(x) = product.f(x) * expectation_score(product.q, x)

"""
    expectation_gradient(backend, f, q)

The gradient of the expectation of `f` under `q` with respect to the parameters of `q`,
``\\nabla_\\theta \\mathbb{E}_{q_\\theta}[f]``, computed by `backend`. The parametrization is that
of [`BayesBase.expectation_score`](@ref): `params(q)` for a distribution of Distributions.jl, and
the natural parameters for an `ExponentialFamilyDistribution`.

The gradient equals ``\\mathbb{E}_q[f \\, \\nabla_\\theta \\log q]``, and it is computed as
`expectation(backend, ExpectationScoreProduct(f, q), q)`. Routers, labels and products therefore
work for gradients as they do for expectations: a router chooses a backend with a closed form for
the gradient when there is one, and otherwise a numerical backend that sums
`f(x) * expectation_score(q, x)` over its points. Backends with closed forms define them with
[`BayesBase.@supports_expectation_gradient`](@ref).

The identity holds when `f` does not depend on `θ` and the support of `q` does not depend on
`θ`, so it does not apply to the bounds of a `Uniform`, for example. The integrand `f` must return
a scalar, which multiplies the score.
"""
expectation_gradient(backend, f::F, q) where {F} = expectation(backend, ExpectationScoreProduct(f, q), q)

"""
    supports_expectation_gradient(backend, F::Type, Q::Type)
    supports_expectation_gradient(backend, f, q)

Whether `backend` computes the gradient of the expectation of integrands of type `F` under
distributions of type `Q`, see [`BayesBase.expectation_gradient`](@ref). Backends declare support
with methods of [`BayesBase.supports_expectation`](@ref) for the integrand
`ExpectationScoreProduct{F, Q}`, not with methods of this function.

A backend that supports every integrand under `Q`, such as a quadrature rule, computes gradients
by summing `f` times the score, so it supports gradients only where
[`BayesBase.has_expectation_score`](@ref) holds for `Q`. Routers apply the same rule, so they pass
over such a backend for a family without a score and choose a backend with a closed form for the
gradient instead.
"""
function supports_expectation_gradient(backend, ::Type{F}, ::Type{Q}) where {F, Q}
    return admits(backend, ExpectationScoreProduct{F, Q}, Q)
end

# A backend that takes every integrand under `Q` computes a gradient with the score of `Q`; one
# that supports the gradient's integrand but not every integrand has a closed form for it. A router
# has already applied this rule to its backends, except for a product, which it computes itself.
function computable(backend, ::Type{<:ExpectationScoreProduct}, ::Type{Q}) where {Q}
    return has_expectation_score(Q) || !supports_expectation(backend, Any, Q)
end
computable(::Union{ExpectationFirstOf, ExpectationDynamicRouter}, ::Type{<:ExpectationScoreProduct}, ::Type) = true
function computable(
        ::Union{ExpectationFirstOf, ExpectationDynamicRouter}, ::Type{<:ExpectationScoreProduct}, ::Type{Q}
    ) where {Q <: FactorizedJoint}
    return has_expectation_score(Q)
end
supports_expectation_gradient(backend, f, q) = supports_expectation_gradient(backend, typeof(f), typeof(q))

function Base.showerror(io::IO, error::UnsupportedExpectationError{B, <:ExpectationScoreProduct}) where {B}
    print(
        io,
        "UnsupportedExpectationError: `",
        error.backend,
        "` cannot compute the gradient of the expectation of `",
        error.f.f,
    )
    return print(io, "` under `", error.q, "`")
end
