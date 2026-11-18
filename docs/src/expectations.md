# [Expectations](@id expectations)

Inference algorithms often need the expectation ``\mathbb{E}_q[f]`` of a function `f` under a
distribution `q`. BayesBase defines a protocol for computing it,
[`BayesBase.expectation`](@ref)`(backend, f, q)`, where the `backend` argument selects the
method: a closed form, a quadrature rule or a sampler, for example. Backends are defined in
other packages, such as
[ExpectationApproximations.jl](https://github.com/ReactiveBayes/ExpectationApproximations.jl).
An algorithm written against the protocol works with every backend that supports its
distributions, and the user of the algorithm chooses the method.

The design is similar to
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl), which
has a `backend` argument that selects the differentiation package and `prepare_*` functions
whose result is reused by later calls. Here, [`BayesBase.prepare_expectation`](@ref)`(backend, q)`
does the work that depends only on `q`, such as computing quadrature nodes and weights, and its
result is reused for many integrands.

The names of the protocol are public but not exported. Call them qualified, as
`BayesBase.expectation`.

```julia
using BayesBase, Distributions

backend = SomeBackend()                               # from a backend package
BayesBase.expectation(backend, x -> x^2, Normal(0.0, 1.0))

prepared = BayesBase.prepare_expectation(backend, Normal(0.0, 1.0))
BayesBase.expectation(prepared, x -> x^2)             # many integrands, nodes computed once
```

## Computing expectations

`expectation(backend, f, q)` returns the expectation, or throws a
[`BayesBase.UnsupportedExpectationError`](@ref) when `backend` cannot compute it. For several
integrands under the same `q`, call `prepare_expectation(backend, q)` once and then
`expectation(prepared, f)` for each integrand.

A backend implements one of two things for the distributions it supports. A closed form is a
method of `expectation` for specific types of `f` and `q`, such as
`expectation(::MyBackend, ::typeof(log), q::Gamma)`. A preparation is a method of
`prepare_expectation` for specific types of `q`. It returns an object with its own method
`expectation(prepared, f)`, which works for any integrand, including integrands whose values are
dual numbers. Most numerical backends return a [`BayesBase.ExpectationPointSet`](@ref), a
weighted sum over points.

Backends define these methods for specific families of distributions, not for all `q`.
Products of distributions and routers rely on this.

```@docs
BayesBase.expectation
BayesBase.prepare_expectation
BayesBase.PreparedExpectation
BayesBase.UnsupportedExpectationError
```

## What a backend supports

A backend declares the pairs it supports with methods of
[`BayesBase.supports_expectation`](@ref). These methods take the types of `f` and `q`, so the
compiler knows the answer. The integrand type `Any` means every integrand, which is what a
numerical method declares:

```julia
BayesBase.supports_expectation(::MyQuadrature, ::Type, ::Type{<:Normal}) = true        # any f
BayesBase.supports_expectation(::MyClosedForms, ::Type{typeof(log)}, ::Type{<:Gamma}) = true
```

For the expectation of a log-density, `E_q[log p(x)]`, the integrand is
`Base.Fix1(logpdf, p)`, which is Julia's callable for `x -> logpdf(p, x)`. A backend with a
closed form declares and implements it by dispatch on that type:

```julia
BayesBase.supports_expectation(::MyClosedForms, ::Type{<:Base.Fix1{typeof(logpdf), <:Normal}}, ::Type{<:Normal}) = true
BayesBase.expectation(::MyClosedForms, f::Base.Fix1{typeof(logpdf), <:Normal}, q::Normal) = …
```

```@docs
BayesBase.supports_expectation
BayesBase.which_expectation_backend
```

## Labels

A label is a property of a computation that a caller can rely on, such as
[`BayesBase.ExpectationExact`](@ref) for a result that is exact up to rounding. A backend
declares a label for all the pairs it supports or for some of them, and only where the property
is guaranteed. Below, every closed form is exact, and the quadrature rule is exact for
categorical distributions:

```julia
BayesBase.has_expectation_label(::MyClosedForms, ::BayesBase.ExpectationExact) = true
BayesBase.has_expectation_label(::MyQuadrature, ::Type, ::Type{<:Categorical}, ::BayesBase.ExpectationExact) = true
```

```@docs
BayesBase.ExpectationLabel
BayesBase.ExpectationExact
BayesBase.ExpectationStochastic
BayesBase.has_expectation_label
BayesBase.is_expectation_exact
BayesBase.is_expectation_stochastic
BayesBase.combine_expectation_label
```

## Random number generators

A stochastic backend stores its own random number generator, and `expectation` does not take
one. The code that sets up the computation, such as an inference engine, gives the backend a
generator with [`BayesBase.with_expectation_rng`](@ref). A generator that the user has already
set is never replaced.

```@docs
BayesBase.with_expectation_rng
BayesBase.get_expectation_rng
```

## Products of distributions

An expectation over several independent variables, which can be of different kinds (for
example a categorical, a multivariate Gaussian and a point mass), is an expectation under a
[`FactorizedJoint`](@ref). The integrand receives a tuple with one value per component. A
backend supports a product when it supports every component for any integrand. The expectation
is computed with a product rule over the point sets of the components. Labels are combined with
[`BayesBase.combine_expectation_label`](@ref): the product is exact only if every component is
exact, and stochastic if any component is stochastic.

```@docs
BayesBase.ExpectationPointSet
BayesBase.ExpectationProductPointSet
```

## Routers

A router is a backend that passes each expectation on to one of several other backends.
[`BayesBase.ExpectationFirstOf`](@ref) chooses from the types of `f` and `q` alone, so the
choice is made at compile time. [`BayesBase.ExpectationDynamicRouter`](@ref) chooses at run
time among the backends that packages have registered, and is meant for interactive use.
`ExpectationFirstOf(ExpectationDynamicRouter())` turns the current choice of the dynamic router
into an `ExpectationFirstOf`.

```@docs
BayesBase.ExpectationFirstOf
BayesBase.ExpectationDynamicRouter
BayesBase.ExpectationBackendRegistry
BayesBase.register_expectation_backend!
BayesBase.expectation_backend_priority
```
