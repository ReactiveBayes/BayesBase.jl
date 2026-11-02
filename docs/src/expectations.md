# [Expectations](@id expectations)

BayesBase defines a protocol for one question that inference algorithms ask over and over:
what is the expectation ``\mathbb{E}_q[f]`` of a function `f` under a distribution `q`?
Backends answer it, and they are defined in other packages: exact closed forms, quadrature,
cubature, sampling. Algorithms only ever call the protocol, so a new backend improves every
algorithm that uses it.

None of these names is exported. They are public and used qualified, as
`BayesBase.expectation`, so they never clash with names exported by other packages.

```julia
using BayesBase, Distributions

backend = SomeBackend()                               # from a backend package
BayesBase.expectation(backend, x -> x^2, Normal(0.0, 1.0))

prepared = BayesBase.prepare_expectation(backend, Normal(0.0, 1.0))
BayesBase.expectation(prepared, x -> x^2)             # many integrands, nodes computed once
```

## Computing expectations

A backend implements one of two things for the distributions it supports.

- **A closed form**: a method of [`BayesBase.expectation`](@ref) for specific types of `f`
  and `q`, such as `expectation(::MyBackend, ::typeof(log), q::Gamma)`.
- **A preparation**: a method of [`BayesBase.prepare_expectation`](@ref) for specific types
  of `q`. It returns an object with its own `expectation(prepared, f)`, which works for any
  integrand, including integrands whose values are dual numbers. Numerical backends return a
  [`BayesBase.PointSet`](@ref), a weighted sum over points.

Backends define methods for specific families of distributions only, never for all `q`.
Products of distributions and the routers below rely on that.

```@docs
BayesBase.expectation
BayesBase.prepare_expectation
BayesBase.Prepared
BayesBase.NoBackendError
BayesBase.Logpdf
```

## What a backend supports

[`BayesBase.supports`](@ref) answers on types, so that the answer is known to the compiler.
Backend owners add methods for the type-level form. The integrand type `Any` means "every
integrand", which is what a numerical method declares:

```julia
BayesBase.supports(::MyQuadrature, ::Type, ::Type{<:Normal}) = true        # any f
BayesBase.supports(::MyClosedForms, ::Type{typeof(log)}, ::Type{<:Gamma}) = true
```

```@docs
BayesBase.supports
BayesBase.which_backend
```

## Labels

A label is a property of a computation that a caller can rely on. A backend declares a label
for every pair it supports, or for some pairs only. Labels are conservative: a label is
declared only where it is guaranteed.

```julia
BayesBase.haslabel(::MyClosedForms, ::BayesBase.Exact) = true
BayesBase.haslabel(::MyQuadrature, ::Type, ::Type{<:Categorical}, ::BayesBase.Exact) = true
```

```@docs
BayesBase.ExpectationLabel
BayesBase.Exact
BayesBase.Stochastic
BayesBase.haslabel
BayesBase.isexact
BayesBase.isstochastic
BayesBase.combine_label
```

## Random number generators

A stochastic backend holds its random number generator. The caller never passes one to
`expectation`. Instead, whoever configures the computation, such as an inference engine,
fills it in from above with [`BayesBase.withrng`](@ref), and an RNG set by the user is never
overwritten.

```@docs
BayesBase.withrng
BayesBase.getrng
```

## Products of distributions

The expectation over several independent variables of different kinds, such as a categorical,
a multivariate Gaussian and a point mass, is an expectation under a
[`FactorizedJoint`](@ref). The integrand receives a tuple with one value per component.
A backend supports a product when it supports every component for any integrand. The product
is computed as a product rule over the point sets of the components, and labels are combined
with [`BayesBase.combine_label`](@ref): exact only if every part is exact, stochastic if any
part is stochastic.

```@docs
BayesBase.PointSet
BayesBase.ProductPointSet
```

## Routers

A router is itself a backend that passes each expectation on to one of several backends.
[`BayesBase.FirstOf`](@ref) decides on types alone, so the compiler makes the choice and
nothing is decided at run time. [`BayesBase.DynamicRouter`](@ref) chooses at run time among
the backends that packages have registered, and is meant for interactive use.
`FirstOf(DynamicRouter())` freezes its current choice into a `FirstOf`.

```@docs
BayesBase.FirstOf
BayesBase.DynamicRouter
BayesBase.BackendRegistry
BayesBase.register_backend!
```
