```@meta
CurrentModule = BayesBase
```

# BayesBase.jl

`BayesBase` is a package that serves as an umbrella, defining, exporting, and re-exporting methods essential for Bayesian statistics specifically for the [`RxInfer` ecosystem](https://github.com/biaslab/RxInfer.jl). 

Related projects:

- [`RxInfer`](https://github.com/biaslab/RxInfer.jl)
- [`ExponentialFamily`](https://github.com/biaslab/ExponentialFamily.jl)

# Index

```@index
```

# [Library API](@id library)

## [Generic densities](@id library-densities)

```@docs
BayesBase.PointMass
BayesBase.ContinuousUnivariateLogPdf
BayesBase.ContinuousMultivariateLogPdf
BayesBase.SampleList
BayesBase.FactorizedJoint
BayesBase.MixtureDistribution
BayesBase.Contingency
```

## [Product API](@id library-prod)

The `prod` function defines an interface to compute a product between two probability distributions over the same variable.
It accepts a strategy as its first argument, which defines how the prod function should behave and what results you can expect.

```@docs
prod(::UnspecifiedProd, left, right)
BayesBase.default_prod_rule
```

### [Product strategies](@id library-prod-strategies)

For certain distributions, it's possible to compute the product using a straightforward mathematical equation, yielding a closed-form solution. 
However, for some distributions, finding a closed-form solution might not be feasible. 
Various strategies ensure consistent behavior in these situations. 
These strategies can either guarantee a fast and closed-form solution or, when necessary, fall back to a slower but more generic method.

```@docs
BayesBase.UnspecifiedProd
BayesBase.ClosedProd
BayesBase.PreserveTypeProd
BayesBase.PreserveTypeLeftProd
BayesBase.PreserveTypeRightProd
BayesBase.GenericProd
BayesBase.ProductOf
BayesBase.LinearizedProductOf
BayesBase.TerminalProdArgument
BayesBase.resolve_prod_strategy
```

These strategies offer flexibility and reliability when working with different types of distributions, ensuring that the package can handle a wide range of cases effectively.

## [Promotion type utilities](@id library-promotion-utilities)

```@docs
BayesBase.deep_eltype
BayesBase.isequal_typeof
BayesBase.paramfloattype
BayesBase.sampletype
BayesBase.samplefloattype
BayesBase.promote_variate_type
BayesBase.promote_paramfloattype
BayesBase.promote_sampletype
BayesBase.promote_samplefloattype
BayesBase.convert_paramfloattype
```

## [Extra stats functions](@id library-statsfuns)

```@docs
BayesBase.mirrorlog
BayesBase.xtlog
BayesBase.logmvbeta
BayesBase.clamplog
BayesBase.mvtrigamma
BayesBase.dtanh
BayesBase.probvec
BayesBase.mean_std
BayesBase.mean_var
BayesBase.mean_cov
BayesBase.mean_invcov
BayesBase.mean_precision
BayesBase.weightedmean
BayesBase.weightedmean_std
BayesBase.weightedmean_var
BayesBase.weightedmean_cov
BayesBase.weightedmean_invcov
```

## [Helper utilities](@id library-helpers)

```@docs
BayesBase.vague
BayesBase.logpdf_sampling_optimized
BayesBase.logpdf_optimized
BayesBase.sampling_optimized
BayesBase.fuse_supports
BayesBase.UnspecifiedDomain
BayesBase.UnspecifiedDimension
BayesBase.distribution_typewrapper
BayesBase.CountingReal
BayesBase.Infinity
BayesBase.MinusInfinity
```
