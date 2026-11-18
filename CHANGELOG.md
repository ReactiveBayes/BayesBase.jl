# Changelog

All notable changes to BayesBase.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

To be released as 1.7.0.

### Added

- A protocol for computing expectations `E_q[f]`, with backends defined in other packages:
  `BayesBase.expectation` and `BayesBase.prepare_expectation`. It also has functions for what a
  backend supports (`supports_expectation`, `which_expectation_backend`), labels for properties
  of a computation (`ExpectationExact`, `ExpectationStochastic`, `has_expectation_label`,
  `is_expectation_exact`, `is_expectation_stochastic`, `combine_expectation_label`), random number
  generators (`with_expectation_rng`, `get_expectation_rng`), point sets and
  products over `FactorizedJoint` (`ExpectationPointSet`, `ExpectationProductPointSet`), and
  routers (`ExpectationFirstOf`, `ExpectationDynamicRouter`, `ExpectationBackendRegistry`,
  `register_expectation_backend!`). The names are public and not exported. A closed form for the
  expectation of a log-density dispatches on the integrand `Base.Fix1(logpdf, p)`. See the new
  "Expectations" page of the documentation.
- `expectation_backend_priority(backend; registry)` returns the priority a backend was registered
  with, or `nothing` if it is not registered. A negative priority marks a last resort.
- Compat.jl is a new dependency, for declaring public names on Julia 1.10.

### Fixed

- `promote_variate_type` for a pair no package supports throws an `ArgumentError` naming the
  pair, instead of a `MethodError` from inside BayesBase.
- `ndims(::PointMass{<:UniformScaling})` and `mean(mirrorlog, ::SampleList)` for a multivariate
  or matrix-variate sample list throw errors that say what is not defined.
- `prod(GenericProd(), ::ProductOf, ::ProductOf)` is no longer ambiguous. The left product is
  multiplied by each side of the right product in turn, with a closed form where one exists.
- The product of a density and a `SampleList` resamples when the effective number of samples
  falls below a tenth of the number of samples. It now resamples with the new weights instead of
  the list's old weights. Before, it returned the list unweighted.

### Changed

- BayesBase requires Julia 1.10 or later, the oldest version its CI tests.
- The code is formatted with [Runic](https://github.com/fredrikekre/Runic.jl) instead of
  JuliaFormatter: `make format` and `make check-format`, and the `Format` workflow in CI.
- The tooling follows the ReactiveBayes PackageTemplate: the Makefile, the CI, CompatHelper,
  TagBot and documentation preview workflows, and the documentation setup.

[Unreleased]: https://github.com/ReactiveBayes/BayesBase.jl/compare/v1.6.1...HEAD
