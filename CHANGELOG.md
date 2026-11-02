# Changelog

All notable changes to BayesBase.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.7.0] - 2026-11-02

### Added

- The expectation protocol: one way to ask for `E_q[f]`, answered by backends defined in other
  packages. `BayesBase.expectation` and `BayesBase.prepare_expectation`; what a backend
  supports (`supports_expectation`, `which_expectation_backend`) and which labels hold
  (`ExpectationExact`, `ExpectationStochastic`, `has_expectation_label`, `is_expectation_exact`,
  `is_expectation_stochastic`, `combine_expectation_label`); random number generators from above
  (`with_expectation_rng`, `get_expectation_rng`); point sets and products over `FactorizedJoint`
  (`ExpectationPointSet`, `ExpectationProductPointSet`); and routers (`ExpectationFirstOf`,
  `ExpectationDynamicRouter`, `ExpectationBackendRegistry`, `register_expectation_backend!`). Every
  name says it is about expectations; they are public and not exported. The expectation of a
  log-density dispatches on Julia's `Base.Fix1(logpdf, p)`. See the new "Expectations" page of the
  documentation.
- Compat.jl is a new dependency, for declaring public names on Julia 1.10.

### Fixed

- `promote_variate_type` for a pair no package supports throws an `ArgumentError` naming the
  pair, instead of a `MethodError` from inside BayesBase.
- `ndims(::PointMass{<:UniformScaling})` and `mean(mirrorlog, ::SampleList)` for a multivariate
  or matrix-variate sample list throw errors that say what is not defined.

### Changed

- BayesBase requires Julia 1.10 or later, the oldest version its CI tests.
- The code is formatted with [Runic](https://github.com/fredrikekre/Runic.jl) instead of
  JuliaFormatter: `make format` and `make check-format`, and the `Format` workflow in CI.
- The tooling follows the ReactiveBayes PackageTemplate: the Makefile, the CI, CompatHelper,
  TagBot and documentation preview workflows, and the documentation setup.

[Unreleased]: https://github.com/ReactiveBayes/BayesBase.jl/compare/v1.7.0...HEAD
[1.7.0]: https://github.com/ReactiveBayes/BayesBase.jl/compare/v1.6.0...v1.7.0
