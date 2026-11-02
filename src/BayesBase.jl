module BayesBase

using StatsAPI, StatsBase, DomainSets, Statistics, Distributions, TinyHugeNumbers, Random
using Compat: @compat

import StatsAPI: params

export params

import Statistics: mean, median, std, var, cov

export mean, median, std, var, cov

import StatsBase: mode, entropy, weights

export mode, entropy, weights

import Distributions:
    failprob,
    succprob,
    insupport,
    support,
    shape,
    scale,
    location,
    rate,
    dof,
    invcov,
    pdf,
    pdf!,
    cdf,
    logpdf,
    logpdf!,
    logdetcov,
    VariateForm,
    ValueSupport,
    Distribution,
    Univariate,
    Multivariate,
    Matrixvariate,
    variate_form,
    value_support,
    component,
    components,
    kurtosis,
    skewness

export failprob,
    succprob,
    insupport,
    support,
    shape,
    scale,
    location,
    rate,
    dof,
    invcov,
    pdf,
    pdf!,
    cdf,
    logpdf,
    logpdf!,
    logdetcov,
    VariateForm,
    ValueSupport,
    Distribution,
    Univariate,
    Multivariate,
    Matrixvariate,
    variate_form,
    value_support,
    component,
    components,
    kurtosis,
    skewness

import DomainSets: dimension, Domain

export dimension, Domain

import Base: precision, prod, prod!

export precision, prod, prod!

import Random: rand, rand!

export rand, rand!

include("statsfuns.jl")
include("promotion.jl")
include("prod.jl")

include("densities/pointmass.jl")
include("densities/function.jl")
include("densities/samplelist.jl")
include("densities/mixture.jl")
include("densities/factorizedjoint.jl")
include("densities/contingency.jl")
include("algebra/arrowheadmatrix.jl")
include("algebra/matrixwithcahcedinverse.jl")

# The expectation protocol: public, not exported
@compat public expectation, prepare_expectation, PreparedExpectation, UnsupportedExpectationError
@compat public supports_expectation, which_expectation_backend, with_expectation_rng, get_expectation_rng
@compat public ExpectationLabel, ExpectationExact, ExpectationStochastic, combine_expectation_label
@compat public has_expectation_label, is_expectation_exact, is_expectation_stochastic
@compat public ExpectationPointSet, ExpectationProductPointSet, ExpectationFirstOf
@compat public ExpectationDynamicRouter, ExpectationBackendRegistry, register_expectation_backend!

include("expectations/protocol.jl")
include("expectations/labels.jl")
include("expectations/rng.jl")
include("expectations/pointset.jl")
include("expectations/products.jl")
include("expectations/firstof.jl")
include("expectations/dynamic.jl")

end
