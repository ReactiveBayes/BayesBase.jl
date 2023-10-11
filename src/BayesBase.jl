module BayesBase

using TinyHugeNumbers

using StatsAPI, StatsBase, DomainSets, Statistics, Distributions, Random

using StatsAPI: params

export params

using Statistics: mean, median, std, var, cov

export mean, median, std, var, cov

using StatsBase: mode, entropy, weights

export mode, entropy, weights

using Distributions:
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
    components

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
    components

using DomainSets: dimension, Domain

export dimension, Domain

using Base: precision, prod, prod!

export precision, prod, prod!

using Random: rand, rand!

export rand, rand!

include("statsfuns.jl")
include("promotion.jl")
include("prod.jl")

include("densities/factorizedjoint.jl")
include("densities/mixture.jl")
include("densities/function.jl")

end
