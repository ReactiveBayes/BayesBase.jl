module BayesBase

using TinyHugeNumbers

using StatsAPI, StatsBase, Statistics, Distributions, Random

using StatsAPI: params

export params

using Statistics: mean, median, std, var, cov

export mean, median, std, var, cov

using StatsBase: mode, entropy

export mode, entropy

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
    value_support

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
    value_support

using Base: precision, prod, prod!

export precision, prod, prod!

using Random: rand, rand!

export rand, rand!

include("statsfuns.jl")
include("promotion.jl")
include("prod.jl")

include("densities/factorizedjoint.jl")

end
