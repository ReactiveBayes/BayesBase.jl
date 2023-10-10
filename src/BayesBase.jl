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
    shape,
    scale,
    rate,
    invcov,
    pdf,
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
    shape,
    scale,
    rate,
    invcov,
    pdf,
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

using Base: precision, eltype, convert, length, isapprox

export precision, eltype, convert, length, isapprox

using Random: rand, rand!

export rand, rand!

include("statsfuns.jl")
include("promotion.jl")
include("prod.jl")

include("densities/factorizedjoint.jl")

end
