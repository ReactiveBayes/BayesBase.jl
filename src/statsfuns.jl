import SpecialFunctions: digamma, logbeta, loggamma, trigamma

export mirrorlog,
    xtlog,
    logmvbeta,
    clamplog,
    mvtrigamma,
    dtanh,
    vague,
    probvec,
    weightedmean,
    mean_cov,
    mean_var,
    mean_std,
    mean_precision,
    mean_invcov,
    weightedmean_cov,
    weightedmean_var,
    weightedmean_std,
    weightedmean_invcov,
    weightedmean_precision,
    compute_logscale,
    logpdf_sampling_optimized,
    logpdf_optimized,
    sampling_optimized,
    UnspecifiedDomain,
    UnspecifiedDimension,
    fuse_supports,
    isequal_typeof,
    distribution_typewrapper,
    CountingReal

"""
    mirrorlog(x)

Returns `log(1 - x)`.
"""
mirrorlog(x) = log(one(x) - x)

"""
    xtlog(x)

Returns `x * log(x)`.
"""
xtlog(x) = x * log(x)

"""
    clamplog(x)

Same as `log` but clamps the input argument `x` to be in the range `tiny <= x <= typemax(x)` such that `log(0)` does not explode.
"""
clamplog(x) = log(clamp(x, tiny, typemax(x)))

"""
    logmvbeta(x)

Uses the numerically stable algorithm to compute the logarithm of the multivariate beta distribution over with the parameter vector x.
"""
logmvbeta(x) = sum(loggamma, x) - loggamma(sum(x))

"""
    mvtrigamma(p, x)

Computes multivariate trigamma function .
"""
mvtrigamma(p, x) = sum(trigamma(x + (one(x) - i) / 2) for i in 1:p)

"""
    dtanh(x)

Alias for `1 - tanh(x) ^ 2`
"""
dtanh(x) = one(x) - abs2(tanh(x))

"""
    vague(distribution_type, [ dims... ])

Returns uninformative probability distribution of the given type.
"""
function vague end

function compute_logscale end

"""
    probvec(d)

Returns the probability vector of the given distribution.
"""
function probvec end

"""
    weightedmean(d)

Returns the weighted mean of the given distribution. 
Alias to `invcov(d) * mean(d)`, but can be specialized
"""
weightedmean(d) = invcov(d) * mean(d)

"""
Alias for `(mean(d), cov(d))`, but can be specialized.
"""
mean_cov(something) = (mean(something), cov(something))

"""
Alias for `(mean(d), var(d))`, but can be specialized.
"""
mean_var(something) = (mean(something), var(something))

"""
Alias for `(mean(d), std(d))`, but can be specialized.
"""
mean_std(something) = (mean(something), std(something))

"""
Alias for `(mean(d), invcov(d))`, but can be specialized.
"""
mean_invcov(something) = (mean(something), invcov(something))

"""
Alias for `mean_invcov(d)`, but can be specialized.
"""
mean_precision(something) = mean_invcov(something)

"""
Alias for `(weightedmean(d), cov(d))`, but can be specialized.
"""
weightedmean_cov(something) = (weightedmean(something), cov(something))

"""
Alias for `(weightedmean(d), var(d))`, but can be specialized.
"""
weightedmean_var(something) = (weightedmean(something), var(something))

"""
Alias for `(weightedmean(d), std(d))`, but can be specialized.
"""
weightedmean_std(something) = (weightedmean(something), std(something))

"""
Alias for `(weightedmean(d), invcov(d))`, but can be specialized.
"""

"""
Alias for `weightedmean_invcov(d)`, but can be specialized.
"""
weightedmean_invcov(something) = (weightedmean(something), invcov(something))
weightedmean_precision(something) = weightedmean_invcov(something)

"""
    logpdf_sampling_optimized(d) 
    
`logpdf_sample_optimized` function takes as an input a distribution `d` and returns corresponding optimized two versions 
for taking `logpdf()` and sampling with `rand!` respectively. Alias for `(logpdf_optimized(d), sampling_optimized(d))`, but can be specialized.
"""
function logpdf_sampling_optimized(something)
    return (logpdf_optimized(something), sampling_optimized(something))
end

"""
    logpdf_optimized(d)

Returns a version of `d` specifically optimized to call `logpdf(d, x)`. By default returns the same `d`, but can be specialized.
"""
logpdf_optimized(something) = something

"""
    sampling_optimized(d)

Returns a version of `d` specifically optimized to call `rand` and `rand!`. By default returns the same `d`, but can be specialized.
"""
sampling_optimized(something) = something

"""Unknown domain that is used as a placeholder when exact domain knowledge is unavailable"""
struct UnspecifiedDomain <: Domain{Any} end

"""Unknown dimension is equal and not equal to any number"""
struct UnspecifiedDimension end

DomainSets.dimension(::UnspecifiedDomain) = UnspecifiedDimension()

Base.in(::Any, ::UnspecifiedDomain) = true

Base.:(!=)(::UnspecifiedDimension, ::Int) = true
Base.:(!==)(::UnspecifiedDimension, ::Int) = true
Base.:(==)(::UnspecifiedDimension, ::Int) = true

"""
    fuse_supports(left, right)

Fuses supports `left` and `right`.
By default, checks that the inputs are identical and throws an error otherwise.
Can implement specific fusions for specific supports.
"""
function fuse_supports(left, right)
    @assert isequal(left, right) "Cannot automatically fuse supports of $(left) & `$(right)`."
    return left
end

fuse_supports(left::UnspecifiedDomain, right) = right
fuse_supports(left, right::UnspecifiedDomain) = left
fuse_supports(::UnspecifiedDomain, ::UnspecifiedDomain) = UnspecifiedDomain()

"""
Strips type parameters from the type of the `distribution`.
"""
distribution_typewrapper(distribution) = generated_distribution_typewrapper(distribution)

"""
    isequal_typeof(left, right)

Alias for `typeof(left) === typeof(right)`, but can be specialized.
"""
isequal_typeof(left, right) = typeof(left) === typeof(right)

# Returns a wrapper distribution for a `<:Distribution` type, this function uses internals of Julia 
# It is not ideal, but is fine for now, if Julia changes it internals such that does not work 
# We will need to write the `distribution_typewrapper` method for each support member of exponential family
# e.g. `distribution_typewrapper(::Bernoulli) = Bernoulli`
@generated function generated_distribution_typewrapper(distribution)
    return Base.typename(distribution).wrapper
end

"""
    CountingReal

`CountingReal` implements a real "number" that counts 'infinities' in a separate field.
See also [`BayesBase.Infinity`](@ref) and [`BayesBase.MinusInfinity`](@ref).

# Arguments
- `value::T`: value of type `<: Real`
- `infinities::Int`: number of added/subtracted infinities

```jldoctest
julia> r = BayesBase.CountingReal(0.0, 0)
CountingReal{Float64}(0.0, 0)

julia> float(r)
0.0

julia> r = r + BayesBase.Infinity(Float64)
CountingReal{Float64}(0.0, 1)

julia> float(r)
Inf

julia> r = r + BayesBase.MinusInfinity(Float64)
CountingReal{Float64}(0.0, 0)

julia> float(r)
0.0
```
"""
struct CountingReal{T<:Real}
    value::T
    infinities::Int
end

CountingReal(value::T) where {T<:Real} = CountingReal{T}(value, 0)

function CountingReal(::Type{T}, infinities::Int) where {T<:Real}
    return CountingReal{T}(zero(T), infinities)
end

value(a::CountingReal) = a.value
infinities(a::CountingReal) = a.infinities

"""An object representing infinity."""
Infinity(::Type{T}) where {T} = CountingReal(zero(T), 1)

"""An object representing minus infinity."""
MinusInfinity(::Type{T}) where {T} = CountingReal(zero(T), -1)

Base.isfinite(a::CountingReal) = !isinf(a)
Base.isinf(a::CountingReal) = !(iszero(infinities(a))) || isinf(value(a))
Base.isnan(a::CountingReal) = isnan(value(a))

Base.eltype(::Type{CountingReal{T}}) where {T} = T
Base.eltype(::Type{CountingReal}) = Real
Base.eltype(::T) where {T<:CountingReal} = eltype(T)

Base.:+(a::CountingReal) = CountingReal(+value(a), +infinities(a))
Base.:-(a::CountingReal) = CountingReal(-value(a), -infinities(a))

Base.:+(a::CountingReal, b::Real) = CountingReal(value(a) + b, infinities(a))
Base.:-(a::CountingReal, b::Real) = CountingReal(value(a) - b, infinities(a))
Base.:+(b::Real, a::CountingReal) = CountingReal(b + value(a), +infinities(a))
Base.:-(b::Real, a::CountingReal) = CountingReal(b - value(a), -infinities(a))

function Base.:*(::CountingReal, ::Real)
    return error("`CountingReal` multiplication with `Real` is dissalowed")
end
function Base.:*(::Real, ::CountingReal)
    return error("`CountingReal` multiplication with `Real` is dissalowed")
end

Base.:*(a::CountingReal, b::Integer) = CountingReal(value(a) * b, infinities(a) * b)
Base.:*(a::Integer, b::CountingReal) = CountingReal(a * value(b), a * infinities(b))

Base.:/(::CountingReal, ::Real) = error("`CountingReal` division is dissalowed")
Base.:/(::Real, ::CountingReal) = error("`CountingReal` division is dissalowed")

function Base.:+(a::CountingReal, b::CountingReal)
    return CountingReal(value(a) + value(b), infinities(a) + infinities(b))
end
function Base.:-(a::CountingReal, b::CountingReal)
    return CountingReal(value(a) - value(b), infinities(a) - infinities(b))
end

Base.convert(::Type{CountingReal}, v::T) where {T<:Real} = CountingReal(v)
Base.convert(::Type{CountingReal{T}}, v::T) where {T<:Real} = CountingReal(v)
function Base.convert(::Type{CountingReal{T}}, v::R) where {T<:Real,R<:Real}
    return CountingReal(convert(T, v))
end
function Base.convert(::Type{CountingReal{T}}, v::CountingReal{R}) where {T<:Real,R<:Real}
    return CountingReal{T}(convert(T, value(v)), infinities(v))
end

Base.float(a::CountingReal{T}) where {T} = isfinite(a) ? value(a) : convert(T, Inf)
Base.zero(::Type{CountingReal{T}}) where {T<:Real} = CountingReal(zero(T))

function Base.promote_rule(::Type{CountingReal{T1}}, ::Type{T2}) where {T1<:Real,T2<:Real}
    return CountingReal{promote_type(T1, T2)}
end
Base.promote_rule(::Type{CountingReal}, ::Type{T}) where {T<:Real} = CountingReal{T}

function Base.:(==)(left::CountingReal{T}, right::CountingReal{T}) where {T}
    return (value(left) == value(right)) && (infinities(left) == infinities(right))
end

"""
    InplaceLogpdf(logpdf!)

Wraps a `logpdf!` function in a type that can later on be used for dispatch. 
The sole purpose of this wrapper type is to allow for in-place logpdf operation on a batch of samples.
Accepts a function `logpdf!` that takes two arguments: `out` and `sample` and writes the logpdf of the sample to the `out` array.
A regular `logpdf` function can be converted to `logpdf!` by using `convert(InplaceLogpdf, logpdf)`.

```jldoctest
julia> using Distributions, BayesBase

julia> d = Beta(2, 3);

julia> inplace = convert(BayesBase.InplaceLogpdf, (sample) -> logpdf(d, sample));

julia> out = zeros(9);

julia> inplace(out, 0.1:0.1:0.9)
9-element Vector{Float64}:
 -0.028399474521697776
  0.42918163472548043
  0.5675839575845996
  0.5469646703818638
  0.4054651081081646
  0.14149956227369964
 -0.2797139028026039
 -0.9571127263944104
 -2.2256240518579173
```

```jldoctest
julia> using Distributions, BayesBase

julia> d = Beta(2, 3);

julia> inplace = BayesBase.InplaceLogpdf((out, sample) -> logpdf!(out, d, sample));

julia> out = zeros(9);

julia> inplace(out, 0.1:0.1:0.9)
9-element Vector{Float64}:
 -0.028399474521697776
  0.42918163472548043
  0.5675839575845996
  0.5469646703818638
  0.4054651081081646
  0.14149956227369964
 -0.2797139028026039
 -0.9571127263944104
 -2.2256240518579173
```
"""
struct InplaceLogpdf{F}
    logpdf!::F
end

function (inplace::InplaceLogpdf)(out, x)
    inplace.logpdf!(out, x)
    return out
end

function Base.convert(::Type{InplaceLogpdf}, something)
    return InplaceLogpdf((out, x) -> map!(something, out, x))
end

function Base.convert(::Type{InplaceLogpdf}, inplace::InplaceLogpdf)
    return inplace
end