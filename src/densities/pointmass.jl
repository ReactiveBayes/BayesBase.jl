using LinearAlgebra: UniformScaling, I
using SpecialFunctions: loggamma, logbeta

export PointMass

"""
    PointMass(point)

A `PointMass` structure represents a delta distribution, a discrete probability distribution 
where all probability mass is concentrated at a single point. This point is specified by 
the provided `point`.
"""
struct PointMass{P}
    point::P
end

getpointmass(distribution::PointMass) = distribution.point
getpointmass(point::Union{Real,AbstractArray}) = point

BayesBase.variate_form(::Type{PointMass{T}}) where {T<:Real} = Univariate
BayesBase.variate_form(::Type{PointMass{V}}) where {T,V<:AbstractVector{T}} = Multivariate
BayesBase.variate_form(::Type{PointMass{M}}) where {T,M<:AbstractMatrix{T}} = Matrixvariate
function BayesBase.variate_form(::Type{PointMass{M}}) where {T,N,M<:AbstractArray{T,N}}
    return ArrayLikeVariate{N}
end
BayesBase.variate_form(::Type{PointMass{U}}) where {T,U<:UniformScaling{T}} = Matrixvariate

function BayesBase.mean(fn::F, distribution::PointMass) where {F<:Function}
    return fn(mean(distribution))
end

##

BayesBase.sampletype(::PointMass{T}) where {T} = T
BayesBase.paramfloattype(distribution::PointMass) = deep_eltype(getpointmass(distribution))
function BayesBase.convert_paramfloattype(::Type{T}, distribution::PointMass) where {T}
    return PointMass(convert_paramfloattype(T, getpointmass(distribution)))
end

##

function Base.getindex(distribution::PointMass, index...)
    return Base.getindex(getpointmass(distribution), index...)
end
function Base.size(distribution::PointMass, index...)
    return Base.size(getpointmass(distribution), index...)
end

# `entropy` for the `PointMass` is not defined
BayesBase.entropy(dist::PointMass{T}) where {T} = MinusInfinity(paramfloattype(dist))

# Real-based univariate point mass

function BayesBase.insupport(distribution::PointMass{T}, x::Real) where {T<:Real}
    return x == getpointmass(distribution)
end
function BayesBase.pdf(distribution::PointMass{T}, x::Real) where {T<:Real}
    return insupport(distribution, x) ? one(T) : zero(T)
end
function BayesBase.logpdf(distribution::PointMass{T}, x::Real) where {T<:Real}
    return insupport(distribution, x) ? zero(T) : convert(T, -Inf)
end

BayesBase.mean(distribution::PointMass{T}) where {T<:Real} = getpointmass(distribution)
BayesBase.mode(distribution::PointMass{T}) where {T<:Real} = mean(distribution)
BayesBase.var(distribution::PointMass{T}) where {T<:Real} = zero(T)
BayesBase.std(distribution::PointMass{T}) where {T<:Real} = zero(T)
BayesBase.cov(distribution::PointMass{T}) where {T<:Real} = zero(T)

function BayesBase.probvec(::PointMass{T}) where {T<:Real}
    return error("probvec(::PointMass{ <: Real }) is not defined")
end

Base.precision(::PointMass{T}) where {T<:Real} = convert(T, Inf)
Base.ndims(::PointMass{T}) where {T<:Real} = 1
Base.eltype(::PointMass{T}) where {T<:Real} = T

# AbstractArray-based multivariate point mass

function BayesBase.insupport(
    distribution::PointMass{V}, x::AbstractArray{T,N}
) where {T<:Real,N,V<:AbstractArray{T,N}}
    return x == getpointmass(distribution)
end

function BayesBase.pdf(
    distribution::PointMass{V}, x::AbstractArray{T,N}
) where {T<:Real,N,V<:AbstractArray{T,N}}
    return insupport(distribution, x) ? one(T) : zero(T)
end

function BayesBase.logpdf(
    distribution::PointMass{V}, x::AbstractArray{T,N}
) where {T<:Real,N,V<:AbstractArray{T,N}}
    return insupport(distribution, x) ? zero(T) : convert(T, -Inf)
end

function BayesBase.mean(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T}}
    return getpointmass(distribution)
end
function BayesBase.mode(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T}}
    return mean(distribution)
end
function BayesBase.var(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T}}
    return zeros(T, ndims(distribution))
end

function BayesBase.std(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T}}
    return zeros(T, ndims(distribution))
end

# For vectors, covariances and probvec are defined
function BayesBase.cov(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T,1}}
    return zeros(T, (ndims(distribution), ndims(distribution)))
end

function BayesBase.probvec(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T,1}}
    return mean(distribution)
end

function BayesBase.cov(distribution::PointMass{M}) where {T<:Real,N,M<:AbstractArray{T,N}}
    return error("cov(::PointMass{ <: AbstractMatrix }) is not defined")
end

function BayesBase.probvec(
    distribution::PointMass{M}
) where {T<:Real,N,M<:AbstractArray{T,N}}
    return error("probvec(::PointMass{ <: AbstractMatrix }) is not defined")
end

function Base.precision(distribution::PointMass{V}) where {T<:Real,V<:AbstractArray{T}}
    return one(T) ./ cov(distribution)
end

# We need this function for backwards compatibility

function Base.ndims(distribution::PointMass{V}) where {T<:Real,V<:AbstractVector{T}}
    return length(mean(distribution))
end

function Base.ndims(distribution::PointMass{M}) where {T<:Real,N,M<:AbstractArray{T,N}}
    return size(mean(distribution))
end

Base.eltype(::PointMass{V}) where {T<:Real,N,V<:AbstractArray{T,N}} = T

# UniformScaling-based matrixvariate point mass

function BayesBase.insupport(
    distribution::PointMass{M}, x::UniformScaling
) where {T<:Real,M<:UniformScaling{T}}
    return x == getpointmass(distribution)
end
function BayesBase.pdf(
    distribution::PointMass{M}, x::UniformScaling
) where {T<:Real,M<:UniformScaling{T}}
    return insupport(distribution, x) ? one(T) : zero(T)
end
function BayesBase.logpdf(
    distribution::PointMass{M}, x::UniformScaling
) where {T<:Real,M<:UniformScaling{T}}
    return insupport(distribution, x) ? zero(T) : convert(T, -Inf)
end

function BayesBase.mean(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return getpointmass(distribution)
end
function BayesBase.mode(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return mean(distribution)
end
function BayesBase.var(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return zero(T) * I
end
function BayesBase.std(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return zero(T) * I
end
function BayesBase.cov(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return error("cov(::PointMass{ <: UniformScaling }) is not defined")
end

function BayesBase.probvec(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return error("probvec(::PointMass{ <: UniformScaling }) is not defined")
end

function Base.precision(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return one(T) ./ cov(distribution)
end
function Base.ndims(distribution::PointMass{M}) where {T<:Real,M<:UniformScaling{T}}
    return size(mean(distribution))
end
Base.eltype(::PointMass{M}) where {T<:Real,M<:UniformScaling{T}} = T

function Base.isapprox(left::PointMass, right::PointMass; kwargs...)
    return Base.isapprox(getpointmass(left), getpointmass(right); kwargs...)
end

Base.isapprox(left::PointMass, right; kwargs...) = false
Base.isapprox(left, right::PointMass; kwargs...) = false

function BayesBase.rand!(::AbstractRNG, dist::PointMass, container::AbstractVector)
    point = mean(dist)
    for i in eachindex(container)
        container[i] = point
    end
    return container
end

function BayesBase.rand(::AbstractRNG, dist::PointMass)
    return mean(dist)
end

function Random.rand(rng::AbstractRNG, dist::PointMass{P}, size::Int64) where {P}
    container = Vector{P}(undef, size)
    return rand!(rng, dist, container)
end

function Random.rand(dist::PointMass, size::Int64)
    return rand(Random.default_rng(), dist, size)
end
