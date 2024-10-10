export ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf
export DiscreteUnivariateLogPdf
# import DomainIntegrals
# import HCubature

import Base: isapprox, in

abstract type AbstractContinuousGenericLogPdf end
abstract type AbstractDiscreteGenericLogPdf end

getdomain(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}) = dist.domain
getlogpdf(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}) = dist.logpdf

BayesBase.value_support(::Type{<:AbstractContinuousGenericLogPdf}) = Continuous
BayesBase.value_support(::Type{<:AbstractDiscreteGenericLogPdf}) = Discrete
# We throw an error on purpose, since we do not want to use `AbstractContinuousGenericLogPdf` much without approximations
# We want to encourage a user to use approximate generic log-pdfs as much as possible instead
function __error_genericlogpdf_not_defined(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}, f::Symbol)
    return error(
        "`$f` is not defined for `$(dist)`. Use functional form constraints to approximate the resulting generic log-pdf object and to use it in the inference procedure.",
    )
end

function BayesBase.mean(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :mean)
end
function BayesBase.median(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :median)
end
function BayesBase.mode(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :mode)
end
function BayesBase.var(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :var)
end
function BayesBase.std(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :std)
end
function BayesBase.cov(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :cov)
end
function BayesBase.invcov(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :invcov)
end
function BayesBase.entropy(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :entropy)
end

function Base.precision(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})
    return __error_genericlogpdf_not_defined(dist, :precision)
end

Base.eltype(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}) = eltype(getdomain(dist))

BayesBase.paramfloattype(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}) = deep_eltype(eltype(dist))
BayesBase.samplefloattype(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}) = paramfloattype(dist)

(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})(x::Real) = logpdf(dist, x)
(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf})(x::AbstractVector{<:Real}) = logpdf(dist, x)

function BayesBase.logpdf(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}, x)
    @assert x ∈ getdomain(dist) "x = $(x) does not belong to the domain ($(getdomain(dist))) of $dist"
    lpdf = getlogpdf(dist)
    return lpdf(x)
end

# We don't expect neither `pdf` nor `logpdf` to be normalised
BayesBase.pdf(dist::Union{AbstractContinuousGenericLogPdf,AbstractDiscreteGenericLogPdf}, x) = exp(logpdf(dist, x))

"""
    DiscreteUnivariateLogPdf{ D <: DomainSets.Domain, F } <: AbstractDiscreteGenericLogPdf

Generic discrete univariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: domain specificatiom from `DomainSets.jl` package, by default the `domain` is set to `DomainSets.Integers()`. Use `BayesBase.UnspecifiedDomain()` to bypass domain checks.
- `logpdf`: callable object that represents the logdensity. Can be un-normalised.
"""
struct DiscreteUnivariateLogPdf{D<:DomainSets.Domain,F} <: AbstractDiscreteGenericLogPdf
    domain::D
    logpdf::F

    function DiscreteUnivariateLogPdf(domain::D, logpdf::F) where {D,F}
        @assert dimension(domain) == 1 "Cannot create DiscreteUnivariateLogPdf. Dimension of domain = $(domain) is not equal to 1."
        return new{D,F}(domain, logpdf)
    end
end

function DiscreteUnivariateLogPdf(f::Function)
    return DiscreteUnivariateLogPdf(DomainSets.Integers(), f)
end

BayesBase.variate_form(::Type{<:DiscreteUnivariateLogPdf}) = Univariate

function BayesBase.promote_variate_type(
    ::Type{Univariate}, ::Type{AbstractDiscreteGenericLogPdf}
)
    return DiscreteUnivariateLogPdf
end

function Base.show(io::IO, dist::DiscreteUnivariateLogPdf)
    return print(io, "DiscreteUnivariateLogPdf(", getdomain(dist), ")")
end
function Base.show(io::IO, ::Type{<:DiscreteUnivariateLogPdf{D}}) where {D}
    return print(io, "DiscreteUnivariateLogPdf{", D, "}")
end

function BayesBase.support(dist::DiscreteUnivariateLogPdf)
    return getdomain(dist)
end

BayesBase.insupport(dist::DiscreteUnivariateLogPdf, x) = x ∈ getdomain(dist)

# Fallback for various optimisation packages which may pass arguments as vectors
function BayesBase.logpdf(dist::DiscreteUnivariateLogPdf, x::AbstractVector{<:Int})
    @assert length(x) === 1 "`DiscreteUnivariateLogPdf` expects either float or a vector of a single float as an input for the `logpdf` function."
    return logpdf(dist, first(x))
end

function Base.convert(
    ::Type{<:DiscreteUnivariateLogPdf}, domain::D, logpdf::F
) where {D<:DomainSets.Domain,F}
    return DiscreteUnivariateLogPdf(domain, logpdf)
end

function BayesBase.convert_paramfloattype(
    ::Type{T}, dist::DiscreteUnivariateLogPdf
) where {T<:Int}
    return convert(
        DiscreteUnivariateLogPdf,
        dist.domain,
        (x) -> dist.logpdf(convert_paramfloattype(T, x)),
    )
end

function BayesBase.vague(::Type{<:DiscreteUnivariateLogPdf})
    return DiscreteUnivariateLogPdf(DomainSets.Integers(), (x) -> 0)
end

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
function BayesBase.isequal_typeof(
    ::DiscreteUnivariateLogPdf{D,F1}, ::DiscreteUnivariateLogPdf{D,F2}
) where {D,F1<:Function,F2<:Function}
    return true
end
## 

"""
    ContinuousUnivariateLogPdf{ D <: DomainSets.Domain, F } <: AbstractContinuousGenericLogPdf

Generic continuous univariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: domain specificatiom from `DomainSets.jl` package, by default the `domain` is set to `DomainSets.FullSpace()`. Use `BayesBase.UnspecifiedDomain()` to bypass domain checks.
- `logpdf`: callable object that represents the logdensity. Can be un-normalised.
"""
struct ContinuousUnivariateLogPdf{D<:DomainSets.Domain,F} <: AbstractContinuousGenericLogPdf
    domain::D
    logpdf::F

    function ContinuousUnivariateLogPdf(domain::D, logpdf::F) where {D,F}
        @assert dimension(domain) == 1 "Cannot create ContinuousUnivariateLogPdf. Dimension of domain = $(domain) is not equal to 1."
        return new{D,F}(domain, logpdf)
    end
end

function ContinuousUnivariateLogPdf(f::Function)
    return ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)
end

BayesBase.variate_form(::Type{<:ContinuousUnivariateLogPdf}) = Univariate

function BayesBase.promote_variate_type(
    ::Type{Univariate}, ::Type{AbstractContinuousGenericLogPdf}
)
    return ContinuousUnivariateLogPdf
end

function Base.show(io::IO, dist::ContinuousUnivariateLogPdf)
    return print(io, "ContinuousUnivariateLogPdf(", getdomain(dist), ")")
end
function Base.show(io::IO, ::Type{<:ContinuousUnivariateLogPdf{D}}) where {D}
    return print(io, "ContinuousUnivariateLogPdf{", D, "}")
end

function BayesBase.support(dist::ContinuousUnivariateLogPdf)
    return getdomain(dist)
end

BayesBase.insupport(dist::ContinuousUnivariateLogPdf, x) = x ∈ getdomain(dist)

# Fallback for various optimisation packages which may pass arguments as vectors
function BayesBase.logpdf(dist::ContinuousUnivariateLogPdf, x::AbstractVector{<:Real})
    @assert length(x) === 1 "`ContinuousUnivariateLogPdf` expects either float or a vector of a single float as an input for the `logpdf` function."
    return logpdf(dist, first(x))
end

function Base.convert(
    ::Type{<:ContinuousUnivariateLogPdf}, domain::D, logpdf::F
) where {D<:DomainSets.Domain,F}
    return ContinuousUnivariateLogPdf(domain, logpdf)
end

function BayesBase.convert_paramfloattype(
    ::Type{T}, dist::ContinuousUnivariateLogPdf
) where {T<:Real}
    return convert(
        ContinuousUnivariateLogPdf,
        dist.domain,
        (x) -> dist.logpdf(convert_paramfloattype(T, x)),
    )
end

function BayesBase.vague(::Type{<:ContinuousUnivariateLogPdf})
    return ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 0)
end

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
function BayesBase.isequal_typeof(
    ::ContinuousUnivariateLogPdf{D,F1}, ::ContinuousUnivariateLogPdf{D,F2}
) where {D,F1<:Function,F2<:Function}
    return true
end

## 

"""
    ContinuousMultivariateLogPdf{ D <: DomainSets.Domain, F } <: AbstractContinuousGenericLogPdf

Generic continuous multivariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: multidimensional domain specification from `DomainSets.jl` package. Use `BayesBase.UnspecifiedDomain()` to bypass domain checks.
- `logpdf`: callable object that accepts an `AbstractVector` as an input and represents the logdensity. Can be un-normalised.
"""
struct ContinuousMultivariateLogPdf{D<:DomainSets.Domain,F} <:
       AbstractContinuousGenericLogPdf
    domain::D
    logpdf::F

    function ContinuousMultivariateLogPdf(
        domain::D, logpdf::F
    ) where {D<:DomainSets.Domain,F}
        @assert DomainSets.dimension(domain) !== 1 "Cannot create ContinuousMultivariateLogPdf. Dimension of domain = $(domain) should not be equal to 1. Use, for example, `DomainSets.FullSpace() ^ 2` to create 2-dimensional full space domain."
        return new{D,F}(domain, logpdf)
    end
end

BayesBase.variate_form(::Type{<:ContinuousMultivariateLogPdf}) = Multivariate

function BayesBase.promote_variate_type(
    ::Type{Multivariate}, ::Type{AbstractContinuousGenericLogPdf}
)
    return ContinuousMultivariateLogPdf
end

function ContinuousMultivariateLogPdf(dims::Int, f::Function)
    return ContinuousMultivariateLogPdf(DomainSets.FullSpace()^dims, f)
end

function Base.show(io::IO, dist::ContinuousMultivariateLogPdf)
    return print(io, "ContinuousMultivariateLogPdf(", getdomain(dist), ")")
end
function Base.show(io::IO, ::Type{<:ContinuousMultivariateLogPdf{D}}) where {D}
    return print(io, "ContinuousMultivariateLogPdf{", D, "}")
end

BayesBase.support(dist::ContinuousMultivariateLogPdf) = getdomain(dist)
BayesBase.insupport(dist::ContinuousMultivariateLogPdf, x) = x ∈ getdomain(dist)

function Base.convert(
    ::Type{<:ContinuousMultivariateLogPdf}, domain::D, logpdf::F
) where {D<:DomainSets.Domain,F}
    return ContinuousMultivariateLogPdf(domain, logpdf)
end

function BayesBase.convert_paramfloattype(
    ::Type{T}, dist::ContinuousMultivariateLogPdf
) where {T<:Real}
    return convert(
        ContinuousMultivariateLogPdf,
        dist.domain,
        (x) -> dist.logpdf(convert_paramfloattype(T, x)),
    )
end

function BayesBase.vague(::Type{<:ContinuousMultivariateLogPdf}, dims::Int)
    return ContinuousMultivariateLogPdf(DomainSets.FullSpace()^dims, (x) -> 0)
end

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
function BayesBase.isequal_typeof(
    ::ContinuousMultivariateLogPdf{D,F1}, ::ContinuousMultivariateLogPdf{D,F2}
) where {D,F1<:Function,F2<:Function}
    return true
end
