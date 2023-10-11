export deep_eltype,
    promote_variate_type,
    paramfloattype,
    promote_paramfloattype,
    convert_paramfloattype,
    sampletype,
    samplefloattype,
    promote_sampletype,
    promote_samplefloattype

# Julia does not really like expressions of the form
# map((e) -> convert(T, e), collection)
# because type `T` is inside lambda function
# https://github.com/JuliaLang/julia/issues/15276
# https://github.com/JuliaLang/julia/issues/47760
struct PromoteTypeConverter{T,C}
    convert::C
end

PromoteTypeConverter(::Type{T}, convert::C) where {T,C} = PromoteTypeConverter{T,C}(convert)

(converter::PromoteTypeConverter{T})(something) where {T} = converter.convert(T, something)

"""
    deep_eltype(T)

Returns:
- `deep_eltype` of `T` if `T` is an `AbstractArray` container
- `T` otherwise

```jldoctest 
julia> deep_eltype(Float64)
Float64

julia> deep_eltype(Vector{Float64})
Float64

julia> deep_eltype(Vector{Matrix{Vector{Float64}}})
Float64
```
"""
function deep_eltype end

deep_eltype(::Type{T}) where {T} = T
deep_eltype(::Type{T}) where {T<:AbstractArray} = deep_eltype(eltype(T))
deep_eltype(any) = deep_eltype(typeof(any))

"""
    promote_variate_PromoteTypeConverter(::Type{ <: VariateForm }, distribution_type)

Promotes (if possible) a `distribution_type` to be of the specified variate form.
"""
function promote_variate_type end

"""
    promote_variate_type(::Type{D}, distribution_type) where { D <: Distribution }

Promotes (if possible) a `distribution_type` to be of the same variate form as `D`.
"""
function promote_variate_type(::Type{D}, T) where {D<:Distribution}
    return promote_variate_type(variate_form(D), T)
end

"""
    paramfloattype(distribution)

Returns the underlying float type of distribution's parameters.

See also: [`promote_paramfloattype`](@ref), [`convert_paramfloattype`](@ref)
"""
function paramfloattype(distribution::Distribution)
    return promote_type(map(deep_eltype, params(distribution))...)
end
paramfloattype(nt::NamedTuple) = promote_paramfloattype(values(nt))
paramfloattype(t::Tuple) = promote_paramfloattype(t...)

# `Bool` is the smallest possible type, should not play any role in the promotion
paramfloattype(::Nothing) = Bool

"""
    promote_paramfloattype(distributions...)

Promotes `paramfloattype` of the `distributions` to a single type. See also `promote_type`.

See also: [`paramfloattype`](@ref), [`convert_paramfloattype`](@ref)
"""
function promote_paramfloattype(distributions...)
    return promote_type(map(paramfloattype, distributions)...)
end

"""
    convert_paramfloattype(::Type{T}, distribution)

Converts (if possible) the params float type of the `distribution` to be of type `T`.

See also: [`paramfloattype`](@ref), [`promote_paramfloattype`](@ref)
"""
function convert_paramfloattype(::Type{T}, distribution::Distribution) where {T}
    return automatic_convert_paramfloattype(
        distribution_typewrapper(distribution),
        map(convert_paramfloattype(T), params(distribution)),
    )
end
function convert_paramfloattype(::Type{T}, collection::NamedTuple) where {T}
    return map(convert_paramfloattype(T), collection)
end
function convert_paramfloattype(collection::NamedTuple)
    return convert_paramfloattype(paramfloattype(collection), collection)
end
function convert_paramfloattype(::Type{T}) where {T}
    return PromoteTypeConverter(T, convert_paramfloattype)
end

# We attempt to automatically construct a new distribution with a desired paramfloattype
# This function assumes that the constructor `D(...)` accepts the same order of parameters as 
# returned from the `params` function. It is the case for distributions from `Distributions.jl`
automatic_convert_paramfloattype(::Type{D}, params) where {D<:Distribution} = D(params...)
function automatic_convert_paramfloattype(::Type{D}, params) where {D}
    return error(
        "Cannot automatically construct a distribution of type `$D` with params = $(params)"
    )
end

"""
    convert_paramfloattype(::Type{T}, container)

Converts (if possible) the elements of the `container` to be of type `T`.
"""
function convert_paramfloattype(::Type{T}, container::AbstractArray) where {T}
    return convert(AbstractArray{T}, container)
end
convert_paramfloattype(::Type{T}, number::Number) where {T} = convert(T, number)
convert_paramfloattype(::Type, ::Nothing) = nothing

"""
    sampletype(distribution)

Returns a type of the distribution. By default fallbacks to the `eltype`.

See also: [`samplefloattype`](@ref), [`promote_sampletype`](@ref), [`promote_samplefloattype`](@ref)
"""
sampletype(distribution) = eltype(distribution)

function sampletype(distribution::Distribution)
    return sampletype(variate_form(typeof(distribution)), distribution)
end
sampletype(::Type{Univariate}, distribution) = eltype(distribution)
sampletype(::Type{Multivariate}, distribution) = Vector{eltype(distribution)}
sampletype(::Type{Matrixvariate}, distribution) = Matrix{eltype(distribution)}

"""
    samplefloattype(distribution)

Returns a type of the distribution or the underlying float type in case if sample is `Multivariate` or `Matrixvariate`. 
By default fallbacks to the `deep_eltype(sampletype(distribution))`.

See also: [`sampletype`](@ref), [`promote_sampletype`](@ref), [`promote_samplefloattype`](@ref)
"""
samplefloattype(distribution) = deep_eltype(sampletype(distribution))

"""
    promote_sampletype(distributions...)

Promotes `sampletype` of the `distributions` to a single type. See also `promote_type`.

See also: [`sampletype`](@ref), [`samplefloattype`](@ref), [`promote_samplefloattype`](@ref)
"""
promote_sampletype(distributions...) = promote_type(map(sampletype, distributions)...)

"""
    promote_samplefloattype(distributions...)

Promotes `samplefloattype` of the `distributions` to a single type. See also `promote_type`.

See also: [`sampletype`](@ref), [`samplefloattype`](@ref), [`promote_sampletype`](@ref)
"""
function promote_samplefloattype(distributions...)
    return promote_type(map(samplefloattype, distributions)...)
end