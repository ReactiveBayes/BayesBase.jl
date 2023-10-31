
import Base: prod, prod!, show, showerror

export prod,
    default_prod_rule,
    ClosedProd,
    PreserveTypeProd,
    PreserveTypeLeftProd,
    PreserveTypeRightProd,
    GenericProd,
    ProductOf,
    LinearizedProductOf,
    TerminalProdArgument,
    resolve_prod_strategy

"""
    UnspecifiedProd

A strategy for the `prod` function, which does not compute the `prod`, but instead fails in run-time and prints a descriptive error message.

See also: [`prod`](@ref), [`ClosedProd`](@ref), [`GenericProd`](@ref)
"""
struct UnspecifiedProd end

"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distributions (or any other objects) over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are multiple strategies for prod function, e.g. `ClosedProd`, `GenericProd` or `PreserveTypeProd`.

See also: [`default_prod_rule`](@ref), [`ClosedProd`](@ref), [`PreserveTypeProd`](@ref), [`GenericProd`](@ref)
"""
function Base.prod(strategy::UnspecifiedProd, left, right)
    throw(MethodError(prod, (strategy, left, right)))
end

Base.prod(::UnspecifiedProd, ::Missing, right) = right
Base.prod(::UnspecifiedProd, left, ::Missing) = left
Base.prod(::UnspecifiedProd, ::Missing, ::Missing) = missing

"""
    default_prod_rule(::Type, ::Type)

Returns the most suitable `prod` rule for two given distribution types.
Returns `UnspecifiedProd` by default.

See also: [`prod`](@ref), [`ClosedProd`](@ref), [`GenericProd`](@ref)
"""
default_prod_rule(::Type, ::Type) = UnspecifiedProd()

function default_prod_rule(not_a_type, ::Type{R}) where {R}
    return default_prod_rule(typeof(not_a_type), R)
end

function default_prod_rule(::Type{L}, not_a_type) where {L}
    return default_prod_rule(L, typeof(not_a_type))
end

function default_prod_rule(not_a_type_left, not_a_type_right)
    return default_prod_rule(typeof(not_a_type_left), typeof(not_a_type_right))
end

"""
    PreserveTypeProd{T}

`PreserveTypeProd` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in some specific form.
By default it uses the strategy from `default_prod_rule` and converts the output to the prespecified type but can be overwritten 
for some distributions for better performance.

See also: [`prod`](@ref), [`ClosedProd`](@ref), [`PreserveTypeLeftProd`](@ref), [`PreserveTypeRightProd`](@ref), [`GenericProd`](@ref)
"""
struct PreserveTypeProd{T} end

PreserveTypeProd(::Type{T}) where {T} = PreserveTypeProd{T}()

function Base.prod(::PreserveTypeProd{T}, left, right) where {T}
    return convert(T, prod(symmetric_default_prod_rule(left, right), left, right))
end

Base.prod(::PreserveTypeProd, ::Missing, right) = right
Base.prod(::PreserveTypeProd, left, ::Missing) = left
Base.prod(::PreserveTypeProd, ::Missing, ::Missing) = missing

"""
    PreserveTypeLeftProd

An alias for the `PreserveTypeProd(L)` where `L` is the type of the `left` argument of the `prod` function.

See also: [`prod`](@ref), [`PreserveTypeProd`](@ref), [`PreserveTypeRightProd`](@ref), [`GenericProd`](@ref)
"""
struct PreserveTypeLeftProd end

function Base.prod(::PreserveTypeLeftProd, left::L, right) where {L}
    return prod(PreserveTypeProd(L), left, right)
end

"""
    PreserveTypeRightProd

An alias for the `PreserveTypeProd(R)` where `R` is the type of the `right` argument of the `prod` function.    

See also: [`prod`](@ref), [`PreserveTypeProd`](@ref), [`PreserveTypeLeftProd`](@ref), [`GenericProd`](@ref)
"""
struct PreserveTypeRightProd end

function Base.prod(::PreserveTypeRightProd, left, right::R) where {R}
    return prod(PreserveTypeProd(R), left, right)
end

"""
    ClosedProd

`ClosedProd` is one of the strategies for `prod` function. For example, if both inputs are of type `Distribution`, then `ClosedProd` would fallback to `PreserveTypeProd(Distribution)`.

See also: [`prod`](@ref), [`PreserveTypeProd`](@ref), [`GenericProd`](@ref)
"""
struct ClosedProd end

Base.prod(::ClosedProd, ::Missing, right) = right
Base.prod(::ClosedProd, left, ::Missing) = left
Base.prod(::ClosedProd, ::Missing, ::Missing) = missing

# We assume that we want to preserve the `Distribution` when working with two `Distribution`s
function Base.prod(::ClosedProd, left::Distribution, right::Distribution)
    return prod(PreserveTypeProd(Distribution), left, right)
end

# This is a hidden prod strategy to ensure symmetricity in the `default_prod_rule`.
# Most of the automatic prod rule resolution relies on the `symmetric_default_prod_rule` instead of just `default_prod_rule`
# The `symmetric_default_prod_rule` will adjust the prod rule in case if there is an available prod rule with swapped arguments
struct SwapArgumentsProd{S}
    strategy::S
end

Base.prod(swap::SwapArgumentsProd, left, right) = prod(swap.strategy, right, left)

function symmetric_default_prod_rule(left, right)
    return symmetric_default_prod_rule(
        default_prod_rule(left, right), default_prod_rule(right, left), left, right
    )
end

symmetric_default_prod_rule(strategy1, strategy2, left, right) = strategy1
symmetric_default_prod_rule(strategy1, ::UnspecifiedProd, left, right) = strategy1
function symmetric_default_prod_rule(::UnspecifiedProd, strategy2, left, right)
    return SwapArgumentsProd(strategy2)
end
function symmetric_default_prod_rule(::UnspecifiedProd, ::UnspecifiedProd, left, right)
    return UnspecifiedProd()
end

"""
    ProductOf

A generic structure representing a product of two distributions. 
Can be viewed as a tuple of `(left, right)`. 
Does not check nor supports neither variate forms during the creation stage.
Uses the `fuse_support` function to fuse supports of two different distributions.

This object does not define any statistical properties (such as `mean` or `var` etc) and cannot be used as a distribution explicitly.
Instead, it must be further approximated as a member of some other distribution. 

See also: [`prod`](@ref), [`GenericProd`](@ref), [`fuse_supports`](@ref)
"""
struct ProductOf{L,R}
    left::L
    right::R
end

getleft(product::ProductOf) = product.left
getright(product::ProductOf) = product.right

function Base.:(==)(left::ProductOf, right::ProductOf)
    return (getleft(left) == getleft(right)) && (getright(left) == getright(right))
end

function Base.show(io::IO, product::ProductOf)
    return print(io, "ProductOf(", getleft(product), ",", getright(product), ")")
end

function BayesBase.support(product::ProductOf)
    return fuse_supports(support(getleft(product)), support(getright(product)))
end

BayesBase.pdf(product::ProductOf, x) = exp(logpdf(product, x))

function BayesBase.logpdf(product::ProductOf, x)
    @assert x ∈ support(product) "The `$(x)` does not belong to the support of the product `$(product)`"
    return logpdf(getleft(product), x) + logpdf(getright(product), x)
end

function BayesBase.variate_form(::Type{ProductOf{L,R}}) where {L,R}
    return _check_product_variate_form(variate_form(L), variate_form(R))
end

_check_product_variate_form(::Type{F}, ::Type{F}) where {F<:VariateForm} = F

function _check_product_variate_form(
    ::Type{F1}, ::Type{F2}
) where {F1<:VariateForm,F2<:VariateForm}
    return error(
        "`ProductOf` has different variate forms for left ($F1) and right ($F2) entries."
    )
end

function BayesBase.value_support(::Type{ProductOf{L,R}}) where {L,R}
    return _check_product_value_support(value_support(L), value_support(R))
end

_check_product_value_support(::Type{S}, ::Type{S}) where {S<:ValueSupport} = S

function _check_product_value_support(
    ::Type{S1}, ::Type{S2}
) where {S1<:ValueSupport,S2<:ValueSupport}
    return error(
        "`ProductOf` has different value supports for left ($S1) and right ($S2) entries."
    )
end

"""
    GenericProd

`GenericProd` is one of the strategies for `prod` function. This strategy does always produces a result, 
even if the closed form product is not availble, in which case simply returns the `ProductOf` object. `GenericProd` sometimes 
fallbacks to the `default_prod_rule` which it may or may not use under some circumstances. 
For example if the `default_prod_rule` is `ClosedProd` - `GenericProd` will try to optimize the tree with 
analytical closed solutions (if possible).

See also: [`prod`](@ref), [`ProductOf`](@ref), [`ClosedProd`](@ref), [`PreserveTypeProd`](@ref), [`default_prod_rule`](@ref)
"""
struct GenericProd end

Base.show(io::IO, ::GenericProd) = print(io, "GenericProd()")

Base.prod(::GenericProd, ::Missing, right) = right
Base.prod(::GenericProd, left, ::Missing) = left
Base.prod(::GenericProd, ::Missing, ::Missing) = missing

function Base.prod(::GenericProd, left::L, right::R) where {L,R}
    return prod(GenericProd(), symmetric_default_prod_rule(L, R), left, right)
end

Base.prod(::GenericProd, specified_prod, left, right) = prod(specified_prod, left, right)
Base.prod(::GenericProd, ::UnspecifiedProd, left, right) = ProductOf(left, right)

# Try to fuse the tree with analytical solutions (if possible)
# Case (L × R) × T
function Base.prod(::GenericProd, left::ProductOf{L,R}, right::T) where {L,R,T}
    return prod(
        GenericProd(),
        symmetric_default_prod_rule(L, T),
        symmetric_default_prod_rule(R, T),
        left,
        right,
    )
end

# (L × R) × T cannot be fused, simply return the `ProductOf`
function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::ProductOf, right
)
    return ProductOf(left, right)
end

# (L × R) × T can be fused efficiently as (L × T) × R, because L × T has defined the `something` default prod
function Base.prod(::GenericProd, something, ::UnspecifiedProd, left::ProductOf, right)
    return ProductOf(prod(something, getleft(left), right), getright(left))
end

# (L × R) × T can be fused efficiently as L × (R × T), because R × T has defined the `something` default prod
function Base.prod(::GenericProd, ::UnspecifiedProd, something, left::ProductOf, right)
    return ProductOf(getleft(left), prod(something, getright(left), right))
end

# (L × R) × T can be fused efficiently as L × (R × T), because both L × T and R × T has defined the `something` default prod, but we choose R × T
function Base.prod(::GenericProd, _, something, left::ProductOf, right)
    return ProductOf(getleft(left), prod(something, getright(left), right))
end

# Case T × (L × R)
function Base.prod(::GenericProd, left::T, right::ProductOf{L,R}) where {L,R,T}
    return prod(
        GenericProd(),
        symmetric_default_prod_rule(T, L),
        symmetric_default_prod_rule(T, R),
        left,
        right,
    )
end

# T × (L × R) cannot be fused, simply return the `ProductOf`
function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left, right::ProductOf
)
    return ProductOf(left, right)
end

# T × (L × R) can be fused efficiently as (T × L) × R, because T × L has defined the `something` default prod
function Base.prod(::GenericProd, something, ::UnspecifiedProd, left, right::ProductOf)
    return ProductOf(prod(something, left, getleft(right)), getright(right))
end

# T × (L × R) can be fused efficiently as L × (T × R), because T × R has defined the `something` default prod
function Base.prod(::GenericProd, ::UnspecifiedProd, something, left, right::ProductOf)
    return ProductOf(getleft(right), prod(something, left, getright(right)))
end

# T × (L × R) can be fused efficiently as L × (T × R), because both T × L and T × R has defined the `something` default prod, but we choose T × L
function Base.prod(::GenericProd, something, _, left, right::ProductOf)
    return ProductOf(prod(something, left, getleft(right)), getright(right))
end

"""
    LinearizedProductOf

An efficient __linearized__ implementation of product of multiple distributions.
This structure prevents `ProductOf` tree from growing too much in case of identical objects. 
This trick significantly reduces Julia compilation times when closed product rules are not available but distributions are of the same type.
Essentially this structure linearizes leaves of the `ProductOf` tree in case if it sees objects of the same type (via dispatch).

See also: [`ProductOf`](@ref), [`GenericProd`]
"""
struct LinearizedProductOf{F}
    vector::Vector{F}
    length::Int # `length` here is needed for extra safety as we implicitly mutate `vector` in `prod`
end

function Base.push!(product::LinearizedProductOf{F}, item::F) where {F}
    vector = product.vector
    vlength = length(vector)
    return LinearizedProductOf(push!(vector, item), vlength + 1)
end

BayesBase.support(dist::LinearizedProductOf) = support(first(dist.vector))

Base.length(product::LinearizedProductOf) = product.length
Base.eltype(product::LinearizedProductOf) = eltype(first(product.vector))

function Base.:(==)(left::LinearizedProductOf, right::LinearizedProductOf)
    return (left.length == right.length) && (left.vector == right.vector)
end

function BayesBase.samplefloattype(product::LinearizedProductOf)
    return samplefloattype(first(product.vector))
end

BayesBase.variate_form(::Type{<:LinearizedProductOf{F}}) where {F} = variate_form(F)
BayesBase.value_support(::Type{<:LinearizedProductOf{F}}) where {F} = value_support(F)

function Base.show(io::IO, product::LinearizedProductOf{F}) where {F}
    return print(io, "LinearizedProductOf(", F, ", length = ", product.length, ")")
end

function BayesBase.logpdf(product::LinearizedProductOf, x)
    @assert x ∈ support(product) "The `$(x)` does not belong to the support of the product `$(product)`"
    return mapreduce(
        (d) -> logpdf(d, x),
        +,
        view(product.vector, 1:min(product.length, length(product.vector))),
    )
end

BayesBase.pdf(dist::LinearizedProductOf, x) = exp(logpdf(dist, x))

# We assume that it is better (really) to preserve the type of the `LinearizedProductOf`, it is just faster for the compiler
function BayesBase.default_prod_rule(::Type{F}, ::Type{LinearizedProductOf{F}}) where {F}
    return PreserveTypeProd(LinearizedProductOf{F})
end
function BayesBase.default_prod_rule(::Type{LinearizedProductOf{F}}, ::Type{F}) where {F}
    return PreserveTypeProd(LinearizedProductOf{F})
end

function Base.prod(
    ::PreserveTypeProd{LinearizedProductOf{F}}, product::LinearizedProductOf{F}, item::F
) where {F}
    return push!(product, item)
end

function Base.prod(
    ::PreserveTypeProd{LinearizedProductOf{F}}, item::F, product::LinearizedProductOf{F}
) where {F}
    return push!(product, item)
end

function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::ProductOf{F,F}, right::F
) where {F}
    return LinearizedProductOf(F[getleft(left), getright(left), right], 3)
end

function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::ProductOf{L,R}, right::R
) where {L,R}
    return ProductOf(getleft(left), LinearizedProductOf(R[getright(left), right], 2))
end

function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::ProductOf{L,R}, right::L
) where {L,R}
    return ProductOf(LinearizedProductOf(L[getleft(left), right], 2), getright(left))
end

function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::L, right::ProductOf{L,R}
) where {L,R}
    return ProductOf(LinearizedProductOf(L[left, getleft(right)], 2), getright(right))
end

function Base.prod(
    ::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::R, right::ProductOf{L,R}
) where {L,R}
    return ProductOf(getleft(right), LinearizedProductOf(R[left, getright(right)], 2))
end

function Base.prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{L,LinearizedProductOf{R}},
    right::R,
) where {L,R}
    return ProductOf(getleft(left), push!(getright(left), right))
end

function Base.prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{LinearizedProductOf{L},R},
    right::L,
) where {L,R}
    return ProductOf(push!(getleft(left), right), getright(left))
end

"""
    TerminalProdArgument(argument)

`TerminalProdArgument` is a specialized wrapper structure. When used as an argument to the `prod` function, it returns itself without considering any product strategy 
and does not perform any safety checks (e.g. `variate_form` or `support`). Attempting to calculate the product of two instances of `TerminalProdArgument` will raise an error. 
Use `.argument` field to get the underlying wrapped argument.
"""
struct TerminalProdArgument{T}
    argument::T
end

BayesBase.mean(prod::TerminalProdArgument) = mean(prod.argument)
BayesBase.median(prod::TerminalProdArgument) = median(prod.argument)
BayesBase.mode(prod::TerminalProdArgument) = mode(prod.argument)
BayesBase.shape(prod::TerminalProdArgument) = shape(prod.argument)
BayesBase.scale(prod::TerminalProdArgument) = scale(prod.argument)
BayesBase.rate(prod::TerminalProdArgument) = rate(prod.argument)
BayesBase.var(prod::TerminalProdArgument) = var(prod.argument)
BayesBase.std(prod::TerminalProdArgument) = std(prod.argument)
BayesBase.cov(prod::TerminalProdArgument) = cov(prod.argument)
BayesBase.invcov(prod::TerminalProdArgument) = invcov(prod.argument)
BayesBase.logdetcov(prod::TerminalProdArgument) = logdetcov(prod.argument)
BayesBase.entropy(prod::TerminalProdArgument) = entropy(prod.argument)
BayesBase.params(prod::TerminalProdArgument) = params(prod.argument)
BayesBase.mean_cov(prod::TerminalProdArgument) = mean_cov(prod.argument)
BayesBase.mean_var(prod::TerminalProdArgument) = mean_var(prod.argument)
BayesBase.mean_invcov(prod::TerminalProdArgument) = mean_invcov(prod.argument)
BayesBase.mean_precision(prod::TerminalProdArgument) = mean_precision(prod.argument)
BayesBase.weightedmean_cov(prod::TerminalProdArgument) = weightedmean_cov(prod.argument)
BayesBase.weightedmean_var(prod::TerminalProdArgument) = weightedmean_var(prod.argument)
BayesBase.probvec(prod::TerminalProdArgument) = probvec(prod.argument)
BayesBase.weightedmean(prod::TerminalProdArgument) = weightedmean(prod.argument)
BayesBase.paramfloattype(prod::TerminalProdArgument) = paramfloattype(prod.argument)
BayesBase.samplefloattype(prod::TerminalProdArgument) = samplefloattype(prod.argument)

function BayesBase.weightedmean_invcov(prod::TerminalProdArgument)
    return weightedmean_invcov(prod.argument)
end
function BayesBase.weightedmean_precision(prod::TerminalProdArgument)
    return weightedmean_precision(prod.argument)
end

Base.precision(prod::TerminalProdArgument) = precision(prod.argument)
Base.length(prod::TerminalProdArgument) = length(prod.argument)
Base.ndims(prod::TerminalProdArgument) = ndims(prod.argument)
Base.size(prod::TerminalProdArgument) = size(prod.argument)
Base.eltype(prod::TerminalProdArgument) = eltype(prod.argument)

function Base.show(io::IO, prod::TerminalProdArgument)
    return print(io, "TerminalProdArgument(", prod.argument, ")")
end

Base.convert(::Type{TerminalProdArgument}, something) = TerminalProdArgument(something)
Base.convert(::Type{TerminalProdArgument}, terminal::TerminalProdArgument) = terminal

Base.isapprox(left::TerminalProdArgument, right; kwargs...) = false
Base.isapprox(left, right::TerminalProdArgument; kwargs...) = false
function Base.isapprox(left::TerminalProdArgument{T}, right::T; kwargs...) where {T}
    return isapprox(left.argument, right; kwargs...)
end
function Base.isapprox(left::T, right::TerminalProdArgument{T}; kwargs...) where {T}
    return isapprox(left, right.argument; kwargs...)
end
function Base.isapprox(left::TerminalProdArgument, right::TerminalProdArgument; kwargs...)
    return isapprox(left.argument, right.argument; kwargs...)
end

function BayesBase.convert_paramfloattype(
    ::Type{T}, terminal::TerminalProdArgument
) where {T}
    return TerminalProdArgument(convert_paramfloattype(T, terminal.argument))
end

function default_prod_rule(::Type{<:TerminalProdArgument}, ::Type{T}) where {T}
    return PreserveTypeProd(TerminalProdArgument)
end
function default_prod_rule(::Type{T}, ::Type{<:TerminalProdArgument}) where {T}
    return PreserveTypeProd(TerminalProdArgument)
end
function default_prod_rule(::Type{<:TerminalProdArgument}, ::Type{<:TerminalProdArgument})
    return PreserveTypeProd(TerminalProdArgument)
end

function Base.prod(
    ::PreserveTypeProd{TerminalProdArgument}, left::TerminalProdArgument, right
)
    return left
end
function Base.prod(
    ::PreserveTypeProd{TerminalProdArgument}, left, right::TerminalProdArgument
)
    return right
end
function Base.prod(
    ::PreserveTypeProd{TerminalProdArgument},
    left::TerminalProdArgument,
    right::TerminalProdArgument,
)
    return error("Invalid product: `$(left)` × `$(right)`")
end

"""
    resolve_prod_strategy(left, right)

Given two strategies, this function returns the one with higher priority, if possible.
"""
function resolve_prod_strategy(left, right)
    return error(
        "Cannot resolve strategies `$(left)` and $(right). Strategies have the same priority.",
    )
end

function resolve_prod_strategy(left::T, right::T) where {T}
    if (left === right)
        return left
    else
        error(
            "Cannot resolve strategies `$(left)` and $(right). Strategies have the same priority.",
        )
    end
end

resolve_prod_strategy(any, ::GenericProd) = any
resolve_prod_strategy(::GenericProd, any) = any
resolve_prod_strategy(::GenericProd, ::GenericProd) = GenericProd()
resolve_prod_strategy(::Nothing, ::GenericProd) = GenericProd()
resolve_prod_strategy(::GenericProd, ::Nothing) = GenericProd()

resolve_prod_strategy(::Nothing, any) = any
resolve_prod_strategy(any, ::Nothing) = any
resolve_prod_strategy(::Nothing, ::Nothing) = nothing