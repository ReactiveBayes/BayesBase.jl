# Macros that define a closed form and declare its support in one step, so that the two cannot
# drift apart. They parse a method definition with three positional arguments (the backend, the
# integrand and the distribution), keep it, and add the methods of `supports_expectation` (and, for
# a gradient, the method of `expectation` that routers call) for the same argument types.

"""
    @supports_expectation function BayesBase.expectation(backend::B, f::F, q::Q) where {…}
        …
    end

Define a closed form of [`BayesBase.expectation`](@ref) and declare that the backend supports it.
Besides the method itself, the macro defines

```julia
BayesBase.supports_expectation(::B, ::Type{<:F}, ::Type{<:Q}) where {…} = true
```

with the types and the `where` parameters of the definition. An argument without a type means
`Any`. An integrand without a type therefore declares every integrand, as a numerical backend
does, see [`BayesBase.supports_expectation`](@ref). The definition can be in long or short form,
and a docstring placed before the macro documents the method. Only methods of `expectation` with
three positional arguments are accepted. Keyword arguments and default values are not.

Support that depends on more than the types of one method, such as "wherever another backend has
a closed form", is declared with a method of `supports_expectation` written out instead.

The macro is public but not exported: write `BayesBase.@supports_expectation`, or import it with
`import BayesBase: @supports_expectation`.

```jldoctest
julia> using Distributions

julia> struct Moments end

julia> BayesBase.@supports_expectation function BayesBase.expectation(::Moments, ::typeof(identity), q::Gamma)
           α, θ = params(q)
           return α * θ
       end

julia> BayesBase.supports_expectation(Moments(), identity, Gamma(2.0, 1.0))
true

julia> BayesBase.supports_expectation(Moments(), identity, Normal(0.0, 1.0))
false

julia> BayesBase.expectation(Moments(), identity, Gamma(2.0, 3.0))
6.0
```
"""
macro supports_expectation(definition)
    method = split_expectation_definition(definition, :expectation, "@supports_expectation")
    support = expectation_support_definition(method, method.F)
    return esc(Expr(:block, documented(definition, __source__), support, nothing))
end

"""
    @supports_expectation_gradient function BayesBase.expectation_gradient(backend::B, f::F, q::Q) where {…}
        …
    end

Define a closed form of [`BayesBase.expectation_gradient`](@ref) and declare that the backend
supports it. Besides the method itself, the macro defines the method that routers reach, which
passes the integrand on to the closed form,

```julia
BayesBase.expectation(backend::B, product::ExpectationScoreProduct{<:F, <:Q}, q::Q) where {…} =
    BayesBase.expectation_gradient(backend, product.f, q)
```

and the declaration

```julia
BayesBase.supports_expectation(::B, ::Type{<:ExpectationScoreProduct{<:F, <:Q}}, ::Type{<:Q}) where {…} = true
```

see [`BayesBase.ExpectationScoreProduct`](@ref). The result must be in the parametrization of
[`BayesBase.expectation_score`](@ref). Otherwise the macro works as
[`BayesBase.@supports_expectation`](@ref).

```jldoctest
julia> using Distributions

julia> struct Moments end

julia> BayesBase.@supports_expectation_gradient function BayesBase.expectation_gradient(::Moments, ::typeof(identity), q::Gamma)
           α, θ = params(q)
           return [θ, α]  # the gradient of α θ with respect to the shape α and the scale θ
       end

julia> BayesBase.supports_expectation_gradient(Moments(), identity, Gamma(2.0, 1.0))
true

julia> router = BayesBase.ExpectationFirstOf(Moments());

julia> BayesBase.expectation_gradient(router, identity, Gamma(2.0, 3.0))
2-element Vector{Float64}:
 3.0
 2.0
```
"""
macro supports_expectation_gradient(definition)
    method = split_expectation_definition(definition, :expectation_gradient, "@supports_expectation_gradient")
    product = Expr(:curly, ExpectationScoreProduct, Expr(:<:, method.F), Expr(:<:, method.Q))
    backend, integrand, distribution = gensym(:backend), gensym(:product), gensym(:q)
    signature = Expr(
        :call,
        qualified(:expectation),
        Expr(:(::), backend, method.B),
        Expr(:(::), integrand, product),
        Expr(:(::), distribution, method.Q),
    )
    forward = Expr(
        :(=),
        with_where(signature, method.parameters),
        Expr(:call, qualified(:expectation_gradient), backend, Expr(:., integrand, QuoteNode(:f)), distribution),
    )
    support = expectation_support_definition(method, product)
    return esc(Expr(:block, documented(definition, __source__), forward, support, nothing))
end

# The parts of a method definition that the macros need: the types of the three arguments and
# the `where` parameters that they use
function split_expectation_definition(definition, name::Symbol, macroname::String)
    fail(reason) = throw(ArgumentError("$macroname expects a method definition of `$name`: $reason"))
    (Meta.isexpr(definition, :function) || Meta.isexpr(definition, :(=))) && length(definition.args) == 2 ||
        fail("got `$definition`")
    parameters = Any[]
    signature = peel_where!(parameters, definition.args[1])
    # A return type, `f(…)::T`. In the short form, `f(…)::T where {T} = …` attaches the `where` to
    # the return type, as for any method definition, so the parameters of a short form with a
    # return type are written `(f(…)::T) where {T} = …`
    Meta.isexpr(signature, :(::)) && length(signature.args) == 2 && (signature = signature.args[1])
    Meta.isexpr(signature, :call) || fail("got `$definition`")
    function_name(signature.args[1]) === name || fail("the function is `$(signature.args[1])`")
    arguments = signature.args[2:end]
    any(argument -> Meta.isexpr(argument, :parameters), arguments) && fail("keyword arguments are not supported")
    length(arguments) == 3 || fail("it must have three positional arguments, the backend, the integrand and the distribution, got $(length(arguments))")
    B, F, Q = map(argument -> argument_type(argument, fail), arguments)
    used = filter(parameter -> any(type -> mentions(type, parameter_name(parameter)), (B, F, Q)), parameters)
    return (; B, F, Q, parameters = used)
end

# Collects the parameters of `expression where {…} where {…}`, the outermost first, and returns
# what they apply to
function peel_where!(parameters, expression)
    while Meta.isexpr(expression, :where)
        append!(parameters, expression.args[2:end])
        expression = expression.args[1]
    end
    return expression
end

function_name(name::Symbol) = name
function_name(name::GlobalRef) = name.name
function_name(name::Expr) = Meta.isexpr(name, :.) && name.args[end] isa QuoteNode ? name.args[end].value : nothing
function_name(name) = nothing

argument_type(::Symbol, fail) = :Any
function argument_type(argument::Expr, fail)
    Meta.isexpr(argument, :(::)) && return last(argument.args)
    Meta.isexpr(argument, :kw) && fail("default values are not supported")
    return fail("cannot read the argument `$argument`")
end
argument_type(argument, fail) = fail("cannot read the argument `$argument`")

# The name of a `where` parameter: `T`, `T <: U`, `L <: T <: U`
parameter_name(parameter::Symbol) = parameter
function parameter_name(parameter::Expr)
    Meta.isexpr(parameter, :<:) && return parameter_name(parameter.args[1])
    Meta.isexpr(parameter, :>:) && return parameter_name(parameter.args[1])
    Meta.isexpr(parameter, :comparison) && return parameter_name(parameter.args[3])
    return nothing
end

mentions(expression, name) = expression === name
mentions(expression::Expr, name) = any(argument -> mentions(argument, name), expression.args)

# `BayesBase.name`, with the module itself, so it resolves wherever the macro is called
qualified(name::Symbol) = Expr(:., BayesBase, QuoteNode(name))

with_where(signature, parameters) = isempty(parameters) ? signature : Expr(:where, signature, parameters...)

# `supports_expectation(::B, ::Type{<:F}, ::Type{<:Q}) where {…} = true`
function expectation_support_definition(method, F)
    signature = Expr(
        :call,
        qualified(:supports_expectation),
        Expr(:(::), method.B),
        Expr(:(::), Expr(:curly, Type, Expr(:<:, F))),
        Expr(:(::), Expr(:curly, Type, Expr(:<:, method.Q))),
    )
    return Expr(:(=), with_where(signature, method.parameters), true)
end

# The definition, marked so that a docstring before the macro call documents the method
function documented(definition, source)
    return Expr(:macrocall, GlobalRef(Base, Symbol("@__doc__")), source, definition)
end
