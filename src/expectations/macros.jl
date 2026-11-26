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
three positional arguments are accepted. Keyword arguments and default values are not. The method
always extends BayesBase's function: a bare `expectation` is read as `BayesBase.expectation`, even
where the name is not imported, and a function of another module is rejected.

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
    method = split_expectation_definition(definition, :expectation, "@supports_expectation", __module__)
    support = expectation_support_definition(method, method.F, __source__)
    return esc(Expr(:block, documented(method.definition, __source__), support, nothing))
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
    method = split_expectation_definition(definition, :expectation_gradient, "@supports_expectation_gradient", __module__)
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
        Expr(
            :block,
            __source__,
            Expr(:call, qualified(:expectation_gradient), backend, Expr(:., integrand, QuoteNode(:f)), distribution),
        ),
    )
    support = expectation_support_definition(method, product, __source__)
    return esc(Expr(:block, documented(method.definition, __source__), forward, support, nothing))
end

# The parts of a method definition that the macros need: the definition, with its function
# rewritten to `BayesBase.name` so that it always extends BayesBase's function, the types of the
# three arguments and the `where` parameters that they use
function split_expectation_definition(definition, name::Symbol, macroname::String, mod::Module)
    fail(reason) = throw(ArgumentError("$macroname expects a method definition of `BayesBase.$name`: $reason"))
    (Meta.isexpr(definition, :function) || Meta.isexpr(definition, :(=))) && length(definition.args) == 2 ||
        fail("got `$definition`")
    definition = copy(definition) # copies the expressions inside, whose function is rewritten below
    parameters = Any[]
    signature = peel_where!(parameters, definition.args[1])
    # A return type, `f(…)::T`. In the short form, `f(…)::T where {T} = …` attaches the `where` to
    # the return type, as for any method definition, so the parameters of a short form with a
    # return type are written `(f(…)::T) where {T} = …`
    Meta.isexpr(signature, :(::)) && length(signature.args) == 2 && (signature = signature.args[1])
    Meta.isexpr(signature, :call) || fail("got `$definition`")
    callee = signature.args[1]
    function_name(callee) === name || fail("the function is `$callee`")
    owner = function_owner(callee, mod)
    owner === nothing || owner === BayesBase ||
        fail("the function is `$callee`, not BayesBase's `$name`")
    signature.args[1] = qualified(name)
    arguments = signature.args[2:end]
    any(argument -> Meta.isexpr(argument, :parameters), arguments) && fail("keyword arguments are not supported")
    length(arguments) == 3 || fail("it must have three positional arguments, the backend, the integrand and the distribution, got $(length(arguments))")
    B, F, Q = map(argument -> argument_type(argument, fail), arguments)
    return (; definition, B, F, Q, parameters = used_parameters(parameters, (B, F, Q)))
end

# The `where` parameters that the argument types mention, directly or through the bound of
# another parameter that is kept, in their original order
function used_parameters(parameters, types)
    kept = Set(parameter_name(p) for p in parameters if any(type -> mentions(type, parameter_name(p)), types))
    grown = true
    while grown
        grown = false
        for parameter in parameters, other in parameters
            name = parameter_name(other)
            if parameter_name(parameter) in kept && !(name in kept) && mentions(parameter, name)
                push!(kept, name)
                grown = true
            end
        end
    end
    return filter(parameter -> parameter_name(parameter) in kept, parameters)
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

# The module that a qualified function name refers to, `nothing` for a bare name
function_owner(::Symbol, mod) = nothing
function_owner(name::GlobalRef, mod) = name.mod
function_owner(name::Expr, mod) = resolve_module(name.args[1], mod)

resolve_module(path::Symbol, mod) = isdefined(mod, path) ? getfield(mod, path) : path
resolve_module(path::Module, mod) = path
function resolve_module(path::Expr, mod)
    Meta.isexpr(path, :.) && path.args[end] isa QuoteNode || return path
    parent = resolve_module(path.args[1], mod)
    name = path.args[end].value
    return parent isa Module && isdefined(parent, name) ? getfield(parent, name) : path
end
resolve_module(path, mod) = path

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
function expectation_support_definition(method, F, source)
    signature = Expr(
        :call,
        qualified(:supports_expectation),
        Expr(:(::), method.B),
        Expr(:(::), Expr(:curly, Type, Expr(:<:, F))),
        Expr(:(::), Expr(:curly, Type, Expr(:<:, method.Q))),
    )
    return Expr(:(=), with_where(signature, method.parameters), Expr(:block, source, true))
end

# The definition, marked so that a docstring before the macro call documents the method
function documented(definition, source)
    return Expr(:macrocall, GlobalRef(Base, Symbol("@__doc__")), source, definition)
end
