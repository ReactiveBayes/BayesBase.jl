"""
    ExpectationLabel

A property of the computation of an expectation that a caller can rely on. Labels are
conservative: a backend declares a label only where it is guaranteed.
"""
abstract type ExpectationLabel end

"""
    Exact()

The result equals the expectation up to rounding.
"""
struct Exact <: ExpectationLabel end

"""
    Stochastic()

The result depends on a random number generator.
"""
struct Stochastic <: ExpectationLabel end

"""
    haslabel(backend, label)
    haslabel(backend, F::Type, Q::Type, label)
    haslabel(backend, f, q, label)

Whether `backend` has `label`: for every pair it supports (the first form, `false` by
default), or for one pair. The pair forms default to the first one; the instance form is
`false` when the pair is not supported.
"""
haslabel(backend, ::ExpectationLabel) = false
function haslabel(backend, ::Type{F}, ::Type{Q}, label::ExpectationLabel) where {F, Q}
    return haslabel(backend, label)
end
function haslabel(backend, f, q, label::ExpectationLabel)
    return supports(backend, f, q) && haslabel(backend, typeof(f), typeof(q), label)
end

"Whether `backend` computes the expectation of `f` under `q` exactly."
isexact(backend, f, q) = haslabel(backend, f, q, Exact())

"Whether `backend` uses random numbers, for every pair or for the pair `(f, q)`."
isstochastic(backend) = haslabel(backend, Stochastic())
isstochastic(backend, f, q) = haslabel(backend, f, q, Stochastic())

"""
    combine_label(label, a::Bool, b::Bool)

Whether a computation made of two parts has `label`, given whether each part has it. Exact
needs both parts to be exact, stochastic either part to be stochastic. A label defined
elsewhere is combined conservatively, like `Exact`, unless its owner adds a method.
"""
combine_label(::ExpectationLabel, a::Bool, b::Bool) = a & b
combine_label(::Stochastic, a::Bool, b::Bool) = a | b
