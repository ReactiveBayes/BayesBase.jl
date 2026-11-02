"""
    ExpectationLabel

A property of the computation of an expectation that a caller can rely on. Labels are
conservative: a backend declares a label only where it is guaranteed.
"""
abstract type ExpectationLabel end

"""
    ExpectationExact()

The result equals the expectation up to rounding.
"""
struct ExpectationExact <: ExpectationLabel end

"""
    ExpectationStochastic()

The result depends on a random number generator.
"""
struct ExpectationStochastic <: ExpectationLabel end

"""
    has_expectation_label(backend, label)
    has_expectation_label(backend, F::Type, Q::Type, label)
    has_expectation_label(backend, f, q, label)

Whether `backend` has `label`: for every pair it supports (the first form, `false` by
default), or for one pair. The pair forms default to the first one; the instance form is
`false` when the pair is not supported.
"""
has_expectation_label(backend, ::ExpectationLabel) = false
function has_expectation_label(backend, ::Type{F}, ::Type{Q}, label::ExpectationLabel) where {F, Q}
    return has_expectation_label(backend, label)
end
function has_expectation_label(backend, f, q, label::ExpectationLabel)
    return supports_expectation(backend, f, q) && has_expectation_label(backend, typeof(f), typeof(q), label)
end

"Whether `backend` computes the expectation of `f` under `q` exactly."
is_expectation_exact(backend, f, q) = has_expectation_label(backend, f, q, ExpectationExact())

"Whether `backend` uses random numbers, for every pair or for the pair `(f, q)`."
is_expectation_stochastic(backend) = has_expectation_label(backend, ExpectationStochastic())
is_expectation_stochastic(backend, f, q) = has_expectation_label(backend, f, q, ExpectationStochastic())

"""
    combine_expectation_label(label, a::Bool, b::Bool)

Whether a computation made of two parts has `label`, given whether each part has it.
`ExpectationExact` needs both parts to be exact, stochastic either part to be stochastic. A label
defined elsewhere is combined conservatively, like `ExpectationExact`, unless its owner adds a
method.
"""
combine_expectation_label(::ExpectationLabel, a::Bool, b::Bool) = a & b
combine_expectation_label(::ExpectationStochastic, a::Bool, b::Bool) = a | b
