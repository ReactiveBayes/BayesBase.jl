@testsnippet ExpectationToys begin
    # Toy backends and measurement helpers shared by the tests of the expectation protocol

    using BayesBase, Distributions, Random, BenchmarkTools

    const P = BayesBase

    "Closed forms of `E[x]` and `E[x²]` under a Normal; exact."
    struct ToyExact end

    P.supports_expectation(::ToyExact, ::Type{typeof(identity)}, ::Type{<:Normal}) = true
    P.supports_expectation(::ToyExact, ::Type{typeof(abs2)}, ::Type{<:Normal}) = true
    P.has_expectation_label(::ToyExact, ::P.ExpectationExact) = true
    P.expectation(::ToyExact, ::typeof(identity), q::Normal) = mean(q)
    P.expectation(::ToyExact, ::typeof(abs2), q::Normal) = mean(q)^2 + var(q)

    """
    Point sets: three-point Gauss–Hermite for a Normal (exact for polynomials up to degree 5, but
    not labelled exact), exact sums for a Categorical and a PointMass.
    """
    struct ToyGrid end

    P.supports_expectation(::ToyGrid, ::Type, ::Type{<:Normal}) = true
    P.supports_expectation(::ToyGrid, ::Type, ::Type{<:Categorical}) = true
    P.supports_expectation(::ToyGrid, ::Type, ::Type{<:PointMass}) = true
    P.has_expectation_label(::ToyGrid, ::Type, ::Type{<:Union{Categorical, PointMass}}, ::P.ExpectationExact) = true

    function P.prepare_expectation(::ToyGrid, q::Normal)
        m, s = mean(q), std(q)
        return P.ExpectationPointSet((m - sqrt(3) * s, m, m + sqrt(3) * s), (1 / 6, 2 / 3, 1 / 6))
    end
    P.prepare_expectation(::ToyGrid, q::Categorical) = P.ExpectationPointSet(1:ncategories(q), probs(q))
    P.prepare_expectation(::ToyGrid, q::PointMass) = P.ExpectationPointSet((mean(q),), (1.0,))

    "Monte Carlo over a Normal with an RNG held by the backend; stochastic."
    struct ToySampler{R}
        n::Int
        rng::R
    end

    ToySampler(n::Int) = ToySampler(n, nothing)

    P.supports_expectation(::ToySampler, ::Type, ::Type{<:Normal}) = true
    P.has_expectation_label(::ToySampler, ::P.ExpectationStochastic) = true
    P.get_expectation_rng(sampler::ToySampler) = sampler.rng
    P.with_expectation_rng(sampler::ToySampler{Nothing}, rng) = ToySampler(sampler.n, rng)
    P.with_expectation_rng(sampler::ToySampler, rng) = sampler

    function P.expectation(sampler::ToySampler, f, q::Normal)
        isnothing(sampler.rng) && throw(ArgumentError("ToySampler has no RNG, use `with_expectation_rng`"))
        return sum(_ -> f(rand(sampler.rng, q)), 1:(sampler.n)) / sampler.n
    end

    "A backend that supports nothing."
    struct ToyNothing end

    """
        folds_to(f, types, value)

    True if the optimised code of `f` for argument `types` returns the constant `value`, that is,
    the result was decided by inference and nothing is computed at run time.
    """
    function folds_to(f, types, value)
        code = only(code_typed(f, types; optimize = true)).first.code
        returns = filter(statement -> statement isa Core.ReturnNode, code)
        length(returns) == 1 || return false
        returned = only(returns).val
        return (returned isa QuoteNode ? returned.value : returned) === value # quoted on Julia 1.10
    end

    """
        allocations(f, args...)

    Allocations of `f(args...)`, measured by BenchmarkTools in a function compiled for the types of
    `f` and `args`. `@allocated` in a generic helper counts the boxed result on Julia 1.10.
    """
    allocations(f, args...) = @ballocated $f($args...) samples = 3 evals = 1
end
