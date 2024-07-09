export Contingency

using LinearAlgebra

"""
    Contingency(P, renormalize = Val(true))

The contingency distribution is a multivariate generalization of the categorical distribution. As a bivariate distribution, the 
contingency distribution defines the joint probability over two unit vectors `v1` and `v2`. The parameter `P` encodes a contingency matrix that specifies the probability of co-occurrence.

    v1 ∈ {0, 1}^d1 where Σ_j v1_j = 1
    v2 ∈ {0, 1}^d2 where Σ_k v2_k = 1

    P ∈ [0, 1]^{d1 × d2}, where Σ_jk P_jk = 1

    f(v1, v2, P) = Contingency(out1, out2 | P) = Π_jk P_jk^{v1_j * v2_k}

A `Contingency` distribution over more than two variables requires higher-order tensors as parameters; these are not implemented in ReactiveMP.

# Arguments:
- `P`, required, contingency matrix
- `renormalize`, optional, supports either `Val(true)` or `Val(false)`, specifies whether matrix `P` must be automatically renormalized. Does not modify the original `P` and allocates a new one for the renormalized version. If set to `false` the contingency matrix `P` **must** be normalized by hand, otherwise the result of related calculations might be wrong

"""
struct Contingency{T,P<:AbstractArray{T}}
    p::P

    Contingency{T,P}(A::AbstractArray) where {T,P<:AbstractArray{T}} = new(A)
end

Contingency(P::AbstractArray) = Contingency(P, Val(true))

function Contingency(P::M, ::Val{true}) where {T,M<:AbstractArray{T}}
    return Contingency{T,M}(P ./ sum(P))
end

Contingency(P::M, ::Val{false}) where {T,M<:AbstractArray{T}} = Contingency{T,M}(P)

BayesBase.components(distribution::Contingency) = distribution.p
BayesBase.component(distribution::Contingency, k) = distribution.p[:, k]

function BayesBase.vague(::Type{<:Contingency}, dims::Int, nvars::Int=2)
    return Contingency(ones([dims for _ in 1:nvars]...))
end

BayesBase.paramfloattype(distribution::Contingency) = deep_eltype(components(distribution))

function BayesBase.convert_paramfloattype(
    ::Type{T}, distribution::Contingency
) where {T<:Real}
    return Contingency(convert_paramfloattype(T, components(distribution)))
end

function BayesBase.entropy(distribution::Contingency)
    P = components(distribution)
    return -mapreduce((p) -> p * clamplog(p), +, P)
end

function Base.isapprox(left::Contingency, right::Contingency; kwargs...)
    return isapprox(components(left), components(right); kwargs...)
end