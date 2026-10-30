module BayesBaseReactantExt

using BayesBase, Reactant, LinearAlgebra
using Reactant: TracedRArray, AnyTracedRArray

const Traced = MatrixWithCachedInverse{<:Any,<:TracedRArray}

# Symmetry cannot be decided on traced data.
BayesBase._checksymmetric(::AnyTracedRArray) = nothing

# Reactant's post-trace walk (`NoStopTracedTrack`) has no visited-object guard, and the reciprocal
# pair is a cycle; stop at wrappers already seen in that mode (every other mode already does).
function Reactant.make_tracer(
    seen, prev::MatrixWithCachedInverse, @nospecialize(path), mode; kwargs...
)
    mode == Reactant.NoStopTracedTrack && haskey(seen, prev) && return seen[prev]
    return Reactant.make_tracer_unknown(seen, prev, path, mode; kwargs...)
end

# Caches are never written while tracing: Reactant cannot write trace-local values back into an
# empty host cache, and a host cache must not hold values from one particular compilation.
# Host-populated caches are used as program inputs; the `factorize`/`invert` recipes are ignored
# because only Reactant's own `cholesky` lowers to the program.
function BayesBase.cached_cholesky_or_compute(::Any, A::Traced)
    @something(BayesBase.cached_factor(A), cholesky(parent(A)))
end

function BayesBase.cached_inverse_or_compute(::Any, A::Traced)
    A.inverse === nothing || return A.inverse
    T = Reactant.unwrapped_eltype(A)
    Ainv = A \ Reactant.promote_to(TracedRArray{T,2}, I(size(A, 1)))
    B = MatrixWithCachedInverse{eltype(Ainv),typeof(Ainv)}(Ainv, A.guard)
    B.inverse = A
    return B
end

# LinearAlgebra's `Cholesky \ b` uses an adjoint-wrapped triangular that Reactant does not
# overload, so solve with the two triangular systems directly.
function Base.:\(A::Traced, b::AbstractVecOrMat)
    C = BayesBase.cached_cholesky_or_compute(A)
    if C.uplo == 'U'
        U = UpperTriangular(C.factors)
        return U \ (adjoint(U) \ b)
    else
        L = LowerTriangular(C.factors)
        return adjoint(L) \ (L \ b)
    end
end

function LinearAlgebra.logdet(A::Traced)
    return 2 * sum(log, diag(BayesBase.cached_cholesky_or_compute(A).factors))
end

end
