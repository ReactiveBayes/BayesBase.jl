export MatrixWithCachedInverse

using LinearAlgebra, ReadOnlyArrays

"""
    MatrixWithCachedInverse(A::AbstractMatrix)

An `AbstractMatrix` wrapper around an owned copy of a dense, real, symmetric positive-definite
matrix that lazily caches its Cholesky factor and its inverse.

Construction copies `A`, checks that it is square and symmetric, and performs no factorization.
The first `A \\ b` or `logdet(A)` computes and caches the Cholesky factor. The first `inv(A)`
computes the inverse once and returns another `MatrixWithCachedInverse` that already links back,
so `inv(A) === inv(A)` and `inv(inv(A)) === A` hold without any recomputation. Positive
definiteness is checked on the first factorization request (`PosDefException`).

Element writes through the wrapper and its inverse are rejected; `Matrix(A)` and `copy(A)` return
independent writable dense copies. `AbstractMatrix{T}(A)` returns a new wrapper with empty caches,
while `convert(AbstractMatrix{T}, A)` returns `A` itself when the element type already matches.
Lazy initialization is synchronized with a lock shared by the pair, so concurrent first requests
share one result.

This first version supports dense `Matrix{T}` inputs for `T` in `Float32`, `Float64` and
`BigFloat`. After the inverse is computed, the pair keeps the input, its Cholesky storage and the
inverse alive as long as either wrapper is reachable.

With `FastCholesky` loaded, `fastcholesky`, `cholinv`, `chollogdet` and `cholinv_logdet` reuse the
same caches.

# Reactant

With `Reactant` loaded (tested with Reactant 0.2.298 on Julia 1.13, CPU backend, `Float32` and
`Float64`), a wrapper converts with `Reactant.to_rarray` in any cache state, including a
populated reciprocal pair, and can be constructed inside a compiled function. Inside compiled
code `inv`, `\\`, `logdet`, `*` and broadcasting are supported with the following contract:

- Caches are never written during tracing. Host-populated caches become inputs of the compiled
  program (no factorization is emitted for them); empty caches are computed from the storage in
  the program on every execution.
- The cache state is part of the compile contract: a function compiled on an empty-cache wrapper
  also accepts populated wrappers, but a function compiled on a populated wrapper fails on an
  empty one.
- Julia object identity and the host lock do not carry across the compiled boundary: a compiled
  `inv` returns a new wrapper whose `inverse` holds the input values but is not `===` the input.
- Symmetry and positive definiteness are not checked on traced input.
"""
mutable struct MatrixWithCachedInverse{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    const storage::ReadOnlyMatrix{T,M}
    const guard::ReentrantLock
    factor::Union{Nothing,Cholesky{T,M}}
    inverse::Union{Nothing,MatrixWithCachedInverse{T,M}}

    # Internal: wraps an already owned matrix without copying it, with empty caches.
    function MatrixWithCachedInverse{T,M}(
        m::M, guard::ReentrantLock
    ) where {T,M<:AbstractMatrix{T}}
        return new{T,M}(ReadOnlyMatrix(m), guard, nothing, nothing)
    end
end

function MatrixWithCachedInverse(m::AbstractMatrix{T}) where {T}
    LinearAlgebra.checksquare(m)
    _checksymmetric(m)
    c = copy(m)
    return MatrixWithCachedInverse{T,typeof(c)}(c, ReentrantLock())
end

function _checksymmetric(m)
    return issymmetric(m) ||
           throw(ArgumentError("MatrixWithCachedInverse requires a symmetric matrix"))
end

Base.parent(A::MatrixWithCachedInverse) = parent(A.storage)
Base.size(A::MatrixWithCachedInverse) = size(A.storage)
Base.@propagate_inbounds Base.getindex(A::MatrixWithCachedInverse, i::Int, j::Int) =
    A.storage[i, j]
Base.Matrix(A::MatrixWithCachedInverse) = Matrix(parent(A))
Base.copy(A::MatrixWithCachedInverse) = copy(parent(A))
Base.:*(A::MatrixWithCachedInverse, x::AbstractVector) = parent(A) * x
Base.:*(A::MatrixWithCachedInverse, X::AbstractMatrix) = parent(A) * X
Base.broadcastable(A::MatrixWithCachedInverse) = parent(A) # `A + B`, `2 .* A`, ... use the parent

# An independent snapshot with empty caches (same-type `convert` returns `A` itself).
function Base.AbstractMatrix{T}(A::MatrixWithCachedInverse) where {T}
    c = AbstractMatrix{T}(parent(A))
    return MatrixWithCachedInverse{T,typeof(c)}(c, ReentrantLock())
end

# The cached Cholesky factor, computed by `factorize(parent(A))` on the first request. The
# factorization never mutates `storage`, and only a successful result is published.
function _factor(factorize::F, A::MatrixWithCachedInverse{T,M}) where {F,T,M}
    Base.@lock A.guard begin
        C = A.factor
        if C === nothing
            C = factorize(parent(A))::Cholesky{T,M}
            A.factor = C
        end
        C
    end
end
_factor(A::MatrixWithCachedInverse) = _factor(m -> cholesky(Symmetric(m)), A)

# The cached inverse wrapper, built from the dense `invert(A)` on the first request.
function _inverse(invert::F, A::MatrixWithCachedInverse{T,M}) where {F,T,M}
    Base.@lock A.guard begin
        B = A.inverse
        if B === nothing
            B = MatrixWithCachedInverse{T,M}(invert(A)::M, A.guard)
            B.inverse = A # the reverse link is set before `B` is published
            A.inverse = B
        end
        B
    end
end
Base.inv(A::MatrixWithCachedInverse) = _inverse(A -> inv(_factor(A)), A)

Base.:\(A::MatrixWithCachedInverse, b::AbstractVecOrMat) = _factor(A) \ b
LinearAlgebra.logdet(A::MatrixWithCachedInverse) = logdet(_factor(A))
