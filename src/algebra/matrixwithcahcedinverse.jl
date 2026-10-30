export MatrixWithCachedInverse

using LinearAlgebra

"""
    MatrixWithCachedInverse(A::AbstractMatrix)

An `AbstractMatrix` wrapper around an owned copy of a dense, real, symmetric positive-definite
matrix that lazily caches its Cholesky factor and its inverse.

Construction copies `A` (a `Symmetric` or `Hermitian` view is materialized into a dense matrix),
checks that it is square and symmetric, and performs no factorization.
The first `cholesky(A)`, `isposdef(A)`, `A \\ b` or `logdet(A)` computes and caches the Cholesky
factor. The first `inv(A)` computes the inverse once and returns another `MatrixWithCachedInverse`
that already links back,
so `inv(A) === inv(A)` and `inv(inv(A)) === A` hold without any recomputation. Positive
definiteness is checked on the first factorization request (`PosDefException`, or `false` from
`isposdef`). Symmetry and, for real elements, Hermitian symmetry are known without scanning.

Element writes through the wrapper and its inverse are rejected; `Matrix(A)` returns an
independent writable dense copy. `copy(A)` preserves the wrapper with independent storage and a
fresh lock, reuses any cached Cholesky factor, and starts with an empty inverse cache.
`AbstractMatrix{T}(A)` returns a new wrapper with empty caches,
while `convert(AbstractMatrix{T}, A)` returns `A` itself when the element type already matches.
Lazy initialization is synchronized with a lock shared by the pair, so concurrent first requests
share one result.

`parent(A)` exposes the owned mutable matrix without copying. Do not mutate it while using the
wrapper: mutations can leave cached factors and inverses stale and break symmetry assumptions.
There is no automatic cache invalidation. The `.storage` field is internal, not public API.
Use `Matrix(A)` when an independent writable matrix is needed.

# FastCholesky

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
    const storage::M
    const guard::ReentrantLock
    factor::Union{Nothing,Cholesky{T,M}}
    inverse::Union{Nothing,MatrixWithCachedInverse{T,M}}

    # Internal: wraps an already owned matrix without copying it, with empty caches.
    function MatrixWithCachedInverse{T,M}(
        m::M, guard::ReentrantLock
    ) where {T,M<:AbstractMatrix{T}}
        return new{T,M}(m, guard, nothing, nothing)
    end
end

function MatrixWithCachedInverse(m::AbstractMatrix{T}) where {T}
    LinearAlgebra.checksquare(m)
    _checksymmetric(m)
    c = _materialize(m)
    return MatrixWithCachedInverse{T,typeof(c)}(c, ReentrantLock())
end

# The owned storage: a plain copy, or both triangles of a `Symmetric`/`Hermitian` view
# written into a dense matrix of the parent's type.
_materialize(m) = copy(m)
_materialize(m::MatrixWithCachedInverse) = copy(parent(m))
_materialize(m::Union{Symmetric,Hermitian}) = copyto!(similar(parent(m)), m)

function _checksymmetric(m)
    return issymmetric(m) ||
           throw(ArgumentError("MatrixWithCachedInverse requires a symmetric matrix"))
end

Base.parent(A::MatrixWithCachedInverse) = A.storage
Base.size(A::MatrixWithCachedInverse) = size(A.storage)
Base.@propagate_inbounds Base.getindex(A::MatrixWithCachedInverse, i::Int, j::Int) =
    A.storage[i, j]
Base.Matrix(A::MatrixWithCachedInverse) = Matrix(parent(A))
function Base.copy(A::MatrixWithCachedInverse{T,M}) where {T,M}
    Base.@lock A.guard begin
        B = MatrixWithCachedInverse{T,M}(copy(parent(A)), ReentrantLock())
        B.factor = A.factor # The cached factor is never mutated by the wrapper.
        # The inverse must be recomputed so its reciprocal link points to B, not A.
        return B
    end
end
Base.transpose(A::MatrixWithCachedInverse) = A
Base.adjoint(A::MatrixWithCachedInverse{<:Real}) = A
Base.:*(A::MatrixWithCachedInverse, x::AbstractVector) = parent(A) * x
Base.:*(A::MatrixWithCachedInverse, X::AbstractMatrix) = parent(A) * X
Base.broadcastable(A::MatrixWithCachedInverse) = parent(A) # `A + B`, `2 .* A`, ... use the parent

# Keep products on the parent's optimized multiplication paths in either operand position.
function LinearAlgebra.mul!(
    y::AbstractVector, A::MatrixWithCachedInverse, x::AbstractVector
)
    return mul!(y, parent(A), x)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::MatrixWithCachedInverse, B::AbstractMatrix
)
    return mul!(C, parent(A), B)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::AbstractMatrix, B::MatrixWithCachedInverse
)
    return mul!(C, A, parent(B))
end
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::MatrixWithCachedInverse, B::MatrixWithCachedInverse
)
    return mul!(C, parent(A), parent(B))
end

function LinearAlgebra.mul!(
    y::AbstractVector, A::MatrixWithCachedInverse, x::AbstractVector, α::Number, β::Number
)
    return mul!(y, parent(A), x, α, β)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::MatrixWithCachedInverse, B::AbstractMatrix, α::Number, β::Number
)
    return mul!(C, parent(A), B, α, β)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix, A::AbstractMatrix, B::MatrixWithCachedInverse, α::Number, β::Number
)
    return mul!(C, A, parent(B), α, β)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::MatrixWithCachedInverse,
    B::MatrixWithCachedInverse,
    α::Number,
    β::Number,
)
    return mul!(C, parent(A), parent(B), α, β)
end

# An independent snapshot with empty caches (same-type `convert` returns `A` itself).
function Base.AbstractMatrix{T}(A::MatrixWithCachedInverse) where {T}
    c = AbstractMatrix{T}(parent(A))
    return MatrixWithCachedInverse{T,typeof(c)}(c, ReentrantLock())
end

# Returns `A.factor` when it is already set; otherwise computes it as `factorize(parent(A))`,
# stores it in `A.factor` and returns it. The factorization never mutates `storage`, only a
# successful result is published, and the whole step runs under `A.guard` so concurrent first
# requests share one result.
function cached_cholesky_or_compute(
    factorize::F, A::MatrixWithCachedInverse{T,M}
) where {F,T,M}
    Base.@lock A.guard begin
        C = A.factor
        if C === nothing
            C = factorize(parent(A))::Cholesky{T,M}
            A.factor = C
        end
        C
    end
end
function cached_cholesky_or_compute(A::MatrixWithCachedInverse)
    return cached_cholesky_or_compute(m -> cholesky(Symmetric(m)), A)
end

# Returns `A.inverse` when it is already set; otherwise computes the dense `invert(A)`, wraps it
# in a new `MatrixWithCachedInverse` sharing `A.guard`, links the pair both ways and returns the
# wrapper. The reverse link is set before the wrapper is published, under `A.guard`.
function cached_inverse_or_compute(invert::F, A::MatrixWithCachedInverse{T,M}) where {F,T,M}
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
function Base.inv(A::MatrixWithCachedInverse)
    return cached_inverse_or_compute(A -> inv(cached_cholesky_or_compute(A)), A)
end

Base.:\(A::MatrixWithCachedInverse, b::AbstractVecOrMat) = cached_cholesky_or_compute(A) \ b
LinearAlgebra.logdet(A::MatrixWithCachedInverse) = logdet(cached_cholesky_or_compute(A))
LinearAlgebra.cholesky(A::MatrixWithCachedInverse) = cached_cholesky_or_compute(A)
LinearAlgebra.issymmetric(::MatrixWithCachedInverse) = true
LinearAlgebra.ishermitian(::MatrixWithCachedInverse{<:Real}) = true

function LinearAlgebra.isposdef(A::MatrixWithCachedInverse)
    try
        return isposdef(cached_cholesky_or_compute(A))
    catch err
        err isa PosDefException || rethrow()
        return false
    end
end
