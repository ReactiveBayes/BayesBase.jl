export ArrowheadMatrix, InvArrowheadMatrix


import LinearAlgebra: SingularException
import Base: getindex
import LinearAlgebra: mul!
import Base: size, *, \, inv, convert, Matrix
"""
    ArrowheadMatrix{O, T, Z, P} <: AbstractMatrix{O}

A structure representing an arrowhead matrix, which is a special type of sparse matrix.

# Fields
- `α::T`: The scalar value at the bottom-right corner of the matrix.
- `z::Z`: A vector representing the last row/column (excluding the corner element).
- `D::P`: A vector representing the diagonal elements (excluding the corner element).

# Constructors
    ArrowheadMatrix(a::T, z::Z, d::D) where {T,Z,D}

Constructs an `ArrowheadMatrix` with the given α, z, and D values. The output type `O`
is automatically determined as the promoted type of all input elements.

# Operations
- Matrix-vector multiplication: `A * x` or `mul!(y, A, x)`
- Linear system solving: `A \\ b` or `ldiv!(x, A, b)`
- Conversion to dense matrix: `convert(Matrix, A)`
- Inversion: `inv(A)` (returns an `InvArrowheadMatrix`)

# Examples
```julia
α = 2.0
z = [1.0, 2.0, 3.0]
D = [4.0, 5.0, 6.0]
A = ArrowheadMatrix(α, z, D)

# Matrix-vector multiplication
x = [1.0, 2.0, 3.0, 4.0]
y = A * x

# Solving linear system
b = [7.0, 8.0, 9.0, 10.0]
x = A \\ b

# Convert to dense matrix
dense_A = convert(Matrix, A)
```

# Notes
- The matrix is singular if α - dot(z ./ D, z) = 0 or if any element of D is zero.
- For best performance, use `ldiv!` for solving linear systems when possible.
"""
struct ArrowheadMatrix{O, T, Z, P} <: AbstractMatrix{O}
    α::T         
    z::Z         
    D::P
end
function ArrowheadMatrix(a::T, z::Z, d::D) where {T,Z,D}
    O = promote_type(typeof(a), eltype(z), eltype(d))
    return ArrowheadMatrix{O, T, Z, D}(a, z, d)
end

function Base.getindex(A::ArrowheadMatrix, i::Int, j::Int)

    @warn "getindex was called on ArrowheadMatrix. This may lead to suboptimal performance. Consider using specialized methods if available." maxlog=1

    n = length(A.D) + 1
    if i < 1 || i > n || j < 1 || j > n
        throw(BoundsError(A, (i, j)))
    end
    
    if i == n && j == n
        return A.α
    elseif i == n
        return A.z[j]
    elseif j == n
        return A.z[i]
    elseif i == j
        return A.D[i]
    else
        return zero(eltype(A))
    end
end

function show(io::IO, ::MIME"text/plain", A::ArrowheadMatrix)
    n = length(A.D) + 1
    println(io, n, "×", n, " ArrowheadMatrix{", eltype(A), "}:")
    
    for i in 1:n-1
        for j in 1:n-1
            if i == j
                print(io, A.D[i])
            else
                print(io, "⋅")
            end
            print(io, "  ")
        end
        println(io, A.z[i])
    end
    
    # Print the last row
    for i in 1:n-1
        print(io, A.z[i], "  ")
    end
    println(io, A.α)
end

function size(A::ArrowheadMatrix)
    n = length(A.D) + 1
    return (n, n)
end

function Base.convert(::Type{Matrix}, A::ArrowheadMatrix{O}) where {O}
    n = length(A.z)
    M = zeros(O, n + 1, n + 1)
    for i in 1:n
        M[i, i] = A.D[i]
    end
    M[1:n, n + 1] .= A.z
    M[n + 1, 1:n] .= A.z
    M[n + 1, n + 1] = A.α
    return M
end

function LinearAlgebra.mul!(y, A::ArrowheadMatrix{T}, x::AbstractVector{T}) where T
    n = length(A.z)
    if length(x) != n + 1
        throw(DimensionMismatch())
    end
    @inbounds @views begin 
        y[1:n] = A.D .* x[1:n] + A.z * x[n + 1]
        y[n + 1] = dot(A.z, x[1:n]) + A.α * x[n + 1]
    end
    return y
end

function linsolve!(y::AbstractVector{T2}, A::ArrowheadMatrix{T}, b::AbstractVector{T2}) where {T, T2}
    n = length(A.z)

    if length(b) != n + 1
        throw(DimensionMismatch())
    end

    z = A.z
    D = A.D
    α = A.α

    # Check for zeros in D to avoid division by zero
    @inbounds for i in 1:n
        if D[i] == 0
            throw(SingularException(1))
        end
    end

    s = zero(T)
    t = zero(T)

    # Compute s and t in a single loop to avoid recomputing z[i] / D[i]
    @inbounds @simd for i in 1:n
        zi = z[i]
        Di = D[i]
        z_div_D = zi / Di
        bi = b[i]

        s += z_div_D * bi       # Accumulate s
        t += z_div_D * zi       # Accumulate t
    end

    denom = α - t
    if denom == 0
        throw(SingularException(1))
    end

    yn1 = (b[n + 1] - s) / denom
    y[n + 1] = yn1

    # Compute y[1:n]
    @inbounds @simd for i in 1:n
        y[i] = (b[i] - z[i] * yn1) / D[i]
    end

    return y
end

function Base.:\(A::ArrowheadMatrix, b::AbstractVector{T}) where T
    y = similar(b)
    return linsolve!(y, A, b)
end

function LinearAlgebra.ldiv!(x::AbstractVector{T}, A::ArrowheadMatrix, b::AbstractVector{T}) where T
    return linsolve!(x, A, b)
end

"""
    InvArrowheadMatrix{O, T, Z, P} <: AbstractMatrix{O}

A wrapper structure representing the inverse of an `ArrowheadMatrix`.

This structure doesn't explicitly compute or store the inverse matrix.
Instead, it stores a reference to the original `ArrowheadMatrix` and
implements efficient operations that leverage the special structure
of the arrowhead matrix.

# Fields
- `A::ArrowheadMatrix{O, T, Z, P}`: The original `ArrowheadMatrix` being inverted.

# Constructors
    InvArrowheadMatrix(A::ArrowheadMatrix{O, T, Z, P})

Constructs an `InvArrowheadMatrix` by wrapping the given `ArrowheadMatrix`.

# Operations
- Matrix-vector multiplication: `A_inv * x` or `mul!(y, A_inv, x)`
  (Equivalent to solving the system A * y = x)
- Linear system solving: `A_inv \\ x`
  (Equivalent to multiplication by the original matrix: A * x)
- Conversion to dense matrix: `convert(Matrix, A_inv)`
  (Computes and returns the actual inverse as a dense matrix)

# Examples
```julia
α = 2.0
z = [1.0, 2.0, 3.0]
D = [4.0, 5.0, 6.0]
A = ArrowheadMatrix(α, z, D)
A_inv = inv(A)  # Returns an InvArrowheadMatrix

# Multiplication (equivalent to solving A * y = x)
x = [1.0, 2.0, 3.0, 4.0]
y = A_inv * x

# Division (equivalent to multiplying by A)
b = [5.0, 6.0, 7.0, 8.0]
x = A_inv \\ b

# Convert to dense inverse matrix
dense_inv_A = convert(Matrix, A_inv)
```

# Notes
- The inverse exists only if the original `ArrowheadMatrix` is non-singular.
- Operations with `InvArrowheadMatrix` do not explicitly compute the inverse,
  but instead solve the corresponding system with the original matrix.

# See Also
- [`ArrowheadMatrix`](@ref): The original arrowhead matrix structure.
"""
struct InvArrowheadMatrix{O, T, Z, P} <: AbstractMatrix{O}
    A::ArrowheadMatrix{O, T, Z, P}
end

function show(io::IO, ::MIME"text/plain", A_inv::InvArrowheadMatrix)
    n = size(A_inv.A, 1)
    println(io, n, "×", n, " InvArrowheadMatrix{", eltype(A_inv), "}:")
    println(io, "Inverse of:")
    show(io, MIME"text/plain"(), A_inv.A)
end


inv(A::ArrowheadMatrix) = InvArrowheadMatrix(A)

function size(A_inv::InvArrowheadMatrix)
    size(A_inv.A)
end

function LinearAlgebra.mul!(y, A_inv::InvArrowheadMatrix{T}, x::AbstractVector{T}) where T
    A = A_inv.A
    return linsolve!(y, A, x)
end

function Base.:\(A_inv::InvArrowheadMatrix{T}, x::AbstractVector{T}) where T
    A = A_inv.A
    return A * x
end

function Base.convert(::Type{Matrix}, A_inv::InvArrowheadMatrix{T}) where T
    A = A_inv.A
    n = length(A.z)
    z = A.z
    D = A.D
    α = A.α

    # Compute t = dot(z ./ D, z)
    t = dot(z ./ D, z)
    denom = α - t
    @assert denom != 0 "Matrix is singular."

    # Compute u = [ (z ./ D); -1 ]
    u = [ z ./ D; -1.0 ]

    # Compute the inverse diagonal elements
    D_inv = 1.0 ./ D

    # Construct the inverse matrix
    M = zeros(T, n + 1, n + 1)
    M[1:n, 1:n] .= Diagonal(D_inv)
    M .+= (u * u') / denom
    return M
end

function LinearAlgebra.dot(x::AbstractVector, A_inv::InvArrowheadMatrix, y::AbstractVector)
    A = A_inv.A
    n = length(A.z)
    
    if length(x) != n + 1 || length(y) != n + 1
        throw(DimensionMismatch("Dimensions must match"))
    end

    # Compute A_inv * y using linsolve!
    temp = similar(y)
    linsolve!(temp, A, y)

    # Compute the dot product of x and temp
    return LinearAlgebra.dot(x, temp)
end

function Base.isapprox(A::InvArrowheadMatrix, B::InvArrowheadMatrix; 
    rtol::Real=sqrt(eps()), atol::Real=0, 
    nans::Bool=false, norm::Function=LinearAlgebra.norm)
    return isapprox(A.A, B.A; rtol=rtol, atol=atol, nans=nans, norm=norm)
end

