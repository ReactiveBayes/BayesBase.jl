@testitem "MatrixWithCachedInverse: construction copies the input and stays lazy" begin
    include("algebrasetup_setuptests.jl")

    for T in (Float32, Float64, BigFloat), n in (1, 5)
        S = T[1 / (i + j - 1) for i in 1:n, j in 1:n] + I
        source = copy(S)
        A = MatrixWithCachedInverse(source)
        source[1, 1] += 1
        @test A isa MatrixWithCachedInverse{T,Matrix{T}}
        @test Matrix(A) == S
        @test size(A) == (n, n)
        @test A[1, 1] == S[1, 1] && A[n] == S[n]
        @test A * ones(T, n) ≈ S * ones(T, n)
        @test A * S ≈ S * S
        @test A.factor === nothing && A.inverse === nothing

        @test A \ ones(T, n) ≈ S \ ones(T, n)
        @test A.factor isa Cholesky{T,Matrix{T}} && A.inverse === nothing

        B = MatrixWithCachedInverse(S)
        @test logdet(B) ≈ logdet(S)
        @test B.factor isa Cholesky{T,Matrix{T}} && B.inverse === nothing
    end
end

@testitem "MatrixWithCachedInverse: products use symmetric dense storage" begin
    include("algebrasetup_setuptests.jl")

    for T in (Float32, Float64, BigFloat), n in (1, 5)
        S = T[1 / (i + j - 1) for i in 1:n, j in 1:n] + I
        A = MatrixWithCachedInverse(S)
        x = T.(1:n)
        X = reshape(T.(1:(2n)), n, 2)

        # Exercise both fresh wrappers and the wrapper returned by inv.
        for B in (A, inv(A))
            D = Matrix(B)
            @test transpose(B) === B
            @test adjoint(B) === B
            @test B * x ≈ D * x
            @test B * X ≈ D * X
            @test X' * B ≈ X' * D
            @test B * B ≈ D * D
            @test A * B ≈ S * D
            @test x' * B ≈ x' * D
            @test x' * B * x ≈ x' * D * x
            @test B' * x ≈ D' * x
            @test transpose(x) * B ≈ transpose(x) * D
            @test transpose(B) * x ≈ transpose(D) * x

            for (left, right, dense_left, dense_right) in (
                (B, x, D, x),
                (B, X, D, X),
                (X', B, X', D),
                (copy(X'), B, copy(X'), D),
                (B, B, D, D),
                (A, B, S, D),
            )
                expected = dense_left * dense_right
                result = fill(T(NaN), size(expected))
                @test mul!(result, left, right) === result
                @test result ≈ expected

                for (α, β) in ((T(2), T(-3)), (T(0), T(2)), (T(1), T(0)))
                    fill!(result, iszero(β) ? T(NaN) : T(2))
                    reference = copy(result)
                    mul!(reference, dense_left, dense_right, α, β)
                    @test mul!(result, left, right, α, β) === result
                    @test result ≈ reference
                end
            end
        end
    end
end

@testitem "MatrixWithCachedInverse: products avoid scalar indexing" begin
    include("algebrasetup_setuptests.jl")

    # A parent with optimized multiplication but no scalar access: falling back to
    # generic multiplication through the wrapper must fail, regardless of timing.
    struct MulOnlyMatrix{T} <: AbstractMatrix{T}
        data::Matrix{T}
    end
    Base.size(A::MulOnlyMatrix) = size(A.data)
    Base.getindex(::MulOnlyMatrix, ::Int, ::Int) = error("unexpected scalar indexing")
    LinearAlgebra.mul!(
        y::AbstractVector, A::MulOnlyMatrix, x::AbstractVector, α::Number, β::Number
    ) = mul!(y, A.data, x, α, β)
    LinearAlgebra.mul!(
        C::AbstractMatrix, A::MulOnlyMatrix, B::Matrix, α::Number, β::Number
    ) = mul!(C, A.data, B, α, β)
    LinearAlgebra.mul!(
        C::AbstractMatrix, A::Matrix, B::MulOnlyMatrix, α::Number, β::Number
    ) = mul!(C, A, B.data, α, β)
    LinearAlgebra.mul!(
        C::AbstractMatrix, A::MulOnlyMatrix, B::MulOnlyMatrix, α::Number, β::Number
    ) = mul!(C, A.data, B.data, α, β)

    S = [2.0 1.0; 1.0 3.0]
    # Use the internal constructor to wrap the instrumented storage without reading it.
    A = MatrixWithCachedInverse{Float64,MulOnlyMatrix{Float64}}(
        MulOnlyMatrix(S), ReentrantLock()
    )
    x = [1.0, 2.0]
    @test S * A ≈ S * S
    @test A * A ≈ S * S
    @test x' * A * x ≈ x' * S * x
    @test A' * x ≈ S' * x
    for (left, right, expected) in
        ((A, x, S * x), (A, S, S * S), (S, A, S * S), (A, A, S * S))
        result = zero(expected)
        @test mul!(result, left, right) ≈ expected
        fill!(result, 1)
        @test mul!(result, left, right, 2, 3) ≈ 2expected .+ 3
    end
end

@testitem "MatrixWithCachedInverse: construction from Symmetric and Hermitian views" begin
    include("algebrasetup_setuptests.jl")

    for T in (Float32, Float64, BigFloat), uplo in (:U, :L)
        n = 5
        S = T[1 / (i + j - 1) for i in 1:n, j in 1:n] + I
        junk = copy(S)
        junk[CartesianIndex(uplo === :U ? (2, 1) : (1, 2))] = T(100) # the ignored triangle holds garbage
        for V in (Symmetric(junk, uplo), Hermitian(junk, uplo))
            A = @inferred MatrixWithCachedInverse(V)
            @test A isa MatrixWithCachedInverse{T,Matrix{T}}
            @test Matrix(A) == S
            @test A.factor === nothing && A.inverse === nothing
            @test inv(A) ≈ inv(S)
            @test inv(inv(A)) === A
            @test A \ ones(T, n) ≈ S \ ones(T, n)
            @test logdet(A) ≈ logdet(S)
        end
        @test MatrixWithCachedInverse(Symmetric(MatrixWithCachedInverse(S))) == S
    end
end

@testitem "MatrixWithCachedInverse: inverse, solve and logdet" begin
    include("algebrasetup_setuptests.jl")

    for T in (Float32, Float64, BigFloat), n in (2, 10)
        S = T[1 / (i + j - 1) for i in 1:n, j in 1:n] + I
        F = cholesky(Symmetric(S))
        b = T.(1:n)
        A = MatrixWithCachedInverse(S)

        B = inv(A)
        @test B isa typeof(A)
        @test Matrix(B) ≈ inv(F)
        @test inv(A) === B
        @test inv(B) === A
        @test B.factor === nothing
        @test A.factor isa Cholesky{T,Matrix{T}}

        @test A \ b ≈ F \ b
        @test A \ Matrix{T}(I, n, n) ≈ inv(F)
        @test logdet(A) ≈ logdet(F)
        @test logdet(B) ≈ -logdet(F)
        @test B.factor isa Cholesky{T,Matrix{T}}
        @test B \ b ≈ S * b
    end
end

@testitem "MatrixWithCachedInverse: ownership and errors" begin
    include("algebrasetup_setuptests.jl")

    S = [2.0 1.0; 1.0 2.0]
    A = MatrixWithCachedInverse(S)
    B = inv(A)
    @test_throws CanonicalIndexError A[1, 1] = 3.0
    @test_throws CanonicalIndexError A[1] = 3.0
    @test_throws CanonicalIndexError B[1, 1] = 3.0

    C, M, Cinv = copy(A), Matrix(A), copy(B)
    @test C isa Matrix{Float64} && M isa Matrix{Float64} && Cinv isa Matrix{Float64}
    C[1, 1] = 0.0
    M[1, 1] = 0.0
    Cinv[1, 1] = 0.0
    @test A[1, 1] == 2.0 && B[1, 1] ≈ inv(S)[1, 1]

    @test_throws DimensionMismatch MatrixWithCachedInverse(ones(2, 3))
    @test_throws ArgumentError MatrixWithCachedInverse([1.0 2.0; 3.0 4.0])

    N = MatrixWithCachedInverse([1.0 2.0; 2.0 1.0]) # symmetric but not positive definite
    @test N * [1.0, 1.0] == [3.0, 3.0]
    @test_throws PosDefException inv(N)
    @test_throws PosDefException N \ [1.0, 1.0]
    @test_throws PosDefException logdet(N)
    @test N.factor === nothing && N.inverse === nothing
end

@testitem "MatrixWithCachedInverse: inference and conversions" begin
    include("algebrasetup_setuptests.jl")

    for T in (Float32, Float64, BigFloat)
        S = T[2 1; 1 2]
        A = @inferred MatrixWithCachedInverse(S)
        @test (@inferred BayesBase.cached_cholesky_or_compute(A)) isa Cholesky{T,Matrix{T}}
        @test (@inferred BayesBase.cached_cholesky_or_compute(A)) === A.factor
        B = @inferred inv(A)
        @test (@inferred inv(A)) === B
        @test (@inferred inv(B)) === A

        @test convert(AbstractMatrix{T}, A) === A
        @test convert(AbstractArray{T}, A) === A
        C = AbstractMatrix{T}(A)
        @test C isa MatrixWithCachedInverse{T,Matrix{T}}
        @test C !== A && C == A && C.factor === nothing && C.inverse === nothing
        D = convert(AbstractMatrix{Float64}, A)
        @test D isa MatrixWithCachedInverse{Float64,Matrix{Float64}} && D == S
        @test Matrix{T}(A) isa Matrix{T}
    end
end

@testitem "MatrixWithCachedInverse: FastCholesky integration" begin
    include("algebrasetup_setuptests.jl")
    using FastCholesky

    S = [1.0 / (i + j - 1) for i in 1:5, j in 1:5] + I
    reference = inv(cholesky(Symmetric(S)))

    # cold `cholinv` behaves like `inv`: it fills the factor cache and the shared inverse pair
    A = MatrixWithCachedInverse(S)
    B = cholinv(A)
    @test B isa typeof(A) && Matrix(B) ≈ reference
    @test A.factor isa Cholesky # the factor is kept for later solves and log-determinants
    @test B === inv(A) && inv(B) === A && cholinv(B) === A

    # cold `fastcholesky` and `chollogdet` fill only the factor cache
    A = MatrixWithCachedInverse(S)
    @test fastcholesky(A) === BayesBase.cached_cholesky_or_compute(A) === A.factor
    @test A.inverse === nothing
    A = MatrixWithCachedInverse(S)
    @test chollogdet(A) ≈ logdet(S)
    @test A.factor isa Cholesky && A.inverse === nothing
    @test cholinv(A) === inv(A) # warm factor is reused for the inverse

    # cold `cholinv_logdet` fills both caches with one factorization
    A = MatrixWithCachedInverse(S)
    Ainv, l = cholinv_logdet(A)
    @test Ainv === inv(A) && Matrix(Ainv) ≈ reference && l ≈ logdet(S)
    @test A.factor isa Cholesky
    @test cholinv_logdet(A) == (Ainv, l)
end

@testitem "MatrixWithCachedInverse: concurrent first inverse requests share one result" begin
    include("algebrasetup_setuptests.jl")

    S = [1.0 / (i + j - 1) for i in 1:100, j in 1:100] + I
    reference = inv(cholesky(Symmetric(S)))
    inv(MatrixWithCachedInverse(S)) # compile on a disposable instance, keeping the tested cache fresh
    cached = MatrixWithCachedInverse(S)

    ready = Channel{Nothing}(4)
    start = Base.Event()
    tasks = map(1:4) do _
        Threads.@spawn begin
            put!(ready, nothing)
            wait(start)
            inverse = inv(cached)
            (inverse, inv(inverse))
        end
    end
    foreach(_ -> take!(ready), 1:4)
    notify(start)
    pairs = fetch.(tasks)
    results = first.(pairs)

    @test all(result -> result ≈ reference, results)
    @test all(result -> result === first(results), results)
    @test all(pair -> last(pair) === cached, pairs)
    @test inv(cached) === first(results)
    @test Matrix(cached) == S
end

@testitem "MatrixWithCachedInverse: warm and reverse inverse requests do not allocate matrices" begin
    include("algebrasetup_setuptests.jl")

    for n in (20, 100)
        S = [1.0 / (i + j - 1) for i in 1:n, j in 1:n] + I
        cached = MatrixWithCachedInverse(S)
        inverse = @inferred inv(cached)
        @test Matrix(inverse) ≈ inv(cholesky(Symmetric(S)))
        inv(cached), inv(inverse) # warm up
        @test (@allocated inv(cached)) <= 256
        @test (@allocated inv(inverse)) <= 256
    end
end
