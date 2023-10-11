@testitem "Real-based PointMass" begin
    using SpecialFunctions: loggamma
    using TinyHugeNumbers

    for T in (Float16, Float32, Float64, BigFloat)
        scalar = rand(T)
        dist = PointMass(scalar)

        @test variate_form(dist) === Univariate
        @test_throws BoundsError dist[2]
        @test_throws BoundsError dist[2, 2]

        @test insupport(dist, scalar)
        @test !insupport(dist, scalar + tiny)
        @test !insupport(dist, scalar - tiny)

        @test @inferred(T, pdf(dist, scalar)) == one(T)
        @test @inferred(T, pdf(dist, scalar + tiny)) == zero(T)
        @test @inferred(T, pdf(dist, scalar - tiny)) == zero(T)

        @test @inferred(T, logpdf(dist, scalar)) == zero(T)
        @test @inferred(T, logpdf(dist, scalar + tiny)) == convert(T, -Inf)
        @test @inferred(T, logpdf(dist, scalar - tiny)) == convert(T, -Inf)

        @test_throws MethodError insupport(dist, ones(T, 2))
        @test_throws MethodError insupport(dist, ones(T, 2, 2))
        @test_throws MethodError pdf(dist, ones(T, 2))
        @test_throws MethodError pdf(dist, ones(T, 2, 2))
        @test_throws MethodError logpdf(dist, ones(T, 2))
        @test_throws MethodError logpdf(dist, ones(T, 2, 2))

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity()

        @test @inferred(T, mean(dist)) == scalar
        @test @inferred(T, mode(dist)) == scalar
        @test @inferred(T, var(dist)) == zero(T)
        @test @inferred(T, std(dist)) == zero(T)
        @test @inferred(T, cov(dist)) == zero(T)
        @test @inferred(T, precision(dist)) == convert(T, Inf)
        @test @inferred(Int, ndims(dist)) == 1
        @test @inferred(Type{T}, eltype(dist)) == T

        @test_throws ErrorException probvec(dist)
        @test @inferred(T, mean(log, dist)) == log(scalar)
        @test @inferred(T, mean(inv, dist)) == inv(scalar)
        @test @inferred(T, mean(mirrorlog, dist)) == log(one(scalar) - scalar)
        @test @inferred(T, mean(loggamma, dist)) == loggamma(scalar)
    end
end

@testitem "Vector-based PointMass" begin
    using SpecialFunctions: loggamma
    using TinyHugeNumbers

    for T in (Float16, Float32, Float64, BigFloat), N in (5, 10)
        vector = rand(T, N)
        dist = PointMass(vector)

        @test variate_form(dist) === Multivariate
        @test dist[2] === vector[2]
        @test dist[3] === vector[3]
        @test_throws BoundsError dist[N + 1]
        @test_throws BoundsError dist[N - 1, N - 1]

        @test insupport(dist, vector)
        @test !insupport(dist, vector .+ tiny)
        @test !insupport(dist, vector .- tiny)

        @test @inferred(T, pdf(dist, vector)) == one(T)
        @test @inferred(T, pdf(dist, vector .+ tiny)) == zero(T)
        @test @inferred(T, pdf(dist, vector .- tiny)) == zero(T)

        @test @inferred(T, logpdf(dist, vector)) == zero(T)
        @test @inferred(T, logpdf(dist, vector .+ tiny)) == convert(T, -Inf)
        @test @inferred(T, logpdf(dist, vector .- tiny)) == convert(T, -Inf)

        @test_throws MethodError insupport(dist, one(T))
        @test_throws MethodError insupport(dist, ones(T, 2, 2))
        @test_throws MethodError pdf(dist, one(T))
        @test_throws MethodError pdf(dist, ones(T, 2, 2))
        @test_throws MethodError logpdf(dist, one(T))
        @test_throws MethodError logpdf(dist, ones(T, 2, 2))

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity()

        @test @inferred(AbstractVector{T}, mean(dist)) == vector
        @test @inferred(AbstractVector{T}, mode(dist)) == vector
        @test @inferred(AbstractVector{T}, var(dist)) == zeros(T, N)
        @test @inferred(AbstractVector{T}, std(dist)) == zeros(T, N)
        @test @inferred(AbstractMatrix{T}, cov(dist)) == zeros(T, N, N)
        @test @inferred(AbstractMatrix{T}, precision(dist)) == fill(convert(T, Inf), (N, N))
        @test @inferred(Int, ndims(dist)) == N
        @test @inferred(Type{T}, eltype(dist)) == T

        @test @inferred(AbstractVector{T}, probvec(dist)) == vector
        @test @inferred(
            AbstractVector{T}, mean(Base.Broadcast.BroadcastFunction(log), dist)
        ) == log.(vector)
        @test @inferred(
            AbstractVector{T}, mean(Base.Broadcast.BroadcastFunction(loggamma), dist)
        ) == loggamma.(vector)
        @test_throws MethodError mean(log, dist)
        @test_throws MethodError mean(loggamma, dist)
        @test_throws MethodError mean(inv, dist)
        @test_throws MethodError mean(mirrorlog, dist)
    end
end

@testitem "Matrix-based PointMass" begin
    using SpecialFunctions: loggamma
    using TinyHugeNumbers

    for T in (Float16, Float32, Float64, BigFloat), N in (5, 10)
        matrix = rand(T, N, N)
        dist = PointMass(matrix)

        @test variate_form(dist) === Matrixvariate
        @test dist[2] === matrix[2]
        @test dist[3] === matrix[3]
        @test dist[3, 3] === matrix[3, 3]
        @test size(dist, 1) === size(matrix, 1)
        @test size(dist, 2) === size(matrix, 2)
        @test_throws BoundsError dist[N^3]
        @test_throws BoundsError dist[N + 1, N + 1]

        @test insupport(dist, matrix)
        @test !insupport(dist, matrix .+ tiny)
        @test !insupport(dist, matrix .- tiny)

        @test @inferred(T, pdf(dist, matrix)) == one(T)
        @test @inferred(T, pdf(dist, matrix .+ tiny)) == zero(T)
        @test @inferred(T, pdf(dist, matrix .- tiny)) == zero(T)

        @test @inferred(T, logpdf(dist, matrix)) == zero(T)
        @test @inferred(T, logpdf(dist, matrix .+ tiny)) == convert(T, -Inf)
        @test @inferred(T, logpdf(dist, matrix .- tiny)) == convert(T, -Inf)

        @test_throws MethodError insupport(dist, one(T))
        @test_throws MethodError insupport(dist, ones(T, 2))
        @test_throws MethodError pdf(dist, one(T))
        @test_throws MethodError pdf(dist, ones(T, 2))
        @test_throws MethodError logpdf(dist, one(T))
        @test_throws MethodError logpdf(dist, ones(T, 2))

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity()

        @test @inferred(AbstractMatrix{T}, mean(dist)) == matrix
        @test @inferred(AbstractMatrix{T}, mode(dist)) == matrix
        @test @inferred(AbstractMatrix{T}, var(dist)) == zeros(N, N)
        @test @inferred(AbstractMatrix{T}, std(dist)) == zeros(N, N)
        @test @inferred(Tuple{Int,Int}, ndims(dist)) == (N, N)
        @test @inferred(Type{T}, eltype(dist)) == T

        @test_throws ErrorException cov(dist)
        @test_throws ErrorException precision(dist)

        @test_throws ErrorException probvec(dist)
        @test @inferred(
            AbstractMatrix{T}, mean(Base.Broadcast.BroadcastFunction(log), dist)
        ) == log.(matrix)
        @test @inferred(
            AbstractMatrix{T}, mean(Base.Broadcast.BroadcastFunction(loggamma), dist)
        ) == loggamma.(matrix)
        @test @inferred(AbstractMatrix{T}, mean(inv, dist)) ≈ inv(matrix)

        # `BigFloat` does not support the log operation
        if T !== BigFloat
            @test mean(log, dist) ≈ log(matrix)
        end

        @test_throws MethodError mean(loggamma, dist)
    end
end

@testitem "UniformScaling-based PointMass" begin
    using LinearAlgebra, TinyHugeNumbers

    for T in (Float16, Float32, Float64, BigFloat)
        matrix = convert(T, 5) * I
        dist = PointMass(matrix)

        @test variate_form(dist) === Matrixvariate
        @test dist[2, 1] == zero(T)
        @test dist[3, 1] == zero(T)
        @test dist[3, 3] === matrix[3, 3]

        @test pdf(dist, matrix) == one(T)
        @test pdf(dist, matrix + convert(T, tiny) * I) == zero(T)
        @test pdf(dist, matrix - convert(T, tiny) * I) == zero(T)

        @test logpdf(dist, matrix) == zero(T)
        @test logpdf(dist, matrix + convert(T, tiny) * I) == convert(T, -Inf)
        @test logpdf(dist, matrix - convert(T, tiny) * I) == convert(T, -Inf)

        @test_throws MethodError insupport(dist, one(T))
        @test_throws MethodError insupport(dist, ones(T, 2))
        @test_throws MethodError pdf(dist, one(T))
        @test_throws MethodError pdf(dist, ones(T, 2))
        @test_throws MethodError logpdf(dist, one(T))
        @test_throws MethodError logpdf(dist, ones(T, 2))

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity()

        @test mean(dist) == matrix
        @test mode(dist) == matrix
        @test var(dist) == zero(T) * I
        @test std(dist) == zero(T) * I

        @test_throws ErrorException cov(dist)
        @test_throws ErrorException precision(dist)

        @test mean(inv, dist) ≈ inv(matrix)
    end
end