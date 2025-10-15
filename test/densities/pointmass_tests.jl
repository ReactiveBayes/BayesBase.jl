@testitem "Real-based PointMass" begin
    using SpecialFunctions: loggamma
    using TinyHugeNumbers

    for T in (Float16, Float32, Float64, BigFloat)
        scalar = rand(T)
        dist = PointMass(scalar)

        @test variate_form(typeof(dist)) === Univariate
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

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity(T)

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
    using BayesBase

    for T in (Float16, Float32, Float64, BigFloat), N in (5, 10)
        vector = rand(T, N)
        dist = PointMass(vector)

        @test variate_form(typeof(dist)) === Multivariate
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

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity(T)

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

        @test variate_form(typeof(dist)) === Matrixvariate
        @test dist[2] === matrix[2]
        @test dist[3] === matrix[3]
        @test dist[3, 3] === matrix[3, 3]
        @test size(dist, 1) === size(matrix, 1)
        @test size(dist, 2) === size(matrix, 2)
        @test_throws BoundsError dist[N ^ 3]
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

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity(T)

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

@testitem "Tensor-based PointMass" begin
    using SpecialFunctions: loggamma
    using TinyHugeNumbers
    using Distributions

    for D in [3, 4, 5]
        for T in (Float16, Float32, Float64, BigFloat), N in (5, 10)
            tensor = rand(T, ntuple(_ -> N, D))
            dist = PointMass(tensor)

            @test variate_form(typeof(dist)) === Distributions.ArrayLikeVariate{D}
            @test dist[2] === tensor[2]
            @test dist[3] === tensor[3]
            @test dist[ntuple(_ -> 3, D)...] === tensor[ntuple(_ -> 3, D)...]
            for i in 1:D
                @test size(dist, i) === size(tensor, i)
            end
            @test_throws BoundsError dist[N ^ (D + 1)]
            @test_throws BoundsError dist[ntuple(_ -> N + 1, D)...]

            @test insupport(dist, tensor)
            @test !insupport(dist, tensor .+ tiny)
            @test !insupport(dist, tensor .- tiny)

            @test @inferred(T, pdf(dist, tensor)) == one(T)
            @test @inferred(T, pdf(dist, tensor .+ tiny)) == zero(T)
            @test @inferred(T, pdf(dist, tensor .- tiny)) == zero(T)

            @test @inferred(T, logpdf(dist, tensor)) == zero(T)
            @test @inferred(T, logpdf(dist, tensor .+ tiny)) == convert(T, -Inf)
            @test @inferred(T, logpdf(dist, tensor .- tiny)) == convert(T, -Inf)

            for i in 1:(D - 1)
                @test_throws MethodError insupport(dist, ones(T, ntuple(_ -> 2, i)...))
                @test_throws MethodError pdf(dist, ones(T, ntuple(_ -> 2, i)...))
                @test_throws MethodError logpdf(dist, ones(T, ntuple(_ -> 2, i)...))
            end

            @test (@inferred entropy(dist)) == BayesBase.MinusInfinity(T)

            @test @inferred(AbstractArray{D,T}, mean(dist)) == tensor
            @test @inferred(AbstractArray{D,T}, mode(dist)) == tensor
            @test @inferred(AbstractArray{D,T}, var(dist)) == zeros(ntuple(_ -> N, D)...)
            @test @inferred(AbstractArray{D,T}, std(dist)) == zeros(ntuple(_ -> N, D)...)
            @test @inferred(Tuple{Int,Int,Int}, ndims(dist)) == ntuple(_ -> N, D)
            @test @inferred(Type{T}, eltype(dist)) == T

            @test_throws ErrorException cov(dist)
            @test_throws ErrorException precision(dist)

            @test_throws ErrorException probvec(dist)
            @test @inferred(
                AbstractArray{D,T}, mean(Base.Broadcast.BroadcastFunction(log), dist)
            ) == log.(tensor)
            @test @inferred(
                AbstractArray{D,T}, mean(Base.Broadcast.BroadcastFunction(loggamma), dist)
            ) == loggamma.(tensor)

            @test_throws MethodError mean(loggamma, dist)
        end
    end
end

@testitem "UniformScaling-based PointMass" begin
    using LinearAlgebra, TinyHugeNumbers

    for T in (Float16, Float32, Float64, BigFloat)
        matrix = convert(T, 5) * I
        dist = PointMass(matrix)

        @test variate_form(typeof(dist)) === Matrixvariate
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

        @test (@inferred entropy(dist)) == BayesBase.MinusInfinity(T)

        @test mean(dist) == matrix
        @test mode(dist) == matrix
        @test var(dist) == zero(T) * I
        @test std(dist) == zero(T) * I

        @test_throws ErrorException cov(dist)
        @test_throws ErrorException precision(dist)

        @test mean(inv, dist) ≈ inv(matrix)
    end
end

@testitem "Base.length for PointMass" begin
    using LinearAlgebra, SpecialFunctions, TinyHugeNumbers, Distributions, BayesBase

    # Scalar PointMass
    scalar = rand(Float64)
    dist_scalar = PointMass(scalar)
    @test length(dist_scalar) == 1

    # Vector PointMass
    N = 5
    vector = rand(Float64, N)
    dist_vector = PointMass(vector)
    @test length(dist_vector) == N

    # Matrix PointMass
    M = 3
    matrix = rand(Float64, M, M)
    dist_matrix = PointMass(matrix)
    @test length(dist_matrix) == M*M

    # Tensor PointMass
    D = 3
    N = 2
    tensor = rand(Float64, ntuple(_ -> N, D))
    dist_tensor = PointMass(tensor)
    @test length(dist_tensor) == N^D
end
