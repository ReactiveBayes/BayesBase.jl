@testitem "mirrorlog" begin
    for T in (Float32, Float64, BigFloat)
        foreach(rand(T, 10)) do number
            @test mirrorlog(number) ≈ log(one(number) - number)
        end
    end
end

@testitem "xtlog" begin
    for T in (Float32, Float64, BigFloat)
        foreach(rand(T, 10)) do number
            @test xtlog(number) ≈ number * log(number)
        end
    end
end

@testitem "clamplog" begin
    using TinyHugeNumbers

    for T in (Float32, Float64, BigFloat)
        foreach(rand(T, 10)) do number
            @test clamplog(number + 2tiny) ≈ log(number + 2tiny)
        end

        @test clamplog(zero(T)) ≈ log(convert(T, tiny))
    end
end

@testitem "dtanh" begin
    for T in (Float32, Float64, BigFloat)
        foreach(rand(T, 10)) do number
            @test dtanh(number) ≈ 1 - tanh(number)^2
        end
    end
end

@testitem "UnspecifiedDomain" begin
    using DomainSets

    @test 1 ∈ UnspecifiedDomain()
    @test (1, 1) ∈ UnspecifiedDomain()
    @test [0, 1] ∈ UnspecifiedDomain()

    @test fuse_supports(UnspecifiedDomain(), UnspecifiedDomain()) === UnspecifiedDomain()
    @test fuse_supports(FullSpace(), UnspecifiedDomain()) === FullSpace()
    @test fuse_supports(UnspecifiedDomain(), FullSpace()) === FullSpace()
end

@testitem "UnspecifiedDimension" begin
    using DomainSets

    @test UnspecifiedDimension() == 1
    @test UnspecifiedDimension() == 2
    @test UnspecifiedDimension() != 1
    @test UnspecifiedDimension() != 2
end

@testitem "isequal_typeof" begin
    @test !isequal_typeof(1, 1.0)
    @test isequal_typeof(1.0, 1.0)
    @test !isequal_typeof([1.0], 1.0)
    @test !isequal_typeof([1.0], [1])
    @test isequal_typeof([1.0], [1.0])
end

@testitem "CountingReal" begin
    import BayesBase: Infinity, MinusInfinity

    for T in (Float32, Float64, BigFloat)
        r = CountingReal(zero(T), 0)

        @test eltype(r) === T
        @test float(r) ≈ zero(T)
        @test float(r + 1) ≈ one(T)
        @test float(1 + r) ≈ one(T)
        @test float(r - 1) ≈ -one(T)
        @test float(1 - r) ≈ one(T)

        @test float(r - 1 + Infinity(T)) ≈ convert(T, Inf)
        @test float(1 - r + Infinity(T)) ≈ convert(T, Inf)
        @test float(r - 1 + Infinity(T) - Infinity(T)) ≈ -one(T)
        @test float(1 - r + Infinity(T) - Infinity(T)) ≈ one(T)
        @test float(r - 1 + Infinity(T) + MinusInfinity(T)) ≈ -one(T)
        @test float(1 - r + Infinity(T) + MinusInfinity(T)) ≈ one(T)
        @test float(r - 1 - Infinity(T) - MinusInfinity(T)) ≈ -one(T)
        @test float(1 - r - Infinity(T) - MinusInfinity(T)) ≈ one(T)

        @test float(convert(CountingReal, r)) ≈ zero(T)
        @test float(convert(CountingReal{Float64}, r)) ≈ zero(Float64)
    end
end

@testitem "mcov!" begin
    using StatsFuns, BayesBase, JET
    import BayesBase: mcov!

    for n in 2:5:20, j in 3:5:20
        X = rand(j, n)
        Y = rand(j, n)
        Z = rand(n, n)

        @inferred(mcov!(Z, X, Y))

        @test all(Z .≈ cov(X, Y))

        tmp1 = zeros(eltype(Z), size(X, 2))
        tmp2 = zeros(eltype(Z), size(Y, 2))
        tmp3 = similar(X)
        tmp4 = similar(Y)

        @inferred(mcov!(Z, X, Y; tmp1=tmp1, tmp2=tmp2, tmp3=tmp3, tmp4=tmp4))

        @test all(Z .≈ cov(X, Y))

        @report_opt mcov!(Z, X, Y; tmp1=tmp1, tmp2=tmp2, tmp3=tmp3, tmp4=tmp4)
        @test @allocated(mcov!(Z, X, Y; tmp1=tmp1, tmp2=tmp2, tmp3=tmp3, tmp4=tmp4)) === 0
    end
end

@testitem "InplaceLogpdf" begin
    import BayesBase: InplaceLogpdf
    using Distributions, LinearAlgebra, StableRNGs

    @testset "Vector based samples" begin
        distribution = Beta(10, 10)
        fn = (x) -> logpdf(distribution, x)
        inplacefn = convert(InplaceLogpdf, fn)

        @test fn !== inplacefn

        rng = StableRNG(42)
        samples = rand(rng, distribution, 100)
        evaluated = map(fn, samples)

        container = similar(evaluated)
        inplacefn(container, samples)

        @test evaluated == container
    end

    @testset "Matrix based samples" begin
        distribution = MvNormal(ones(2), ones(2))
        fn = (x) -> logpdf(distribution, x)
        inplacefn = convert(InplaceLogpdf, fn)

        @test inplacefn !== fn

        rng = StableRNG(42)
        samples = rand(rng, distribution, 100)
        evaluated = map(fn, eachcol(samples))

        container = similar(evaluated)
        inplacefn(container, eachcol(samples))

        @test evaluated == container
    end

    @testset "Do not convert already inplace version" begin
        distribution = MvNormal(ones(2), ones(2))
        fn = InplaceLogpdf((out, x) -> logpdf!(out, distribution, x))
        inplacefn = convert(InplaceLogpdf, fn)

        @test inplacefn === fn

        rng = StableRNG(42)
        samples = rand(rng, distribution, 100)
        evaluated = zeros(100)
        fn(evaluated, eachcol(samples))

        container = similar(evaluated)
        inplacefn(container, eachcol(samples))

        @test evaluated == container
    end

    @testset "Shouldn't allocate anything for simple `logpdf!`" begin
        fn = InplaceLogpdf((out, x) -> out .= log.(x))
        samples = 1:10
        out = zeros(10)
        fn(out, samples)
        @test out == log.(samples)
        @test @allocated(fn(out, samples)) === 0
    end
end
