
@testitem "ArrowheadMatrix: Construction and Properties" begin
    include("algebrasetup_setuptests.jl")
    α = 2.0
    z = [1.0, 2.0, 3.0]
    D = [4.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    @test size(A) == (4, 4)
end

@testitem "ArrowheadMatrix: Multiplication with Vector" begin
    include("algebrasetup_setuptests.jl")
    for n in 2:20
        α = randn()
        z = randn(n)
        D = randn(n)
        A = ArrowheadMatrix(α, z, D)

        x = randn(n+1)
        y = A * x

        dense_A = [Diagonal(D) z; z' α]
        converted_A = convert(Matrix, A)
        @test dense_A ≈ converted_A

        y_expected = dense_A * x
        @test y ≈ y_expected
    end
end

@testitem "ArrowheadMatrix: Solving Linear System" begin
    include("algebrasetup_setuptests.jl")
    for n in 2:20
        α = randn()^2 .+ 1
        z = randn(n)
        D = randn(n) .^ 2 .+ 1
        A = ArrowheadMatrix(α, z, D)

        x = randn(n+1)
        y = A \ x

        dense_A = convert(Matrix, A)
        y_expected = dense_A \ x

        @test y ≈ y_expected
    end
end

@testitem "InvArrowheadMatrix: Construction and Properties" begin
    include("algebrasetup_setuptests.jl")

    α = 2.0
    z = [1.0, 2.0, 3.0]
    D = [4.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    A_inv = inv(A)
    @test size(A_inv) == (4, 4)
end

@testitem "InvArrowheadMatrix: Multiplication with Vector" begin
    α = 2.0
    z = [1.0, 2.0, 3.0]
    D = [4.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    b = [7.0, 8.0, 9.0, 10.0]
    A_inv = inv(A)
    b = [7.0, 8.0, 9.0, 10.0]
    x = A_inv * b

    x_expected = A \ b
    @test x ≈ x_expected
end

@testitem "InvArrowheadMatrix: Division with Vector" begin
    α = 2.0
    z = [1.0, 2.0, 3.0]
    D = [4.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    A_inv = inv(A)
    x = [1.0, 2.0, 3.0, 4.0]
    y = A_inv \ x

    y_expected = A * x
    @test y == y_expected
end

@testitem "InvArrowheadMatrix: Conversion to Dense Matrix" begin
    include("algebrasetup_setuptests.jl")

    α = 2.0
    z = [1.0, 2.0, 3.0]
    D = [4.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    A_inv = inv(A)

    A_inv_dense = convert(Matrix, A_inv)
    A_dense = convert(Matrix, A)

    # Verify that A_inv_dense * A_dense ≈ Identity matrix
    I_approx = A_inv_dense * A_dense
    I_n = Matrix{Float64}(I, size(A_dense))
    @test I_approx ≈ I_n
end

@testitem "ArrowheadMatrix: division vs ldiv!" begin
    include("algebrasetup_setuptests.jl")

    for n in [10, 20]
        α = rand()^2 + 1.0  # Ensure α is not too close to zero
        z = randn(n)
        D = rand(n) .^ 2 .+ 1.0  # Ensure D elements are not too close to zero
        A = ArrowheadMatrix(α, z, D)

        b = randn(n+1)
        x1 = A \ b

        x2 = similar(b)
        LinearAlgebra.ldiv!(x2, A, b)
        @test x1 ≈ x2

        allocs = @allocations LinearAlgebra.ldiv!(x2, A, b)
        @test allocs == 0
    end
end

@testitem "ArrowheadMatrix: Performance comparison with dense matrix" begin
    using BenchmarkTools
    using StableRNGs

    include("algebrasetup_setuptests.jl")

    rng = StableRNG(1234)

    for n in [10, 100, 1000]
        α = rand(rng)^2 + 1.0  # Ensure α is not too close to zero
        z = randn(rng, n)
        D = rand(rng, n) .^ 2 .+ 1.0 # Ensure D elements are not too close to zero
        A_arrow = ArrowheadMatrix(α, z, D)

        # Create equivalent dense matrix
        A_dense = [Diagonal(D) z; z' α]

        b = randn(rng, n+1)

        benchmark_arrow = @benchmark $A_arrow \ $b;
        benchmark_dense = @benchmark $A_dense \ $b;

        # our implementation is at least k times faster on average
        # where k is dimensionality divided by 3
        k = n ÷ 3
        @test minimum(benchmark_arrow.times) < minimum(benchmark_dense.times)/k
        @test benchmark_arrow.allocs < benchmark_dense.allocs

        x_arrow = A_arrow \ b
        x_dense = A_dense \ b
        @test x_arrow ≈ x_dense
    end
end

@testitem "ArrowheadMatrix: Performance comparison with cholinv" begin
    using BenchmarkTools
    using FastCholesky
    using StableRNGs

    include("algebrasetup_setuptests.jl")

    rng = StableRNG(1234)
    for n in [10, 100, 1000]
        α = rand(rng)^2 + 1.0  # Ensure α is not too close to zero
        z = randn(rng, n)
        D = rand(rng, n) .^ 2 .+ 1.0 # Ensure D elements are not too close to zero
        A_arrow = ArrowheadMatrix(α, z, D)

        # Create equivalent dense matrix
        A_dense = [Diagonal(D) z; z' α]

        b = randn(rng, n+1)

        benchmark_arrow = @benchmark cholinv($A_arrow) * $b;
        benchmark_dense = @benchmark cholinv($A_dense) * $b;

        # our implementation is at least k times faster on average
        # where k is dimensionality divided by 3
        k = n ÷ 3
        @test minimum(benchmark_arrow.times) < minimum(benchmark_dense.times)/k
        @test benchmark_arrow.allocs < benchmark_dense.allocs

        x_arrow = A_arrow \ b
        x_dense = A_dense \ b
        @test x_arrow ≈ x_dense
    end
end

@testitem "ArrowheadMatrix: Memory allocation comparison with dense matrix" begin
    using Test
    include("algebrasetup_setuptests.jl")

    function memory_size(x)
        return Base.summarysize(x)
    end

    sizes = [10, 100, 1000, 10000]
    arrow_mem = zeros(Int, length(sizes))
    dense_mem = zeros(Int, length(sizes))

    for (i, n) in enumerate(sizes)
        α = rand()^2 + 1.0
        z = randn(n)
        D = rand(n) .^ 2 .+ 1.0

        A_arrow = ArrowheadMatrix(α, z, D)
        A_dense = [Diagonal(D) z; z' α]

        arrow_mem[i] = memory_size(A_arrow)
        dense_mem[i] = memory_size(A_dense)
        @test arrow_mem[i] < dense_mem[i]
    end

    mem_ratio = dense_mem ./ arrow_mem

    for i in 2:length(sizes)
        ratio_growth = mem_ratio[i] / mem_ratio[i - 1]
        size_growth = sizes[i] / sizes[i - 1]
        @test isapprox(ratio_growth, size_growth, rtol=0.5)
    end
end

@testitem "ArrowheadMatrix: Error handling comparison with dense matrix" begin
    include("algebrasetup_setuptests.jl")

    function test_error_consistency(A_arrow, A_dense, operation)
        arrow_error = nothing
        dense_error = nothing

        try
            operation(A_arrow)
        catch e
            arrow_error = e
        end

        try
            operation(A_dense)
        catch e
            dense_error = e
        end

        if isnothing(arrow_error) && isnothing(dense_error)
            @test true  # Both succeeded, no error
        elseif !isnothing(arrow_error) && !isnothing(dense_error)
            @test typeof(arrow_error) == typeof(dense_error)  # Same error type
        else
            @test false  # One threw an error while the other didn't
        end
    end

    for n in [3, 10]
        α = randn()
        z = randn(n)
        D = randn(n)
        A_arrow = ArrowheadMatrix(α, z, D)
        A_dense = [Diagonal(D) z; z' α]

        # Test invalid dimension for multiplication
        invalid_vector = randn(n+2)
        test_error_consistency(A_arrow, A_dense, A -> A * invalid_vector)

        # Test multiplication with matrix of incorrect size
        invalid_matrix = randn(n+2, n)
        test_error_consistency(A_arrow, A_dense, A -> A * invalid_matrix)

        # Test singularity in linear solve
        singular_α = 0.0
        singular_z = zeros(n)
        singular_D = vcat(0.0, ones(n-1))
        A_arrow_singular = ArrowheadMatrix(singular_α, singular_z, singular_D)
        A_dense_singular = [Diagonal(singular_D) singular_z; singular_z' singular_α]
        b = randn(n+1)
        test_error_consistency(A_arrow_singular, A_dense_singular, A -> A \ b)

        # Test linear solve with vector of incorrect size
        invalid_b = randn(n+2)
        test_error_consistency(A_arrow, A_dense, A -> A \ invalid_b)

        # Test BoundsError consistency
        test_error_consistency(A_arrow, A_dense, A -> A[n + 2, n + 2])
        test_error_consistency(A_arrow, A_dense, A -> A[0, 1])
        test_error_consistency(A_arrow, A_dense, A -> A[1, 0])
        test_error_consistency(A_arrow, A_dense, A -> A[-1, -1])

        #Test ≈ error
        test_error_consistency(A_arrow, A_dense, A -> A ≈ zeros(n+1, n+1))

        #Test matmul error 
        test_error_consistency(A_arrow, A_dense, A -> A * zeros(n+1, n+1))
        test_error_consistency(A_arrow, A_dense, A -> zeros(n+1, n+1) * A)

        #Test dot (x, inv(A), y)
        test_error_consistency(A_arrow, A_dense, A -> dot(zeros(n+1), inv(A), zeros(n)))
        test_error_consistency(A_arrow, A_dense, A -> dot(zeros(n), inv(A), zeros(n+1)))
    end
end

@testitem "ArrowheadMatrix getindex based methods: matmul and ≈" begin
    include("algebrasetup_setuptests.jl")

    @testset "ArrowheadMatrix: matmul" begin
        for n in [3, 5, 10]
            α = randn()
            z = randn(n)
            D = randn(n)
            A = ArrowheadMatrix(α, z, D)

            B = randn(n+1, n+1)

            C_right = A * B
            C_right_dense = convert(Matrix, A) * B
            @test C_right ≈ C_right_dense

            C_left = B * A
            C_left_dense = B * convert(Matrix, A)
            @test C_left ≈ C_left_dense

            # Check that the result is a dense matrix
            @test typeof(C_right) <: Matrix
            @test typeof(C_left) <: Matrix
        end
    end

    @testset "ArrowheadMatrix: ≈" begin
        for n in [3, 5, 10]
            α = randn()
            z = randn(n)
            D = randn(n)
            A = ArrowheadMatrix(α, z, D)
            B = ArrowheadMatrix(α+1, z, D)
            dense_A = convert(Matrix, A)
            @test A ≈ dense_A
            @test !(A ≈ B)
            @test inv(A) ≈ inv(A)
            @test !(inv(A) ≈ inv(B))
        end
    end
end

@testitem "ArrowheadMatrix: getindex with Warning" begin
    include("algebrasetup_setuptests.jl")

    α = 2.0
    z = [1.0, 2.0, 3.0]
    D = [4.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)

    # Test that the warning is shown only once
    @test_logs (
        :warn,
        "getindex was called on ArrowheadMatrix. This may lead to suboptimal performance. Consider using specialized methods if available.",
    ) begin
        @test A[1, 1] == 4.0
        @test A[2, 2] == 5.0
        @test A[3, 3] == 6.0
        @test A[4, 4] == 2.0
    end
end

@testitem "InvArrowheadMatrix: dot(x, A, y) comparison with dense matrix" begin
    using LinearAlgebra
    include("algebrasetup_setuptests.jl")

    for n in [3, 5, 10]
        α = rand() + n
        z = randn(n)
        D = rand(n) .+ n

        A = ArrowheadMatrix(α, z, D)
        A_inv = inv(A)

        x = randn(n + 1)
        y = randn(n + 1)

        result_arrowhead = dot(x, A_inv, y)
        A_dense = Matrix(A)
        A_inv_dense = inv(A_dense)
        result_dense = dot(x, A_inv_dense * y)

        @test isapprox(result_arrowhead, result_dense, rtol=1e-5)
    end
end
