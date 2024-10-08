

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
    for n in 2:2
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
        D = randn(n).^2 .+ 1
        A = ArrowheadMatrix(α, z, D)
        
        x = randn(n+1)
        y = A \ x
        
        dense_A = convert(Matrix, A)
        y_expected = dense_A \ x

        @test y ≈ y_expected
    end
end

@testitem "ArrowheadMatrix: Handling Singular Matrices" begin
    include("algebrasetup_setuptests.jl")

    α = 0.0
    z = [0.0, 0.0, 0.0]
    D = [1.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    
    b = [7.0, 8.0, 9.0, 10.0]

    @test_throws DomainError A \ b

    α = 0.0
    z = [1.0, 0.0, 0.0]
    D = [0.0, 5.0, 6.0]
    A = ArrowheadMatrix(α, z, D)
    b = [7.0, 8.0, 9.0, 10.0]

    @test_throws DomainError A \ b
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
        D = rand(n).^2 .+ 1.0  # Ensure D elements are not too close to zero
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
    include("algebrasetup_setuptests.jl")

    for n in [10, 100, 1000]
        α = rand()^2 + 1.0  # Ensure α is not too close to zero
        z = randn(n)
        D = rand(n).^2 .+ 1.0 # Ensure D elements are not too close to zero
        A_arrow = ArrowheadMatrix(α, z, D)
        
        # Create equivalent dense matrix
        A_dense = [Diagonal(D) z; z' α]
        
        b = randn(n+1)
        
        # Warm-up runs
        _ = A_arrow \ b
        _ = A_dense \ b
        
        # Benchmark ArrowheadMatrix division
        time_arrow = @btime $A_arrow \ $b
        allocs_arrow = @allocations A_arrow \ b
        
        # Benchmark dense matrix division
        time_dense = @btime $A_dense \ $b
        allocs_dense = @allocations A_dense \ b
        
        # Check if ArrowheadMatrix is more efficient
        @test time_arrow < time_dense
        @test allocs_arrow < allocs_dense
        
        # Check correctness
        x_arrow = A_arrow \ b
        x_dense = A_dense \ b
        @test x_arrow ≈ x_dense
    end
end