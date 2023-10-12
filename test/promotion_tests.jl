@testitem "TypeConverter" begin
    import BayesBase: PromoteTypeConverter

    for original_T in (Float16, Float32, Float64),
        target_T in (Float16, Float32, Float64),
        n in (1, 2, 3)

        converter = PromoteTypeConverter(target_T, convert)

        @test typeof(@inferred(converter(rand(original_T)))) === target_T
    end
end

@testitem "convert_paramfloattype" begin
    include("./promotion_setuptests.jl")

    for T in (Float32, Float64, BigFloat)
        @test @inferred(eltype(convert_paramfloattype(T, [1.0, 1.0]))) === T
        @test @inferred(eltype(convert_paramfloattype(T, [1.0 1.0; 1.0 1.0]))) === T
        @test @inferred(eltype(convert_paramfloattype(T, 1.0))) === T

        for distribution in generate_random_distributions()
            @test @inferred(paramfloattype(convert_paramfloattype(T, distribution))) === T
        end
    end
end

@testitem "sampletype" begin
    include("./promotion_setuptests.jl")

    for distribution in generate_random_distributions()
        sample = rand(distribution)
        @test @inferred(sampletype(distribution)) === typeof(sample)
    end
end

@testitem "promote_sampletype" begin
    include("./promotion_setuptests.jl")

    combinations = [
        Iterators.product(
            generate_random_distributions(Univariate),
            generate_random_distributions(Univariate),
        ),
        Iterators.product(
            generate_random_distributions(Multivariate),
            generate_random_distributions(Multivariate),
        ),
        Iterators.product(
            generate_random_distributions(Matrixvariate),
            generate_random_distributions(Matrixvariate),
        ),
    ]
    for combination in combinations
        for distributions in combination
            samples = rand.(distributions)
            @static if VERSION >= v"1.8"
                @test @inferred(promote_sampletype(distributions...)) ===
                    promote_type(typeof.(samples)...)
            else
                @test promote_sampletype(distributions...) ===
                    promote_type(typeof.(samples)...)
            end
        end
    end
end

@testitem "deep_eltype" begin
    include("./promotion_setuptests.jl")

    for type in [Float32, Float64, Complex{Float64}, BigFloat]
        @test deep_eltype(type) === type
        @test deep_eltype(zero(type)) === type

        scaling = one(type) * I(2)
        vector = zeros(type, 10)
        matrix = zeros(type, 10, 10)

        vector_of_scaling = [scaling, scaling]
        vector_of_vectors = [vector, vector]
        vector_of_matrices = [matrix, matrix]
        matrix_of_scaling = [scaling scaling; scaling scaling]
        matrix_of_vector = [vector vector; vector vector]
        matrix_of_matrices = [matrix matrix; matrix matrix]

        @test deep_eltype(scaling) === type
        @test deep_eltype(vector) === type
        @test deep_eltype(matrix) === type
        @test deep_eltype(vector_of_scaling) === type
        @test deep_eltype(vector_of_vectors) === type
        @test deep_eltype(vector_of_matrices) === type
        @test deep_eltype(matrix_of_scaling) === type
        @test deep_eltype(matrix_of_vector) === type
        @test deep_eltype(matrix_of_matrices) === type
    end
end

@testitem "samplefloattype" begin
    include("./promotion_setuptests.jl")

    for distribution in generate_random_distributions()
        sample = rand(distribution)
        @test @inferred(samplefloattype(distribution)) === deep_eltype(typeof(sample))
    end
end

@testitem "promote_samplefloattype" begin
    include("./promotion_setuptests.jl")

    combinations = [
        Iterators.product(
            generate_random_distributions(Univariate),
            generate_random_distributions(Univariate),
        ),
        Iterators.product(
            generate_random_distributions(Univariate),
            generate_random_distributions(Matrixvariate),
        ),
        Iterators.product(
            generate_random_distributions(Multivariate),
            generate_random_distributions(Multivariate),
        ),
        Iterators.product(
            generate_random_distributions(Multivariate),
            generate_random_distributions(Matrixvariate),
        ),
        Iterators.product(
            generate_random_distributions(Matrixvariate),
            generate_random_distributions(Matrixvariate),
        ),
        Iterators.product(
            generate_random_distributions(Univariate),
            generate_random_distributions(Matrixvariate),
            generate_random_distributions(Matrixvariate),
        ),
    ]

    for combination in combinations
        for distributions in combination
            samples = rand.(distributions)
            @static if VERSION >= v"1.8"
                @test @inferred(promote_samplefloattype(distributions...)) ===
                    promote_type(deep_eltype.(typeof.(samples))...)
            else
                @test promote_samplefloattype(distributions...) ===
                    promote_type(deep_eltype.(typeof.(samples))...)
            end
        end
    end
end