
@testitem "Contingency: components" begin
    @test components(Contingency(ones(3, 3))) == ones(3, 3) ./ 9
    @test components(Contingency(ones(3, 3), Val(true))) == ones(3, 3) ./ 9
    @test components(Contingency(ones(3, 3), Val(false))) == ones(3, 3) # Matrix is wrong, but just to test that `false` is working
    @test components(Contingency(ones(4, 4))) == ones(4, 4) ./ 16
    @test components(Contingency(ones(4, 4), Val(true))) == ones(4, 4) ./ 16
    @test components(Contingency(ones(4, 4), Val(false))) == ones(4, 4)
end

@testitem "Contingency: vague" begin
    @test_throws MethodError vague(Contingency)

    d1 = vague(Contingency, 3)

    @test typeof(d1) <: Contingency
    @test components(d1) ≈ ones(3, 3) ./ 9

    d2 = vague(Contingency, 4)

    @test typeof(d2) <: Contingency
    @test components(d2) ≈ ones(4, 4) ./ 16

    d3 = vague(Contingency, 5, 3)
    @test typeof(d3) <: Contingency
    @test components(d3) ≈ ones(5, 5, 5) ./ 125
end

@testitem "Contingency: entropy" begin
    @test entropy(Contingency([0.7 0.1; 0.1 0.1])) ≈ 0.9404479886553263
    @test entropy(Contingency(10.0 * [0.7 0.1; 0.1 0.1])) ≈ 0.9404479886553263
    @test entropy(Contingency([0.07 0.41; 0.31 0.21])) ≈ 1.242506182893139
    @test entropy(Contingency(10.0 * [0.07 0.41; 0.31 0.21])) ≈ 1.242506182893139
    @test entropy(Contingency([0.09 0.00; 0.00 0.91])) ≈ 0.30253782309749805
    @test entropy(Contingency(10.0 * [0.09 0.00; 0.00 0.91])) ≈ 0.30253782309749805
    @test !isnan(entropy(Contingency([0.0 1.0; 1.0 0.0])))
    @test !isinf(entropy(Contingency([0.0 1.0; 1.0 0.0])))

    @test entropy(
        Contingency(
            stack([
                0.3 * [0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3],
                0.7 * [0.2 0.1 0.7; 0.4 0.3 0.3; 0.1 0.6 0.3],
            ],),
        ),
    ) ≈ 2.6390313416381166

    @test entropy(Contingency(ones(2, 2, 2))) == 2.0794415416798357
end

@testitem "Contingency: isapprox" begin
    for n in 2:5
        A = rand(n, n)
        @test Contingency(A) ≈ Contingency(A)
        @test Contingency(A, Val(true)) ≈ Contingency(A, Val(true))
        @test Contingency(A, Val(false)) ≈ Contingency(A, Val(false))
    end
end
