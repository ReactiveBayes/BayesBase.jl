@testitem "ExpectationPointSet: a weighted sum over its points" tags = [:expectations] setup = [ExpectationToys] begin

    points = P.ExpectationPointSet((1.0, 2.0, 3.0), (0.2, 0.3, 0.5))
    @test P.expectation(points, identity) ≈ 0.2 + 0.6 + 1.5
    @test P.expectation(points, x -> x^2) ≈ 0.2 + 1.2 + 4.5
    @test length(points) == 3

    # points may be anything indexable: ranges, vectors, vectors of vectors
    @test P.expectation(P.ExpectationPointSet(1:3, [0.2, 0.3, 0.5]), identity) ≈ 2.3
    @test P.expectation(P.ExpectationPointSet([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5]), sum) ≈ 5.0
end

@testitem "ExpectationPointSet checks its lengths" tags = [:expectations] setup = [ExpectationToys] begin

    @test_throws DimensionMismatch P.ExpectationPointSet((1.0, 2.0), (1.0,))
    @test_throws ArgumentError P.ExpectationPointSet((), ())
end

@testitem "ExpectationPointSet: the result type follows the integrand, so duals pass through" tags = [:expectations] setup = [ExpectationToys] begin
    using ForwardDiff

    points = P.ExpectationPointSet((-1.0, 0.0, 1.0), (0.25, 0.5, 0.25))
    # E[(θ x + 1)²] = θ² E[x²] + 1 = θ² / 2 + 1, derivative θ
    g(θ) = P.expectation(points, x -> (θ * x + 1)^2)
    @test g(2.0) ≈ 3.0
    @test ForwardDiff.derivative(g, 2.0) ≈ 2.0
    @test @inferred(P.expectation(points, x -> ForwardDiff.Dual(x, 1.0))) isa
        ForwardDiff.Dual
end

@testitem "ExpectationPointSet: evaluation does not allocate" tags = [:expectations, :alloc] setup = [ExpectationToys] begin
    using ForwardDiff

    points = P.ExpectationPointSet((-1.0, 0.0, 1.0), (0.25, 0.5, 0.25))
    vector = P.ExpectationPointSet([-1.0, 0.0, 1.0], [0.25, 0.5, 0.25])
    @test allocations(P.expectation, points, abs2) == 0
    @test allocations(P.expectation, vector, abs2) == 0
    derivative(v, θ) = ForwardDiff.derivative(t -> P.expectation(v, x -> (t * x)^2), θ)
    @test allocations(derivative, vector, 2.0) == 0
end
