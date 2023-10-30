@testitem "FactorizedJoint" begin
    using Distributions

    vmultipliers = [
        (Normal(),),
        (Normal(), Beta(1.0, 1.0)),
        (Normal(), Gamma(), MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])),
    ]

    @testset "getindex" begin
        for multipliers in vmultipliers
            product = FactorizedJoint(multipliers)
            @test length(product) === length(multipliers)
            for i in eachindex(multipliers)
                @test product[i] === multipliers[i]
            end
        end
    end

    @testset "entropy" begin
        for multipliers in vmultipliers
            product = FactorizedJoint(multipliers)
            @test entropy(product) ≈ mapreduce(entropy, +, multipliers)
        end
    end

    @testset "convert_paramfloattype" begin 
        for T in (Float32, Float64, BigFloat),  multipliers in vmultipliers
            for component in components(convert_paramfloattype(T, FactorizedJoint(multipliers)))
                @test paramfloattype(component) === T
            end
        end
    end

    @testset "isapprox" begin
        @test FactorizedJoint((Normal(),)) ≈ FactorizedJoint((Normal(),))
        @test !(FactorizedJoint((Normal(0, 1),)) ≈ FactorizedJoint((Normal(1, 1),)))

        @test FactorizedJoint((Gamma(1.0, 1.0), Normal(0.0, 1.0))) ≈
            FactorizedJoint((Gamma(1.000001, 1.0), Normal(0.0, 1.0000000001))) atol = 1e-5
        @test !(
            FactorizedJoint((Gamma(1.0, 1.0), Normal(0.0, 1.0))) ≈
            FactorizedJoint((Gamma(1.000001, 1.0), Normal(0.0, 5.0000000001)))
        )
        @test !(
            FactorizedJoint((Gamma(1.0, 2.0), Normal(0.0, 1.0))) ≈
            FactorizedJoint((Gamma(1.000001, 1.0), Normal(0.0, 1.0000000001)))
        )
    end
end