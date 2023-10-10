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


