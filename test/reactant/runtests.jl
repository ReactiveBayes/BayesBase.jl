# The Reactant extension (`ext/BayesBaseReactantExt.jl`): converting a `MatrixWithCachedInverse`
# with `Reactant.to_rarray`, compiling `inv`, `\`, `logdet` and arithmetic on it, and the cache
# contract across the compiled boundary. Run with `make test-reactant`; it is not part of the root
# suite, since Reactant is a heavy dependency that changes often.

using Test, LinearAlgebra
using BayesBase, Reactant

spd(T, n, s) = T[1 / (i + j - 1) for i in 1:n, j in 1:n] + T(s) * I

function wrapper(S, state)
    W = MatrixWithCachedInverse(S)
    state == :factor && BayesBase.cached_cholesky_or_compute(W)
    state == :pair && inv(W)
    return W
end

count_cholesky(hlo) = count("stablehlo.cholesky", string(hlo))

f_inv(W, b) = inv(W) * b
f_solve(W, b) = W \ b
f_logdet(W) = logdet(W)
f_invinv(W, b) = inv(inv(W)) * b
f_arith(W, b) = (W + W) * b .- (2 .* W) * b

@testset "MatrixWithCachedInverse compiled with Reactant" begin
    @testset "the extension is loaded" begin
        @test Base.get_extension(BayesBase, :BayesBaseReactantExt) !== nothing
    end

    @testset "conversion, compilation and reuse" begin
        for T in (Float32, Float64), state in (:empty, :factor, :pair)
            S, S2 = spd(T, 4, 1), spd(T, 4, 2.5)
            b = T.(1:4)
            W, W2 = wrapper(S, state), wrapper(S2, state)
            R, R2, rb = Reactant.to_rarray(W), Reactant.to_rarray(W2), Reactant.to_rarray(b)

            # conversion keeps the cache state, the reciprocal cycle and the shared lock
            @test R isa MatrixWithCachedInverse{T} &&
                parent(R) isa Reactant.AbstractConcreteArray
            @test (BayesBase.cached_factor(R) === nothing) == (state == :empty)
            @test (R.inverse === nothing) == (state != :pair)
            if state == :pair
                @test R.inverse.inverse === R && R.inverse.guard === R.guard
            end

            # host caches become program inputs; empty caches are computed in the program
            expected = state == :empty ? 1 : 0
            @test count_cholesky(@code_hlo(f_inv(R, rb))) == expected
            @test count_cholesky(@code_hlo(f_solve(R, rb))) == expected
            @test count_cholesky(@code_hlo(f_logdet(R))) == expected
            @test count_cholesky(@code_hlo(f_invinv(R, rb))) == 0 # reverse lookup never factorizes
            @test count_cholesky(@code_hlo(f_arith(R, rb))) == 0

            c_inv = @compile f_inv(R, rb)
            c_solve = @compile f_solve(R, rb)
            c_logdet = @compile f_logdet(R)
            c_invinv = @compile f_invinv(R, rb)
            c_arith = @compile f_arith(R, rb)
            for (Rk, Sk) in ((R, S), (R2, S2)) # reuse on a different snapshot with matching state
                @test Array(c_inv(Rk, rb)) ≈ inv(Sk) * b
                @test Array(c_solve(Rk, rb)) ≈ Sk \ b
                @test T(c_logdet(Rk)) ≈ logdet(Sk)
                @test Array(c_invinv(Rk, rb)) ≈ Sk * b
                @test Array(c_arith(Rk, rb)) ≈ zeros(T, 4) atol = sqrt(eps(T))
            end

            # neither the host wrappers nor the converted ones gained cache entries
            @test (BayesBase.cached_factor(W) === nothing) ==
                (state == :empty) ==
                (BayesBase.cached_factor(R) === nothing)
            @test (W.inverse === nothing) == (state != :pair) == (R.inverse === nothing)
        end
    end

    @testset "construction inside compiled code and the call boundary" begin
        h(M, b) = (W=MatrixWithCachedInverse(M); inv(W) * b + W \ b + logdet(W) * (W * b))
        href(S, b) = inv(S) * b + S \ b + logdet(S) * (S * b)
        for T in (Float32, Float64)
            S, S2 = spd(T, 4, 1), spd(T, 4, 2.5)
            b = T.(1:4)
            rS, rS2, rb = Reactant.to_rarray(S),
            Reactant.to_rarray(S2),
            Reactant.to_rarray(b)
            ch = @compile h(rS, rb)
            @test Array(ch(rS, rb)) ≈ href(S, b)
            @test Array(ch(rS2, rb)) ≈ href(S2, b)
        end

        S = spd(Float64, 4, 1)
        b = Float64.(1:4)
        rb = Reactant.to_rarray(b)
        Re, Rp = Reactant.to_rarray(wrapper(S, :empty)),
        Reactant.to_rarray(wrapper(S, :pair))
        # compiled on an empty cache: extra host cache entries are ignored
        ce = @compile f_inv(Re, rb)
        @test Array(ce(Rp, rb)) ≈ inv(S) * b
        # compiled on a populated pair: the cache is a program input and cannot be missing
        cp = @compile f_inv(Rp, rb)
        @test_throws Exception cp(Re, rb)

        # object identity does not cross the boundary, values do
        g(W) = inv(W)
        cg = @compile g(Re)
        out = cg(Re)
        @test out isa MatrixWithCachedInverse && Array(parent(out)) ≈ inv(S)
        @test out.inverse !== Re && Array(parent(out.inverse)) ≈ S
    end
end
