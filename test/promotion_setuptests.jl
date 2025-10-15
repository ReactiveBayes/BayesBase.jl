using BayesBase, LinearAlgebra, Distributions, StableRNGs

function generate_random_distributions(
    (::Type{V})=Any; seed=abs(rand(Int)), Types=(Float32, Float64)
) where {V}
    rng = StableRNG(seed)
    distributions = []

    # Add `Univariate` distributions
    for T in Types
        push!(distributions, Normal(rand(rng, T), rand(rng, T)))
        push!(distributions, Beta(rand(rng, T), rand(rng, T)))
        push!(distributions, Gamma(rand(rng, T), rand(rng, T)))
    end

    # Add `Multivariate` distributions
    for T in Types, n in (2, 3)
        push!(distributions, MvNormal(rand(rng, T, n)))
    end

    # Add `Matrixvariate` distributions
    for T in Types, n in (2, 3)
        push!(distributions, InverseWishart(5one(T), Matrix(Diagonal(ones(n)))))
        push!(distributions, Wishart(5one(T), Matrix(Diagonal(ones(n)))))
    end

    return filter((dist) -> variate_form(typeof(dist)) <: V, distributions)
end
