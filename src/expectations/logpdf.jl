"""
    Logpdf(dist)

The integrand `x -> logpdf(dist, x)`, as a type that backends can dispatch on, so that a closed
form of `E_q[log p(x)]` can be found. `dist` is anything with a `logpdf`, a product of
distributions or a message included.
"""
struct Logpdf{D}
    dist::D
end

(integrand::Logpdf)(x) = logpdf(integrand.dist, x)
