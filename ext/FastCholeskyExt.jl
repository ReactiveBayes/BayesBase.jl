module FastCholeskyExt

using FastCholesky
using BayesBase
using LinearAlgebra: logdet, inv

function FastCholesky.cholinv(input::ArrowheadMatrix)
    return inv(input)
end

# Cold caches are filled with `fastcholesky`; the factor is kept so later solves, log-determinants
# and inverses reuse it. Warm caches are shared with `inv`, `\` and `logdet`.
function FastCholesky.fastcholesky(A::MatrixWithCachedInverse)
    return BayesBase.cached_cholesky_or_compute(fastcholesky, A)
end
function FastCholesky.cholinv(A::MatrixWithCachedInverse)
    return BayesBase.cached_inverse_or_compute(A -> inv(fastcholesky(A)), A)
end
FastCholesky.chollogdet(A::MatrixWithCachedInverse) = logdet(fastcholesky(A))
FastCholesky.cholinv_logdet(A::MatrixWithCachedInverse) = (cholinv(A), chollogdet(A))

end
