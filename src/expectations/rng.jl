"""
    with_expectation_rng(backend, rng)

`backend` with `rng` filled in wherever it has no random number generator yet. A backend that
already has one keeps it, so an RNG set by the user is never overwritten from above. Composite
backends fill their parts. The default returns `backend` unchanged.
"""
with_expectation_rng(backend, rng) = backend

"""
    get_expectation_rng(backend)

The random number generator of `backend`, or `nothing`.
"""
get_expectation_rng(backend) = nothing
