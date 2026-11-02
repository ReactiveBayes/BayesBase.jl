"""
    withrng(backend, rng)

`backend` with `rng` filled in wherever it has no random number generator yet. A backend that
already has one keeps it, so an RNG set by the user is never overwritten from above. Composite
backends fill their parts. The default returns `backend` unchanged.
"""
withrng(backend, rng) = backend

"""
    getrng(backend)

The random number generator of `backend`, or `nothing`.
"""
getrng(backend) = nothing
