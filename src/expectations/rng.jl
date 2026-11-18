"""
    with_expectation_rng(backend, rng)

`backend` with `rng` as its random number generator if it has none yet. A backend that already
has a generator keeps it, so a generator set by the user is never replaced. Composite backends,
such as routers, pass `rng` on to their parts. The default returns `backend` unchanged.
"""
with_expectation_rng(backend, rng) = backend

"""
    get_expectation_rng(backend)

The random number generator of `backend`, or `nothing`.
"""
get_expectation_rng(backend) = nothing
