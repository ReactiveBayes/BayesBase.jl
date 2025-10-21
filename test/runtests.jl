using Aqua, Hwloc, ReTestItems, BayesBase

# `ambiguities = false` - there are quite some ambiguities, but these should be normal and should not be encountered under normal circumstances
# `piracy = false` - we extend/add some of the methods to the objects defined in the Distributions.jl
Aqua.test_all(
    BayesBase;
    ambiguities=false,
    piracies=false,
    deps_compat=(; check_extras=false, check_weakdeps=true),
)

ncores = max(Hwloc.num_physical_cores(), 1)
nthreads = max(Hwloc.num_virtual_cores(), 1)
threads_per_core = max(Int(floor(nthreads / ncores)), 1)

runtests(BayesBase; nworkers=ncores, nworker_threads=threads_per_core, memory_threshold=1.0)
