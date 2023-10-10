using Aqua, CpuId, ReTestItems, BayesBase

# `ambiguities = false` - there are quite some ambiguities, but these should be normal and should not be encountered under normal circumstances
# `piracy = false` - we extend/add some of the methods to the objects defined in the Distributions.jl
Aqua.test_all(BayesBase; ambiguities=false, piracy=false)

runtests(
    BayesBase;
    nworkers=cpucores(),
    nworker_threads=Int(cputhreads() / cpucores()),
    memory_threshold=1.0,
)
