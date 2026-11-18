@testitem "every protocol name is public and none is exported" tags = [:expectations] setup = [ExpectationToys] begin
    const PROTOCOL = [
        :expectation,
        :prepare_expectation,
        :PreparedExpectation,
        :UnsupportedExpectationError,
        :supports_expectation,
        :which_expectation_backend,
        :ExpectationLabel,
        :ExpectationExact,
        :ExpectationStochastic,
        :has_expectation_label,
        :is_expectation_exact,
        :is_expectation_stochastic,
        :combine_expectation_label,
        :with_expectation_rng,
        :get_expectation_rng,
        :ExpectationPointSet,
        :ExpectationProductPointSet,
        :ExpectationFirstOf,
        :ExpectationDynamicRouter,
        :ExpectationBackendRegistry,
        :register_expectation_backend!,
        :expectation_backend_priority,
    ]

    for name in PROTOCOL
        @test isdefined(BayesBase, name)
        if VERSION >= v"1.11"
            @test Base.ispublic(BayesBase, name)
            @test !Base.isexported(BayesBase, name)
        else
            @test name ∉ names(BayesBase) # before 1.11 `names` lists exported names only
        end
    end
end

@testitem "loading next to RxInfer's names does not clash" tags = [:expectations] setup = [ExpectationToys] begin
    # MessagePassingRulesBase and ReactiveMP export `Stochastic` and `isstochastic` for node
    # types; the protocol's names are its own, and not exported, so a module may use both
    module UsesBoth
    using BayesBase
    struct Stochastic end # stands in for the node type exported by ReactiveMP
    isstochastic(::Stochastic) = true
    const label = BayesBase.ExpectationStochastic()
    end
    @test UsesBoth.isstochastic(UsesBoth.Stochastic())
    @test UsesBoth.label isa BayesBase.ExpectationLabel
end
