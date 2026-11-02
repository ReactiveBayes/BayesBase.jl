@testitem "every protocol name is public and none is exported" tags = [:expectations] setup = [ExpectationToys] begin
    const PROTOCOL = [
        :expectation,
        :prepare_expectation,
        :Prepared,
        :NoBackendError,
        :supports,
        :which_backend,
        :ExpectationLabel,
        :Exact,
        :Stochastic,
        :haslabel,
        :isexact,
        :isstochastic,
        :combine_label,
        :withrng,
        :getrng,
        :Logpdf,
        :PointSet,
        :ProductPointSet,
        :FirstOf,
        :DynamicRouter,
        :BackendRegistry,
        :register_backend!,
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
    # types; BayesBase does not export the protocol, so a module may use both
    module UsesBoth
    using BayesBase
    struct Stochastic end # stands in for the node type exported by ReactiveMP
    isstochastic(::Stochastic) = true
    const label = BayesBase.Stochastic()
    end
    @test UsesBoth.isstochastic(UsesBoth.Stochastic())
    @test UsesBoth.label isa BayesBase.ExpectationLabel
end
