# BayesBase

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/BayesBase.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/BayesBase.jl/dev/)
[![Build Status](https://github.com/reactivebayes/BayesBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/reactivebayes/BayesBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/reactivebayes/BayesBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/reactivebayes/BayesBase.jl)
[![Code style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`BayesBase` is a package that serves as an umbrella, defining, exporting, and re-exporting methods essential for Bayesian statistics specifically for the [`RxInfer` ecosystem](https://github.com/reactivebayes/RxInfer.jl). 

It also defines the expectation protocol: one way to ask for `E_q[f]`, which backends in other
packages answer, such as the quadrature rules of
[`ExpectationApproximations`](https://github.com/reactivebayes/ExpectationApproximations.jl). See
the [Expectations](https://reactivebayes.github.io/BayesBase.jl/dev/expectations/) page of the
documentation.

Related projects:

- [`RxInfer`](https://github.com/reactivebayes/RxInfer.jl)
- [`ExponentialFamily`](https://github.com/reactivebayes/ExponentialFamily.jl)


# License

[MIT License](LICENSE) Copyright (c) 2023-2024 BIASlab, 2024-present ReactiveBayes
