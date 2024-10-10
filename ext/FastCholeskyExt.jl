module FastCholeskyExt

    using FastCholesky
    using BayesBase

    function FastCholesky.cholinv(input::ArrowheadMatrix)
        return inv(input)
    end

end