using BayesBase, Distributions

import BayesBase:
    prod,
    default_prod_rule,
    ProductOf,
    LinearizedProductOf,
    getleft,
    getright,
    UnspecifiedProd,
    PreserveTypeProd,
    PreserveTypeLeftProd,
    PreserveTypeRightProd,
    ClosedProd,
    GenericProd

## ===========================================================================
## Tests fixtures

# An object, which does not specify any prod rules
struct SomeUnknownObject end

# Two objects that 
# - implement `ClosedProd` between each other 
# - implement `prod` with `ClosedProd` between each other 
# - can be eaily converted between each other
# - can be converted to an `Int`
struct ObjectWithClosedProd1 end
struct ObjectWithClosedProd2 end

function BayesBase.default_prod_rule(
    ::Type{ObjectWithClosedProd1}, ::Type{ObjectWithClosedProd1}
)
    return PreserveTypeProd(ObjectWithClosedProd1)
end
function BayesBase.default_prod_rule(
    ::Type{ObjectWithClosedProd2}, ::Type{ObjectWithClosedProd2}
)
    return PreserveTypeProd(ObjectWithClosedProd2)
end
function BayesBase.default_prod_rule(
    ::Type{ObjectWithClosedProd1}, ::Type{ObjectWithClosedProd2}
)
    return PreserveTypeProd(ObjectWithClosedProd1)
end
function BayesBase.default_prod_rule(
    ::Type{ObjectWithClosedProd2}, ::Type{ObjectWithClosedProd1}
)
    return PreserveTypeProd(ObjectWithClosedProd2)
end

function BayesBase.prod(
    ::PreserveTypeProd{ObjectWithClosedProd1},
    ::ObjectWithClosedProd1,
    ::ObjectWithClosedProd1,
)
    return ObjectWithClosedProd1()
end

function BayesBase.prod(
    ::PreserveTypeProd{ObjectWithClosedProd2},
    ::ObjectWithClosedProd2,
    ::ObjectWithClosedProd2,
)
    return ObjectWithClosedProd2()
end

function BayesBase.prod(
    ::PreserveTypeProd{ObjectWithClosedProd1},
    ::ObjectWithClosedProd1,
    ::ObjectWithClosedProd2,
)
    return ObjectWithClosedProd1()
end

function BayesBase.prod(
    ::PreserveTypeProd{ObjectWithClosedProd2},
    ::ObjectWithClosedProd2,
    ::ObjectWithClosedProd1,
)
    return ObjectWithClosedProd2()
end

function Base.convert(::Type{ObjectWithClosedProd1}, ::ObjectWithClosedProd2)
    return ObjectWithClosedProd1()
end
function Base.convert(::Type{ObjectWithClosedProd2}, ::ObjectWithClosedProd1)
    return ObjectWithClosedProd2()
end

Base.convert(::Type{Int}, ::ObjectWithClosedProd1) = 1
Base.convert(::Type{Int}, ::ObjectWithClosedProd2) = 2

struct ADistributionObject <: ContinuousUnivariateDistribution end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution}, ::ADistributionObject, ::ADistributionObject
)
    return ADistributionObject()
end
