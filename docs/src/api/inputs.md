# Inputs

## Index

```@index
Pages = ["inputs.md"]
```

## Types

```@docs
Parameter
Interval
MomentConstraints
AbstractProbabilityBox
ProbabilityBox
MomentProbabilityBox
RandomVariable
IntervalVariable
EmpiricalDistribution
JointDistribution
```

## Functions

```@docs
UncertaintyQuantification.check_moments_and_range
UncertaintyQuantification.chebyshev_cdf
UncertaintyQuantification.chebyshev_quantile
UncertaintyQuantification.cantelli_cdf
UncertaintyQuantification.cantelli_quantile
UncertaintyQuantification.markov_cdf
UncertaintyQuantification.markov_quantile
UncertaintyQuantification.meanvarminmax_cdf
UncertaintyQuantification.meanvarminmax_quantile
sample(rv::RandomVariable, n::Integer=1)
sample(inputs::Vector{<:UQInput}, n::Integer=1)
GaussianMixtureModel
isimprecise(i::UQInput)
isimprecise(inputs::AbstractVector{<:UQInput})
```
