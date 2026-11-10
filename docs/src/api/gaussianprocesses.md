# Gaussian Process Regression

Methods for Gaussian process regression.

## Index

```@index
Pages = ["gaussianprocesses.md"]
```

## Types

```@docs
GaussianProcess
AdaptiveGaussianProcess
MaximumLikelihoodEstimation
```

## Functions

```@docs
evaluate!(gp::GaussianProcess, data::DataFrame; mean::Bool=true, variance::Bool=false)
sample!(gp::GaussianProcess, data::DataFrame, n::Int=1)
logpdf(gp::GaussianProcess, data::DataFrame)
evaluate!(agp::AdaptiveGaussianProcess, data::DataFrame; mean::Bool=true, variance::Bool=false)
sample!(agp::AdaptiveGaussianProcess, data::DataFrame, n::Int=1)
refine!(agp::AdaptiveGaussianProcess, n::Int)
logpdf(agp::AdaptiveGaussianProcess, data::DataFrame)
```

## Aqcuisition Functions

```@docs
MaximumVariance
ExpectedImprovement
ProbabilityOfImprovement
UpperConfidenceBound
DeviationNumber
ExpectedFeasibility
MaximinDistance
ExpectedImprovementForGlobalFit
```
