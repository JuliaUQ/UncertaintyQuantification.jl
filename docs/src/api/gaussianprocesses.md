# Gaussian Process Regression

Methods for Gaussian process regression.

## Index

```@index
Pages = ["gaussianprocesses.md"]
```

## Types

```@docs
GaussianProcess
MaximumLikelihoodEstimation
MaximumVariance
ExpectedImprovement
ProbabilityOfImprovement
UpperConfidenceBound
DeviationNumber
ExpectedFeasibility
MaximinDistance
ExpectedImprovementForGlobalFit
AdaptiveGaussianProcess
```

## Functions

```@docs
evaluate!(gp::GaussianProcess, data::DataFrame; mean::Bool=true, variance::Bool=false)
sample!(gp::GaussianProcess, data::DataFrame, n::Int=1)
logpdf(gp::GaussianProcess, data::DataFrame)
evaluate!(gp::AdaptiveGaussianProcess, data::DataFrame; mean::Bool=true, variance::Bool=false)
sample!(gp::AdaptiveGaussianProcess, data::DataFrame, n::Int=1)
refine!(gp::AdaptiveGaussianProcess, n::Int)
logpdf(gp::AdaptiveGaussianProcess, data::DataFrame)
```
