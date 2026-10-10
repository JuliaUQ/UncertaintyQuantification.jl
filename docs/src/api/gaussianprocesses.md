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
```

## Functions

```@docs
AdaptiveGaussianProcess
evaluate!(gp::GaussianProcess, data::DataFrame; mode::Symbol = :mean)
sample!(gp::GaussianProcess, data::DataFrame, n_samples::Int)
```
