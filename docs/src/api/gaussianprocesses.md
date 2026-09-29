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
IdentityTransformChoice
ZScoreTransformChoice
UnitRangeTransformChoice
StandardNormalTransformChoice
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
evaluate!(gp::GaussianProcess, data::DataFrame; mode::Symbol = :mean, n_samples::Int = 1)
```
