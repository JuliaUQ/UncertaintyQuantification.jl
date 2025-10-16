# Gaussian Process Regression

Methods for Gaussian process regression.

## Index

```@index
Pages = ["gaussianprocesses.md"]
```

## Types

```@docs
GaussianProcess
NoHyperparameterOptimization
MaximumLikelihoodEstimation
IdentityTransform
ZScoreTransform
UnitRangeTransform
StandardNormalTransform
```

## Functions

```@docs
evaluate!(gp::GaussianProcess, data::DataFrame; mode::Symbol = :mean, n_samples::Int = 1)
```
