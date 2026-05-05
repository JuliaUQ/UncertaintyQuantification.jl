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
```

## Functions

```@docs
evaluate!(gp::GaussianProcess, data::DataFrame; mode::Symbol = :mean, n_samples::Int = 1)
optimize_hyperparameters(gp_model::GaussianProcess, optimization::UncertaintyQuantification.AbstractHyperparameterOptimization)
with_gaussian_noise(gp::AbstractGPs.GP, σ²::Real)
```
