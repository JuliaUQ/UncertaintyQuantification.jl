# Inputs

## Index

```@index
Pages = ["inputs.md"]
```

## Types

```@docs
Parameter
Interval
ProbabilityBox
RandomVariable
IntervalVariable
EmpiricalDistribution
JointDistribution
```

## Functions

```@docs
sample(rv::RandomVariable, n::Integer=1)
sample(inputs::Vector{<:UQInput}, n::Integer=1)
GaussianMixtureModel
```
