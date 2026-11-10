# Models

## Index

```@index
Pages = ["models.md"]
```

## Types

```@docs
UQModel
Model
LinearBasisFunctionModel
```

## Methods

```@docs
evaluate!(m::Model, df::DataFrame)
reliability(ipm::IntervalPredictorModel, ϵ::Real)
isimprecise(i::UQModel)
isimprecise(inputs::AbstractVector{<:UQModel})
isimprecise(inputs::AbstractVector{<:UQInput},models::AbstractVector{<:UQModel})
```
