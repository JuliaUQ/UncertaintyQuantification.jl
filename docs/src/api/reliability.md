# Reliability

## Index

```@index
Pages = ["reliability.md"]
```

## Types

```@docs
FORM
DoubleLoop
RandomSlicing
```

## Methods

```@docs
probability_of_failure(models::Union{Vector{<:UQModel},UQModel},performance::Function,inputs::Union{Vector{<:UQInput},UQInput},sim::FORM)
probability_of_failure(models::Union{Vector{<:UQModel},UQModel},performance::Function,inputs::Union{Vector{<:UQInput},UQInput},sim::AbstractMonteCarlo)
probability_of_failure(models::Union{Vector{<:UQModel},UQModel},performance::Function,inputs::Union{Vector{<:UQInput},UQInput},dl::DoubleLoop)
probability_of_failure(models::Union{Vector{<:UQModel},UQModel},performance::Function,inputs::Union{Vector{<:UQInput},UQInput},rs::RandomSlicing)
```
