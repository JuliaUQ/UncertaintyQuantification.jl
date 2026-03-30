# Transport Maps

Transport maps for density transformation and variational inference.

## Index

```@index
Pages = ["transportmaps.md"]
```

## Types

```@docs
TransportMap
TransportMapFromSamples
```

## Functions

```@docs
mapfromdensity
mapfromsamples
to_physical_space!(tm::TransportMap, Z::DataFrame)
to_physical_space!(tm::TransportMapFromSamples, Z::DataFrame)
to_standard_normal_space!(tm::TransportMap, X::DataFrame)
to_standard_normal_space!(tm::TransportMapFromSamples, X::DataFrame)
sample(tm::AbstractTransportMap, n::Integer=1)
pdf(tm::TransportMap, x::AbstractVector{<:Real})
pdf(tm::TransportMap, X::AbstractMatrix{<:Real})
pdf(tm::TransportMapFromSamples, x::AbstractVecOrMat{<:Real})
logpdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
mean(tm::TransportMap, quad::AbstractQuadratureWeights=SparseSmolyakWeights)
mean(tm::TransportMapFromSamples, quad::AbstractQuadratureWeights=SparseSmolyakWeights)
var(tm::TransportMap, quad::AbstractQuadratureWeights=SparseSmolyakWeights)
var(tm::TransportMapFromSamples, quad::AbstractQuadratureWeights=SparseSmolyakWeights)
std(tm::TransportMap, quad::AbstractQuadratureWeights=SparseSmolyakWeights)
std(tm::TransportMapFromSamples, quad::AbstractQuadratureWeights=SparseSmolyakWeights)
median(tm::TransportMap)
median(tm::TransportMapFromSamples)
variancediagnostic
```
