"""
    TransportMap

A transport map for Bayesian updating, used to transform between standard normal space and physical space.

# Fields

- `map::TransportMaps.PolynomialMap`: The polynomial transport map.
- `target::TransportMaps.MapTargetDensity`: The target density.
- `quadrature::TransportMaps.AbstractQuadratureWeights`: The quadrature weights used in optimization.
- `names::Vector{Symbol}`: The names of the variables.
"""
struct TransportMap <: AbstractTransportMap
    map::TransportMaps.PolynomialMap
    target::TransportMaps.MapTargetDensity
    quadrature::TransportMaps.AbstractQuadratureWeights
    names::Vector{Symbol}
end

function _to_dataframe(X::AbstractMatrix{<:Real}, names::Vector{Symbol})
    return DataFrame(X, names)
end

function _to_dataframe(x::AbstractVector{<:Real}, names::Vector{Symbol})
    return DataFrame(permutedims(x), names)
end

function evaluate(tm::TransportMap, Z::Matrix{<:Real})
    return TransportMaps.evaluate(tm.map, Z)
end

function inverse(tm::TransportMap, X::Matrix{<:Real})
    return TransportMaps.inverse(tm.map, X)
end

function _logprior(df::DataFrame, prior::Vector{<:RandomVariable{<:UnivariateDistribution}})
    return vec(sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), prior)...); dims=2))
end

function _logposterior(
    x::AbstractVecOrMat{<:Real},
    model::UQModel,
    prior::Function,
    loglikelihood::Function,
    names::Vector{Symbol},
)
    df = _to_dataframe(x, names)

    evaluate!(model, df)

    val = prior(df) + loglikelihood(df)
    return isa(x, Vector) ? only(val) : val
end

# New version of `evaluate!` to use `map` instead of `pmap` for `ParallelModel`
function _evaluate!(m::ParallelModel, df::DataFrame)
    df[!, m.name] = map(m.func, eachrow(df))
    return nothing
end

# Version for parallel model, since we can't use `evaluate!` with `pmap` with `Mooncake`
function _logposterior(
    x::AbstractVecOrMat{<:Real},
    model::ParallelModel,
    prior::Function,
    loglikelihood::Function,
    names::Vector{Symbol},
)
    df = _to_dataframe(x, names)

    _evaluate!(model, df)

    val = prior(df) + loglikelihood(df)
    return isa(x, Vector) ? only(val) : val
end

# Overloaded `TransportMaps.logpdf` to use matrix-valued evaluation of model
logpdf(density::TransportMaps.MapTargetDensity, X::Matrix{<:Real}) = density.logdensity(X)

"""
    bayesianupdating(prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, loglikelihood::Function, model::UQModel, transportmap::TransportMaps.PolynomialMap, quadrature::TransportMaps.AbstractQuadratureWeights)

Perform Bayesian updating using transport maps to construct an optimized map for transforming between standard normal space and physical space.

This method optimizes a polynomial transport map to approximate the posterior distribution based on the given prior, log-likelihood, and model.

# Arguments

- `prior::Vector{<:RandomVariable{<:UnivariateDistribution}}`: A vector of random variables defining the prior distributions.
- `loglikelihood::Function`: A function that computes the log-likelihood for a DataFrame of samples.
- `model::UQModel`: The model to evaluate during updating.
- `transportmap::TransportMaps.PolynomialMap`: The initial polynomial transport map to optimize.
- `quadrature::TransportMaps.AbstractQuadratureWeights`: The quadrature weights used in the optimization process.

# Returns

A `TransportMap` object containing the optimized map, target density, quadrature, and variable names.

### Notes

`loglikelihood` must be a Julia function defined in terms of a `DataFrame` of samples, evaluating the log-likelihood for each row.

For example, a log-likelihood based on a normal distribution with observed data `Data`:

```julia
loglikelihood(df) = [sum(logpdf.(Normal.(df_i.x, 1), Data)) for df_i in eachrow(df)]
```
If model evaluation is required, ensure `model`` is provided and compatible with the inputs.
"""
function bayesianupdating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, #! make this work for different kinds of inputs!
    likelihood::Function,
    model::UQModel, #! only works with one model
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights,
)
    # Check that all RandomVariables have univariate distributions
    if !all(rv -> rv.dist isa UnivariateDistribution, prior)
        error(
            "All prior distributions must be univariate. Multivariate distributions are not supported.",
        )
    end

    posterior =
        x -> _logposterior(x, model, df -> _logprior(df, prior), likelihood, names(prior))

    if model isa Model
        # If model is `Model`: use auto diff with Mooncake and prepared gradient
        autodiff_backend = AutoMooncake()
        @warn "Initializing model with Mooncake. This may take a while."
        target = MapTargetDensity(posterior, autodiff_backend, length(prior))

    elseif model isa ExternalModel
        # If model is `ExternalModel`: use finite differences
        autodiff_backend = AutoFiniteDiff()
        @warn "Using finite difference approximation."
        target = MapTargetDensity(posterior, autodiff_backend)

    elseif model isa ParallelModel
        # If model is `ParallelModel`: use auto diff with Mooncake and prepared gradient
        autodiff_backend = AutoMooncake()
        @warn "Initializing model with Mooncake. This may take a while."
        target = MapTargetDensity(posterior, autodiff_backend, length(prior))
    end

    @warn "Starting Map Optimization..."

    optimize!(transportmap, target, quadrature)

    return TransportMap(transportmap, target, quadrature, names(prior))
end

function sample(tm::TransportMap, n::Integer=1)
    Z = randn(n, numberdimensions(tm.map))
    X = evaluate(tm, Z)
    return _to_dataframe(X, tm.names)
end

function to_physical_space!(tm::TransportMap, Z::DataFrame)
    X = evaluate(tm, Matrix(Z[!, tm.names]))
    Z[!, tm.names] .= X
    return nothing
end

function to_standard_normal_space!(tm::TransportMap, X::DataFrame)
    Z = inverse(tm, Matrix(X[!, tm.names]))
    X[!, tm.names] .= Z
    return nothing
end

function Base.show(io::IO, tm::TransportMap)
    print(io, "TransportMap(")
    print(io, "map=$(tm.map), ")
    print(io, "target=$(tm.target), ")
    print(io, "quadrature=$(tm.quadrature), ")
    print(io, "names=$(tm.names)")
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", tm::TransportMap)
    println(io, "Transport Map:")
    println(io, "  Map: $(tm.map)")
    println(io, "  Target: $(tm.target)")
    println(io, "  Quadrature: $(tm.quadrature)")
    return println(io, "  Names: $(tm.names)")
end
