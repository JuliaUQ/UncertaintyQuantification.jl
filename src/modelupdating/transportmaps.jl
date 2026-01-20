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
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tm::TransportMap)
    println(io, "Transport Map:")
    println(io, "  Map: $(tm.map)")
    println(io, "  Target: $(tm.target)")
    println(io, "  Quadrature: $(tm.quadrature)")
    println(io, "  Names: $(tm.names)")
    return nothing
end

# todo: add variance diagnostic
# todo: add map-from-samples and adaptive map
# todo: maybe have a more general implementation in a different file and here only bmu-specific interface?


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
    logprior::Function,
    loglikelihood::Function,
    names::Vector{Symbol},
)
    df = _to_dataframe(x, names)

    _evaluate!(model, df)

    val = logprior(df) + loglikelihood(df)
    return isa(x, Vector) ? only(val) : val
end

# Overloaded `TransportMaps.logpdf` to use matrix-valued evaluation of model
logpdf(density::TransportMaps.MapTargetDensity, X::Matrix{<:Real}) = density.logdensity(X)

# Internal implementation
function _transportmap_updating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, #! needs to be adjusted, so far only
    likelihood::Function,
    model::UQModel,
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights,
    autodiff_backend::AbstractADType,
)
    # Check that all RandomVariables have univariate distributions
    if !all(rv -> rv.dist isa UnivariateDistribution, prior)
        error(
            "All prior distributions must be univariate. Multivariate distributions are not supported.",
        )
    end

    posterior =
        x -> _logposterior(x, model, df -> _logprior(df, prior), likelihood, names(prior))

    @warn "Initializing model with $(autodiff_backend)."

    # Create target density
    if !isa(autodiff_backend, AutoFiniteDiff)
        target = MapTargetDensity(posterior, autodiff_backend, length(prior))
    else
        target = MapTargetDensity(posterior, autodiff_backend)
    end

    @warn "Starting Map Optimization..."
    optimize!(transportmap, target, quadrature)

    return TransportMap(transportmap, target, quadrature, names(prior))
end

"""
    bayesianupdating(prior, loglikelihood, model::Model, transportmap, quadrature; autodiff_backend=AutoMooncake())

Perform Bayesian updating using transport maps with a `Model`.

Uses Mooncake autodiff by default for gradient computation.
"""
function bayesianupdating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
    likelihood::Function,
    model::Model,
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights;
    autodiff_backend::AbstractADType=AutoMooncake(),
)
    return _transportmap_updating(
        prior, likelihood, model, transportmap, quadrature, autodiff_backend
    )
end

"""
    bayesianupdating(prior, loglikelihood, model::ParallelModel, transportmap, quadrature; autodiff_backend=AutoMooncake())

Perform Bayesian updating using transport maps with a `ParallelModel`.

Uses Mooncake autodiff by default for gradient computation.
"""
function bayesianupdating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
    likelihood::Function,
    model::ParallelModel,
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights;
    autodiff_backend::AbstractADType=AutoMooncake(),
)
    return _transportmap_updating(
        prior, likelihood, model, transportmap, quadrature, autodiff_backend
    )
end

"""
    bayesianupdating(prior, loglikelihood, model::ExternalModel, transportmap, quadrature; autodiff_backend=AutoFiniteDiff())

Perform Bayesian updating using transport maps with an `ExternalModel`.

Uses finite difference approximation by default since external models typically don't support autodiff.
"""
function bayesianupdating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
    likelihood::Function,
    model::ExternalModel,
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights;
    autodiff_backend::AbstractADType=AutoFiniteDiff(),
)
    return _transportmap_updating(
        prior, likelihood, model, transportmap, quadrature, autodiff_backend
    )
end
