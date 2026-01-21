# BMU-specific TM implementation, the general implementation is found in `inputs/transportmaps.jl`

# Todo: Look into MAP stuff from Jan and setup in a similar manner (maybe merge files)

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

# Overloaded `logpdf` to use matrix-valued evaluation of model
logpdf(density::MapTargetDensity, X::Matrix{<:Real}) = density.logdensity(X)

# Internal implementation
function _transportmap_updating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, #! needs to be adjusted, so far only
    likelihood::Function,
    model::UQModel,
    transportmap::PolynomialMap,
    quadrature::AbstractQuadratureWeights,
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

    return mapfromdensity(transportmap, target, quadrature, names(prior))
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
    transportmap::PolynomialMap,
    quadrature::AbstractQuadratureWeights;
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
    transportmap::PolynomialMap,
    quadrature::AbstractQuadratureWeights;
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
    transportmap::PolynomialMap,
    quadrature::AbstractQuadratureWeights;
    autodiff_backend::AbstractADType=AutoFiniteDiff(),
)
    return _transportmap_updating(
        prior, likelihood, model, transportmap, quadrature, autodiff_backend
    )
end
