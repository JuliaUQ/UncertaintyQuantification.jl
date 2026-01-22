# BMU-specific TM implementation, the general implementation is found in `inputs/transportmaps.jl`
struct TransportMapBMU <: AbstractBayesianMethod #! this needs a better name...
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    transportmap::AbstractTriangularMap
    quadrature::AbstractQuadratureWeights
    islog::Bool

    function TransportMapBMU(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        transportmap::AbstractTriangularMap,
        quadrature::AbstractQuadratureWeights,
        islog::Bool=true,
    )
        @assert length(prior) == size(quadrature.points, 2) #! make `TransportMaps.numberdimensions(quad::AbstractQuadratureWeights)`
        @assert length(prior) == numberdimensions(transportmap)
        return new(prior, transportmap, quadrature, islog)
    end
end

function setupoptimizationproblem(
    prior::Union{Function,Nothing},
    likelihood::Function,
    model::UQModel, #! make work with Vector{<:UQModel}
    transportmap::TransportMapBMU,
    autodiff_backend::AbstractADType,
)

    # if no prior is given, generate prior function from transportmap.prior
    if isnothing(prior)
        prior = if transportmap.islog
            df -> vec(
                sum(
                    hcat(
                        map(rv -> logpdf.(rv.dist, df[:, rv.name]), transportmap.prior)...,
                    );
                    dims=2,
                ),
            )
        else
            df -> vec(
                prod(
                    hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), transportmap.prior)...);
                    dims=2,
                ),
            )
        end
    end

    posterior = if transportmap.islog
        df -> prior(df) .+ likelihood(df)
    else
        df -> log.(prior(df)) .+ log.(likelihood(df))
    end

    rv_names = names(transportmap.prior)
    target_density = x -> begin
        df = _to_dataframe(x, rv_names)
        _evaluate!(model, df)

        val = posterior(df)
        return isa(x, Vector) ? only(val) : val
    end

    # Create target density
    if autodiff_backend ∉ [AutoFiniteDiff, AutoFiniteDifferences]
        target = MapTargetDensity(
            target_density, autodiff_backend, length(transportmap.prior)
        )
    else
        target = MapTargetDensity(target_density, autodiff_backend)
    end

    return target
end

# New version of `evaluate!` to use `map` instead of `pmap` for `ParallelModel`
function _evaluate!(m::ParallelModel, df::DataFrame)
    df[!, m.name] = map(m.func, eachrow(df))
    return nothing
end

_evaluate!(m::UQModel, df::DataFrame) = evaluate!(m, df)

# Overloaded `logpdf` to use matrix-valued evaluation of model
logpdf(density::MapTargetDensity, X::Matrix{<:Real}) = density.logdensity(X)

function mapfromdensity(
    tm::TransportMapBMU,
    target::MapTargetDensity,
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    return mapfromdensity(
        tm.transportmap, target, tm.quadrature, names(tm.prior), optimizer, optim_options
    )
end

"""
    bayesianupdating(prior, loglikelihood, model::Model, transportmap, quadrature; autodiff_backend=AutoMooncake())

Perform Bayesian updating using transport maps with a `Model`.

Uses Mooncake autodiff by default for gradient computation.
"""
function bayesianupdating(
    likelihood::Function,
    model::Model, #! make work for Vector{Model}
    transportmap::TransportMapBMU,
    prior::Union{Function,Nothing}=nothing,
    autodiff_backend::AbstractADType=AutoMooncake(),#! maybe this should be moved to `TransportMapBMU`?
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(
        prior, likelihood, model, transportmap, autodiff_backend
    )

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end

"""
    bayesianupdating(prior, loglikelihood, model::ParallelModel, transportmap, quadrature; autodiff_backend=AutoMooncake())

Perform Bayesian updating using transport maps with a `ParallelModel`.

Uses Mooncake autodiff by default for gradient computation.
"""
function bayesianupdating(
    likelihood::Function,
    model::ParallelModel,
    transportmap::TransportMapBMU,
    prior::Union{Function,Nothing}=nothing,
    autodiff_backend::AbstractADType=AutoMooncake(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(
        prior, likelihood, model, transportmap, autodiff_backend
    )

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end

"""
    bayesianupdating(prior, loglikelihood, model::ExternalModel, transportmap, quadrature; autodiff_backend=AutoFiniteDiff())

Perform Bayesian updating using transport maps with an `ExternalModel`.

Uses finite difference approximation by default since external models typically don't support autodiff.
"""
function bayesianupdating(
    likelihood::Function,
    model::ExternalModel,
    transportmap::TransportMapBMU,
    prior::Union{Function,Nothing}=nothing,
    autodiff_backend::AbstractADType=AutoFiniteDiff(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(
        prior, likelihood, model, transportmap, autodiff_backend
    )

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end
