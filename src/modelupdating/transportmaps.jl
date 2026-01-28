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
    models::Vector{<:UQModel},
    transportmap::TransportMapBMU,
    gradient::Union{AbstractADType,Function},
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

        if !isempty(models)
            _evaluate!(models, df)
        end

        val = posterior(df)
        return isa(x, Vector) ? only(val) : val
    end

    if isa(gradient, Function) ||
        isa(gradient, AutoFiniteDiff) ||
        isa(gradient, AutoFiniteDifferences)
        # Analytical gradient or auto-diff with finite differences
        target = MapTargetDensity(target_density, gradient)
    else
        @warn "Setting up automatic differentiation. This may take a while."
        target = MapTargetDensity(target_density, gradient, length(transportmap.prior))
    end

    return target
end

# New version of `evaluate!` to use `map` instead of `pmap` for `ParallelModel`
function _evaluate!(m::ParallelModel, df::DataFrame)
    df[!, m.name] = map(m.func, eachrow(df))
    return nothing
end

_evaluate!(m::UQModel, df::DataFrame) = evaluate!(m, df)

function _evaluate!(models::Vector{<:UQModel}, df::DataFrame)
    for m in models
        _evaluate!(m, df)
    end
    return nothing
end

#! Test this
_evaluate!(models::Vector{ExternalModel}, df::DataFrame) = evaluate!(models, df)

# Overloaded `logpdf` to use matrix-valued evaluation of model
TransportMaps.logpdf(density::MapTargetDensity, X::Matrix{<:Real}) = density.logdensity(X)

# todo: Implement custom finite-diff when using external model (writing files and stuff)
# function TransportMaps.grad_logpdf(density::MapTargetDensity, X::Matrix{<:Real})

#     # Custem AutoDiff implementation when using the external model to deal with files,
#     # specially with the creation of folders, input and output files (not-unique)
#     if isa(density.ad_backend, AutoFiniteDiff) || isa(density.ad_backend, AutoFiniteDifferences)


#     end

# end

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
    bayesianupdating(prior, loglikelihood, models, transportmap, quadrature; gradient=AutoMooncake())

Perform Bayesian updating using transport maps with internal models `Vector{Model}` or `Vector{ParallelModel}`.

Uses Mooncake autodiff by default for gradient computation.
"""
function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    transportmap::TransportMapBMU,
    prior::Union{Function,Nothing}=nothing,
    gradient::Union{AbstractADType,Function}=AutoMooncake(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(prior, likelihood, models, transportmap, gradient)

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end

"""
    bayesianupdating(prior, loglikelihood, model::ExternalModel, transportmap, quadrature; gradient=AutoFiniteDiff())

Perform Bayesian updating using transport maps with an `ExternalModel`.

Uses finite difference approximation by default for external solvers.
"""
function bayesianupdating(
    likelihood::Function,
    model::ExternalModel,
    transportmap::TransportMapBMU,
    prior::Union{Function,Nothing}=nothing,
    gradient::Union{AbstractADType,Function}=AutoFiniteDiff(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(prior, likelihood, [model], transportmap, gradient)

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end
