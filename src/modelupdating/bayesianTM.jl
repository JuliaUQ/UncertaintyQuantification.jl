# BMU-specific TM implementation, the general implementation is found in `inputs/transportmaps.jl`
"""
    TransportMapBayesian(prior, transportmap, quadrature, islog, transformprior)

Passed to [`bayesianupdating`](@ref) to perform variational inference with a transport map
that maps between the posterior and standard normal space, fitted with [`mapfromdensity`](@ref).
The `prior` is a vector of prior distributions, `transportmap` is the transport map to-be optimized,
and `quadrature` specifies the quadrature points in the standard normal space.
The flag `islog` specifies whether densities are already log-densities.
The flag `transformprior` specifies whether the prior should be transformed to standard normal space.

Alternative constructor

```julia
    TransportMapBayesian(prior, transportmap, quadrature)  # `islog` = true, `transformprior` = true
```

# References

[marzoukSamplingMeasureTransport2016](@cite), [ramgraberTriangularTransport2025](@cite)

See also [`MaximumAPosterioriBayesian`](@ref), [`MaximumLikelihoodBayesian`](@ref), [`TransitionalMarkovChainMonteCarlo`](@ref).

"""
struct TransportMapBayesian <: AbstractBayesianMethod
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    transportmap::AbstractTriangularMap
    quadrature::AbstractQuadratureWeights
    islog::Bool
    transformprior::Bool

    function TransportMapBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        transportmap::AbstractTriangularMap,
        quadrature::AbstractQuadratureWeights,
        islog::Bool=true,
        transformprior::Bool=true,
    )
        @assert length(prior) == size(quadrature.points, 2)
        @assert length(prior) == numberdimensions(transportmap)
        return new(prior, transportmap, quadrature, islog, transformprior)
    end
end

# Setup of the optimization problem for Bayesian updating with transport map
function setupoptimizationproblem(
    prior::Union{Function,Nothing},
    likelihood::Function,
    models::Vector{<:UQModel},
    transportmap::TransportMapBayesian,
    gradient::Union{AbstractADType,Function},
)
    # Transform the prior to be standard normal (in this case, ignore the given prior)
    if transportmap.transformprior
        if !isnothing(prior)
            @warn "Prior function given while transforming to standard normal space. Given prior will be ignored."
        end

        prior = if transportmap.islog
            df -> vec(
                sum(
                    hcat(
                        map(rv -> logpdf.(Normal(), df[:, rv.name]), transportmap.prior)...,
                    );
                    dims=2,
                ),
            )
        else
            df -> vec(
                prod(
                    hcat(map(rv -> pdf.(Normal(), df[:, rv.name]), transportmap.prior)...);
                    dims=2,
                ),
            )
        end

    else

        # if no prior is given, generate prior function from transportmap.prior
        if isnothing(prior)
            prior = if transportmap.islog
                df -> vec(
                    sum(
                        hcat(
                            map(
                                rv -> logpdf.(rv.dist, df[:, rv.name]),
                                transportmap.prior,
                            )...,
                        );
                        dims=2,
                    ),
                )
            else
                df -> vec(
                    prod(
                        hcat(
                            map(rv -> pdf.(rv.dist, df[:, rv.name]), transportmap.prior)...,
                        );
                        dims=2,
                    ),
                )
            end
        end
    end

    logprior = if transportmap.islog
        df -> prior(df)
    else
        df -> log.(prior(df))
    end

    loglikelihood = if transportmap.islog
        df -> likelihood(df)
    else
        df -> log.(likelihood(df))
    end

    rv_names = names(transportmap.prior)
    target_density = x -> begin
        df = _to_dataframe(x, rv_names)

        # evaluate prior (potentially in transformed standard normal space)
        prior_vals = logprior(df)

        # Transform to physical space for model evaluation
        if transportmap.transformprior
            to_physical_space!(transportmap.prior, df)
        end

        if !isempty(models)
            _evaluate!(models, df)
        end

        like_vals = loglikelihood(df)

        val = prior_vals .+ like_vals
        return isa(x, Vector) ? only(val) : val
    end

    if isa(gradient, Function) ||
        isa(gradient, AutoFiniteDiff) ||
        isa(gradient, AutoFiniteDifferences)
        # Analytical gradient or auto-diff with finite differences
        target = MapTargetDensity(target_density, gradient; isvectorized=true)
    else
        @warn "Setting up automatic differentiation. This may take a while."
        target = MapTargetDensity(target_density, gradient, length(transportmap.prior); isvectorized=true)
    end

    return target
end

# Construct map from density for Bayesian updating
function mapfromdensity(
    tm::TransportMapBayesian,
    target::MapTargetDensity,
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    # transform back to physical space if map is fitted in standard normal space
    if tm.transformprior
        return mapfromdensity(
            tm.transportmap,
            target,
            tm.quadrature,
            names(tm.prior),
            tm.prior,
            optimizer,
            optim_options,
        )
    else
        return mapfromdensity(
            tm.transportmap,
            target,
            tm.quadrature,
            names(tm.prior),
            nothing,
            optimizer,
            optim_options,
        )
    end

end

"""
    bayesianupdating(likelihood, models, transportmap, prior, gradient, optimizer, optim_options)

Perform Bayesian updating using transport maps with internal models `Vector{Model}` or `Vector{ParallelModel}`. The `likelihood` is a function that evaluates the likelihood, `models` is a vector of models, and `transportmap` is a [`TransportMapBayesian`](@ref).  The optional `prior` function can be provided (default: `nothing`). The `gradient` can be specified as either an `AbstractADType` or a `Function` (default: `AutoMooncake()`). Uses Mooncake autodiff by default for gradient computation. The `optimizer` specifies the optimization method from Optim.jl (default: `LBFGS()`), and `optim_options` allows passing options to the optimizer (default: `Optim.Options()`). Returns the optimized transport map.

Alternative calls

```julia
    bayesianupdating(likelihood, models, transportmap)  # prior = nothing, gradient = AutoMooncake(), optimizer = LBFGS(), optim_options = Optim.Options()
    bayesianupdating(likelihood, models, transportmap, prior)
    bayesianupdating(likelihood, models, transportmap, prior, gradient)
```

See also [`TransportMapBayesian`](@ref), [`MaximumAPosterioriBayesian`](@ref).
"""
function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    transportmap::TransportMapBayesian,
    prior::Union{Function,Nothing}=nothing,
    gradient::Union{AbstractADType,Function}=AutoMooncake(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(prior, likelihood, models, transportmap, gradient)

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end

"""
    bayesianupdating(likelihood, model, transportmap, prior, gradient, optimizer, optim_options)

Perform Bayesian updating using transport maps with an `ExternalModel`. The `likelihood` is a function that evaluates the likelihood, `model` is an `ExternalModel`, and `transportmap` is a [`TransportMapBayesian`](@ref). The optional `prior` function can be provided (default: `nothing`). The `gradient` can be specified as either an `AbstractADType` or a `Function` (default: `AutoFiniteDiff()`). Uses finite difference approximation by default for external solvers. The `optimizer` specifies the optimization method from Optim.jl (default: `LBFGS()`), and `optim_options` allows passing options to the optimizer (default: `Optim.Options()`). Returns the optimized transport map.

Alternative calls

```julia
    bayesianupdating(likelihood, model, transportmap)  # prior = nothing, gradient = AutoFiniteDiff(), optimizer = LBFGS(), optim_options = Optim.Options()
    bayesianupdating(likelihood, model, transportmap, prior)
    bayesianupdating(likelihood, model, transportmap, prior, gradient)
```

See also [`TransportMapBayesian`](@ref), [`MaximumAPosterioriBayesian`](@ref).
"""
function bayesianupdating(
    likelihood::Function,
    model::ExternalModel,
    transportmap::TransportMapBayesian,
    prior::Union{Function,Nothing}=nothing,
    gradient::Union{AbstractADType,Function}=AutoFiniteDiff(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    optim_options::Optim.Options=Optim.Options(),
)
    target = setupoptimizationproblem(prior, likelihood, [model], transportmap, gradient)

    return mapfromdensity(transportmap, target, optimizer, optim_options)
end

# Helper functions
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

_evaluate!(models::Vector{ExternalModel}, df::DataFrame) = evaluate!(models, df)
