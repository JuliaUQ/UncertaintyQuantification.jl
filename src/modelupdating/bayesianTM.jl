"""
    TransportMapBayesian(prior, transportmap, quadrature, gradient, optimizer, optim_options, islog, transformprior)

Passed to [`bayesianupdating`](@ref) to perform variational inference with a transport map
that maps between the posterior and standard normal space, fitted with [`mapfromdensity`](@ref).
The `prior` is a vector of prior distributions, `transportmap` is the transport map to-be optimized,
and `quadrature` specifies the quadrature points in the standard normal space.
The `gradient` can be specified as either an `AbstractADType` or a `Function` (default: `AutoFiniteDiff()`).
The `optimizer` specifies the optimization method from Optim.jl (default: `LBFGS()`), and
`optim_options` allows passing options to the optimizer (default: `Optim.Options()`).
The flag `islog` specifies whether densities are already log-densities.
The flag `transformprior` specifies whether the prior should be transformed to standard normal space.

Alternative constructor

```julia
TransportMapBayesian(prior, transportmap, quadrature)  # gradient = AutoMooncake(), optimizer = LBFGS(), optim_options = Optim.Options(), `islog` = true, `transformprior` = true,
```

# References

[marzoukSamplingMeasureTransport2016](@cite), [ramgraberTriangularTransport2025](@cite)

See also [`MaximumAPosterioriBayesian`](@ref), [`MaximumLikelihoodBayesian`](@ref), [`TransitionalMarkovChainMonteCarlo`](@ref).

"""
struct TransportMapBayesian <: AbstractBayesianMethod
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    transportmap::AbstractTriangularMap
    quadrature::AbstractQuadratureWeights
    gradient::Union{AbstractADType,Function}
    optimizer::Optim.AbstractOptimizer
    optim_options::Optim.Options
    islog::Bool
    transformprior::Bool

    function TransportMapBayesian(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        transportmap::AbstractTriangularMap,
        quadrature::AbstractQuadratureWeights,
        gradient::Union{AbstractADType,Function}=AutoFiniteDiff(),
        optimizer::Optim.AbstractOptimizer=LBFGS(),
        optim_options::Optim.Options=Optim.Options();
        islog::Bool=true,
        transformprior::Bool=true,
    )
        @assert length(prior) == size(quadrature.points, 2)
        @assert length(prior) == numberdimensions(transportmap)
        return new(
            prior,
            transportmap,
            quadrature,
            gradient,
            optimizer,
            optim_options,
            islog,
            transformprior,
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", tm::TransportMapBayesian)
    println(io, "TransportMapBayesian")
    println(io, "  Variable names: ", names(tm.prior), ", ")
    println(io, "  Map: ", tm.transportmap)
    println(io, "  Quadrature: ", tm.quadrature)
    println(io, "  Gradient: ", typeof(tm.gradient))
    println(io, "  Optimizer: ", typeof(tm.optimizer))
    println(io, "  Log-density: ", tm.islog)
    print(io, "  Transform prior: ", tm.transformprior)
    return nothing
end

function Base.show(io::IO, tm::TransportMapBayesian)
    print(io, "TransportMapBayesian(variables=", names(tm.prior), ", ")
    print(io, size(tm.quadrature.points, 1), " quad pts, ")
    print(io, "grad=", typeof(tm.gradient).name.name, ", ")
    print(io, "optim=", typeof(tm.optimizer).name.name, ")")
    return nothing
end

# Setup of the optimization problem for Bayesian updating with transport map
function setupoptimizationproblem(
    prior::Function,
    likelihood::Function,
    models::Vector{<:UQModel},
    tm::TransportMapBayesian,
)
    # Check gradient estimation and optimizer for external model
    if !isempty(models)
        if any([isa(m, ExternalModel) for m in models])
            @assert (
                isa(tm.gradient, Function) ||
                isa(tm.gradient, AutoFiniteDiff) ||
                isa(tm.gradient, AutoFiniteDifferences)
            ) "Unsupported gradient with ExternalModel. Supported are: AutoFiniteDiff, AutoFiniteDifferences or custom function."
            if isa(tm.optimizer, Optim.FirstOrderOptimizer)
                @warn "Using gradient-based optimizer with external model."
            end
        end
    end

    # Check supported gradient estimation
    @assert (
        isa(tm.gradient, Function) ||
        isa(tm.gradient, AutoFiniteDiff) ||
        isa(tm.gradient, AutoFiniteDifferences) ||
        isa(tm.gradient, AutoMooncake)
    ) "Unsupported gradient. Supported are: AutoFiniteDiff, AutoFiniteDifferences, AutoMooncake or custom function."

    logprior = if tm.islog
        df -> prior(df)
    else
        df -> log.(prior(df))
    end

    loglikelihood = if tm.islog
        df -> likelihood(df)
    else
        df -> log.(likelihood(df))
    end

    rv_names = names(tm.prior)
    target_density = x -> begin
        df = _to_dataframe(x, rv_names)

        # evaluate prior (potentially in transformed standard normal space)
        prior_vals = logprior(df)

        # Transform to physical space for model evaluation
        if tm.transformprior
            to_physical_space!(tm.prior, df)
        end

        if !isempty(models)
            _evaluate!(models, df)
        end

        like_vals = loglikelihood(df)

        val = prior_vals .+ like_vals
        return isa(x, Vector) ? only(val) : val
    end

    if (
        isa(tm.gradient, Function) ||
        isa(tm.gradient, AutoFiniteDiff) ||
        isa(tm.gradient, AutoFiniteDifferences)
    ) || isa(tm.optimizer, Optim.ZerothOrderOptimizer)
        # Analytical gradient, finite differences or gradient-free optimization
        target = MapTargetDensity(
            target_density, tm.gradient; isvectorized=true, threaded=false
        )
    else
        @warn "Setting up automatic differentiation. This may take a while."
        target = MapTargetDensity(
            target_density, tm.gradient, length(tm.prior); isvectorized=true, threaded=false
        )
    end

    return target
end

# Construct map from density for Bayesian updating
function mapfromdensity(tm::TransportMapBayesian, target::MapTargetDensity)
    # transform back to physical space if map is fitted in standard normal space
    if tm.transformprior
        return mapfromdensity(
            tm.transportmap,
            target,
            tm.quadrature,
            names(tm.prior),
            tm.prior,
            tm.optimizer,
            tm.optim_options,
        )
    else
        return mapfromdensity(
            tm.transportmap,
            target,
            tm.quadrature,
            names(tm.prior),
            nothing,
            tm.optimizer,
            tm.optim_options,
        )
    end
end

"""
    bayesianupdating(prior, likelihood, models, tm)

Perform bayesian updating using the given `prior`, `likelihood`, `models` and `tm` [`TransportMapBayesian`](@ref).
Returns the optimized [`TransportMap](@ref) which can be used to evaluate the posterior pdf and to generate samples.

Alternatively, the `prior` function can be omitted and the prior is constructed from the given vector of random variables in `tm`.

```julia
bayesianupdating(likelihood, models, tm)
```

See also [`TransportMapBayesian`](@ref), [`MaximumAPosterioriBayesian`](@ref), [`TransportMap](@ref).
"""
function bayesianupdating(
    prior::Function,
    likelihood::Function,
    models::Vector{<:UQModel},
    tm::TransportMapBayesian,
)
    if tm.transformprior
        @warn "Prior function given while transforming to standard normal prior. Given prior will be ignored."
        prior = if tm.islog
            df -> vec(
                sum(
                    hcat(map(rv -> logpdf.(Normal(), df[:, rv.name]), tm.prior)...);
                    dims=2,
                ),
            )
        else
            df -> vec(
                prod(hcat(map(rv -> pdf.(Normal(), df[:, rv.name]), tm.prior)...); dims=2),
            )
        end
    end

    target = setupoptimizationproblem(prior, likelihood, models, tm)

    return mapfromdensity(tm, target)
end

function bayesianupdating(
    likelihood::Function, models::Vector{<:UQModel}, tm::TransportMapBayesian
)

    # Transform the prior to be standard normal
    if tm.transformprior
        prior = if tm.islog
            df -> vec(
                sum(
                    hcat(map(rv -> logpdf.(Normal(), df[:, rv.name]), tm.prior)...);
                    dims=2,
                ),
            )
        else
            df -> vec(
                prod(hcat(map(rv -> pdf.(Normal(), df[:, rv.name]), tm.prior)...); dims=2),
            )
        end

    else
        prior = if tm.islog
            df -> vec(
                sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), tm.prior)...); dims=2),
            )
        else
            df -> vec(prod(hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), tm.prior)...); dims=2))
        end
    end

    target = setupoptimizationproblem(prior, likelihood, models, tm)

    return mapfromdensity(tm, target)
end

# Helper functions to enable auto-diff with Mooncake
function _evaluate!(m::ParallelModel, df::DataFrame)
    df[!, m.name] = map(m.func, eachrow(df))
    return nothing
end

_evaluate!(m::UQModel, df::DataFrame) = evaluate!(m, df)
_evaluate!(m::ExternalModel, df::DataFrame) = evaluate!(m, df)

function _evaluate!(models::Vector{<:UQModel}, df::DataFrame)
    for m in models
        _evaluate!(m, df)
    end
    return nothing
end

_evaluate!(models::Vector{ExternalModel}, df::DataFrame) = evaluate!(models, df)
