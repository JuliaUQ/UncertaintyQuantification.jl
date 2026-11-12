# todo: remove this struct
"""
    TransportMapBayesian(
        transportmap, quadrature, gradient = AutoFiniteDiff(),
        optimizer = LBFGS(), optim_options = Optim.Options();
        islog = true, transformprior = true
    )

Configure Bayesian inference using a transport map. Pass the prior separately
as `bayesianupdating(prior, likelihood, method, models=UQModel[])`.
`transportmap` and `quadrature` must have the same dimension. `gradient` may be
an ADTypes backend or a custom function; optimizer settings are stored here.
`islog` specifies whether the likelihood returns log densities.

With `transformprior=true`, the prior must have independent univariate
marginals; optimization takes place in standard normal space and the result
maps back to physical space. For other joint priors, use `transformprior=false`.

# References

[marzoukSamplingMeasureTransport2016](@cite), [ramgraberTriangularTransport2025](@cite)
"""
struct TransportMapBayesian <: AbstractBayesianMethod
    transportmap::AbstractTriangularMap
    quadrature::AbstractQuadratureWeights
    gradient::Union{ADTypes.AbstractADType, Function}
    optimizer::Optim.AbstractOptimizer
    optim_options::Optim.Options
    islog::Bool
    transformprior::Bool

    function TransportMapBayesian(
            transportmap::AbstractTriangularMap,
            quadrature::AbstractQuadratureWeights,
            gradient::Union{ADTypes.AbstractADType, Function} = AutoFiniteDiff(),
            optimizer::Optim.AbstractOptimizer = LBFGS(),
            optim_options::Optim.Options = Optim.Options();
            islog::Bool = true,
            transformprior::Bool = true,
        )
        @assert size(quadrature.points, 2) == numberdimensions(transportmap) "Map and quadrature dimensions must match."
        return new(
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
    println(io, "  Dimensions: ", numberdimensions(tm.transportmap))
    println(io, "  Map: ", tm.transportmap)
    println(io, "  Quadrature: ", tm.quadrature)
    println(io, "  Gradient: ", typeof(tm.gradient))
    println(io, "  Optimizer: ", typeof(tm.optimizer))
    println(io, "  Log-density: ", tm.islog)
    print(io, "  Transform prior: ", tm.transformprior)
    return nothing
end

function Base.show(io::IO, tm::TransportMapBayesian)
    print(io, "TransportMapBayesian(dimensions=", numberdimensions(tm.transportmap), ", ")
    print(io, size(tm.quadrature.points, 1), " quad pts, ")
    print(io, "grad=", typeof(tm.gradient).name.name, ", ")
    print(io, "optim=", typeof(tm.optimizer).name.name, ")")
    return nothing
end

# Setup of the optimization problem for Bayesian updating with transport map
function setupoptimizationproblem(
        prior::JointDistribution,
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

    logprior = _log_prior(prior)

    loglikelihood = if tm.islog
        df -> likelihood(df)
    else
        df -> log.(likelihood(df))
    end

    rv_names = names(prior)
    target_density = x -> begin
        df = _to_dataframe(x, rv_names)

        # evaluate prior (potentially in transformed standard normal space)
        prior_vals = tm.transformprior ? vec(sum(logpdf.(Normal(), Matrix(df[:, rv_names])); dims = 2)) : logprior(df)

        # Transform to physical space for model evaluation
        if tm.transformprior
            to_physical_space!(prior.m, df)
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
            target_density, tm.gradient; isvectorized = true, threaded = false
        )
    else
        @warn "Setting up automatic differentiation. This may take a while."
        target = MapTargetDensity(
            target_density, tm.gradient, length(rv_names); isvectorized = true, threaded = false
        )
    end

    return target
end

"""
    bayesianupdating(
        prior::JointDistribution, likelihood,
        method::TransportMapBayesian, models = UQModel[]
    )

Fit a transport map to the posterior. The argument order and prior handling
match the other Bayesian methods. The likelihood accepts a DataFrame and
returns one value per row, using log densities when `method.islog` is true.
Models are evaluated in physical space before the likelihood. Returns a
[`JointDistribution`](@ref) containing the fitted [`TransportMap`](@ref).
"""
function bayesianupdating(
        prior::JointDistribution,
        likelihood::Function,
        method::TransportMapBayesian,
        models::Vector{<:UQModel} = UQModel[],
    )
    length(names(prior)) == numberdimensions(method.transportmap) || throw(DimensionMismatch("Prior and transport map dimensions must match."))
    transform_density = if method.transformprior
        prior.d isa IndependentCopula && prior.m isa Vector{<:RandomVariable{<:UnivariateDistribution}} ||
            throw(ArgumentError("Prior transformation requires independent univariate marginals; use transformprior=false for other joint priors."))
        prior.m
    else
        nothing
    end
    target = setupoptimizationproblem(prior, likelihood, models, method)
    return mapfromdensity(
        method.transportmap, target, method.quadrature, names(prior),
        transform_density, method.optimizer, method.optim_options
    )
end

# Helper functions to enable auto-diff with Mooncake
_evaluate!(m::UQModel, df::DataFrame) = evaluate!(m, df)
_evaluate!(m::ExternalModel, df::DataFrame) = evaluate!(m, df)

function _evaluate!(models::Vector{<:UQModel}, df::DataFrame)
    for m in models
        _evaluate!(m, df)
    end
    return nothing
end

_evaluate!(models::Vector{ExternalModel}, df::DataFrame) = evaluate!(models, df)
