"""
    MaximumLikelihoodEstimate(
        x0::DataFrame; islog = true, lowerbounds, upperbounds,
        filtertolerance = 1.0e-6, optimizer = Optim.LBFGS(),
        optimoptions = Optim.Options()
    )

Configure [`maximumlikelihood`](@ref). Each row of `x0` is an optimization start,
with one named parameter per column. No prior is required. `islog` specifies
whether the likelihood returns log densities. Bounds default to unbounded
intervals, with one entry per parameter. Modes closer than `filtertolerance`
are merged; set it to zero to keep every optimization result.

`optimizer` and `optimoptions` configure Optim.jl. This estimator is separate
from `MaximumLikelihoodEstimation`, which configures Gaussian-process training.
"""
struct MaximumLikelihoodEstimate
    x0::DataFrame
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}
    filtertolerance::Float64
    optimizer::Optim.AbstractOptimizer
    optimoptions::Optim.Options

    function MaximumLikelihoodEstimate(
            x0::DataFrame;
            islog::Bool = true,
            lowerbounds::AbstractVector{<:Real} = fill(-Inf, ncol(x0)),
            upperbounds::AbstractVector{<:Real} = fill(Inf, ncol(x0)),
            filtertolerance::Real = 1.0e-6,
            optimizer::Optim.AbstractOptimizer = Optim.LBFGS(),
            optimoptions::Optim.Options = Optim.Options(),
        )
        starts = _pointestimate_starts(x0, lowerbounds, upperbounds, filtertolerance)
        return new(
            starts, islog, Float64.(lowerbounds), Float64.(upperbounds),
            Float64(filtertolerance), optimizer, optimoptions
        )
    end
end

"""
    maximumlikelihood(likelihood, estimate::MaximumLikelihoodEstimate, models = UQModel[])

Find local maxima of a likelihood without a prior. The likelihood accepts a
DataFrame and returns one value per row, using log densities when
`estimate.islog` is true. Any supplied models are evaluated before the likelihood.
All optimization settings are stored in `estimate`.

Returns a DataFrame containing the parameter values and a `logMLE` column for
log-likelihood input, or an `MLE` column for density input. Results follow the
starting-row order, after merging modes within `estimate.filtertolerance`.

```julia
estimate = MaximumLikelihoodEstimate(DataFrame(μ = [0.0, 2.0]))
result = maximumlikelihood(df -> -0.5 .* (df.μ .- 1) .^ 2, estimate)
```
"""
function maximumlikelihood(
        likelihood::Function, estimate::MaximumLikelihoodEstimate,
        models::Vector{<:UQModel} = UQModel[],
    )
    loglikelihood = df -> begin
        if !isempty(models)
            evaluate!(models, df)
        end
        values = likelihood(df)
        return estimate.islog ? values : log.(values)
    end
    return _maximize_logdensity(loglikelihood, estimate, estimate.islog ? :logMLE : :MLE)
end

# Shared validation and numerical optimization for MLE and MAP. Neither helper
# depends on a prior or a Bayesian estimator type.
function _pointestimate_starts(x0::DataFrame, lowerbounds, upperbounds, tolerance)
    nrow(x0) > 0 && ncol(x0) > 0 || throw(ArgumentError("x0 must contain at least one starting point and parameter."))
    length(lowerbounds) == length(upperbounds) == ncol(x0) || throw(DimensionMismatch("Bounds must have one entry per parameter."))
    all(lowerbounds .< upperbounds) || throw(ArgumentError("Lower bounds must be smaller than upper bounds."))
    isfinite(tolerance) && tolerance >= 0 || throw(ArgumentError("filtertolerance must be finite and nonnegative."))
    starts = Matrix{Float64}(x0)
    all(isfinite, starts) || throw(ArgumentError("Starting points must be finite."))
    all(lowerbounds' .<= starts .<= upperbounds') || throw(ArgumentError("Starting points must lie within the bounds."))
    return DataFrame(starts, names(x0))
end

function _maximize_logdensity(logtarget::Function, estimate, value_column::Symbol)
    variables = Symbol.(names(estimate.x0))
    value_column in variables && throw(ArgumentError("Parameter names must not include the result column $value_column."))
    objective = x -> -only(logtarget(DataFrame(permutedims(x), variables)))
    results = DataFrame(zeros(0, length(variables) + 1), [variables; value_column])
    modes = Vector{Float64}[]
    unbounded = all(==(-Inf), estimate.lowerbounds) && all(==(Inf), estimate.upperbounds)
    for row in eachrow(estimate.x0)
        start = collect(row)
        result = if unbounded
            Optim.optimize(objective, start, estimate.optimizer, estimate.optimoptions)
        else
            Optim.optimize(
                objective, estimate.lowerbounds, estimate.upperbounds,
                start, Optim.Fminbox(estimate.optimizer), estimate.optimoptions
            )
        end
        mode = Optim.minimizer(result)
        logvalue = -Optim.minimum(result)
        isfinite(logvalue) || throw(ArgumentError("The estimated mode must have finite log density."))
        if estimate.filtertolerance > 0 && any(x -> norm(x - mode) < estimate.filtertolerance, modes)
            continue
        end
        push!(modes, copy(mode))
        push!(results, [mode; estimate.islog ? logvalue : exp(logvalue)])
    end
    return results
end
