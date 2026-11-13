"""
    MaximumAPosterioriBayesian(
        x0::DataFrame; islog = true, lowerbounds, upperbounds,
        filtertolerance = 1.0e-6, optimizer = Optim.LBFGS(),
        optimoptions = Optim.Options()
    )

Configure a maximum a posteriori (MAP) estimate. Each row of `x0` is a named
starting point. Pass the prior separately to [`bayesianupdating`](@ref).
`islog` specifies whether the likelihood returns log densities. Bounds default
to unbounded intervals, with one entry per parameter. Modes closer than
`filtertolerance` are merged; set it to zero to retain every result.
`optimizer` and `optimoptions` configure Optim.jl.

For maximum likelihood estimation without a prior, use
[`MaximumLikelihoodEstimate`](@ref) with [`maximumlikelihood`](@ref).
"""
struct MaximumAPosterioriBayesian <: AbstractBayesianPointEstimate
    x0::DataFrame
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}
    filtertolerance::Float64
    optimizer::Optim.AbstractOptimizer
    optimoptions::Optim.Options

    function MaximumAPosterioriBayesian(
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
    bayesianupdating(
        prior::JointDistribution, likelihood,
        estimate::MaximumAPosterioriBayesian, models = UQModel[]
    )

Find local maxima of the posterior. The likelihood accepts a DataFrame and
returns one value per row, using log densities when `estimate.islog` is true.
Any supplied models are evaluated before the likelihood. Optimization settings
are stored in `estimate`.

Returns a DataFrame with parameter values and either a `logMAP` column
(log-likelihood input) or a `MAP` column (density input). These values are the
unnormalized posterior log densities or densities at the estimated modes.
"""
function bayesianupdating(
        prior::JointDistribution, likelihood::Function,
        estimate::MaximumAPosterioriBayesian,
        models::Vector{<:UQModel} = UQModel[],
    )
    issetequal(names(prior), Symbol.(names(estimate.x0))) || throw(ArgumentError("Starting-point columns must match the prior variables."))
    log_posterior = _log_posterior(prior, likelihood, models, estimate.islog)
    return _maximize_logdensity(log_posterior, estimate, estimate.islog ? :logMAP : :MAP)
end
