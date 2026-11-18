"""
    bayesianupdating(
        prior::JointDistribution, likelihood, estimate::LaplaceEstimate,
        models = UQModel[]
    )

Approximate the posterior using a general [`LaplaceEstimate`](@ref). The
likelihood accepts a DataFrame and returns one value per row, using log values
when `estimate.islog` is true. Any supplied models are evaluated before the
likelihood. All optimization and Hessian settings are stored in `estimate`.
Returns a [`JointDistribution`](@ref) with a Gaussian mixture approximation.
"""
function bayesianupdating(
        prior::JointDistribution,
        likelihood::Function,
        estimate::LaplaceEstimate,
        models::Vector{<:UQModel} = UQModel[],
    )
    issetequal(names(prior), Symbol.(names(estimate.x0))) || throw(ArgumentError("Starting-point columns must match the prior variables."))
    log_posterior = _log_posterior(prior, likelihood, models, estimate.islog)
    return _laplaceestimate(log_posterior, estimate)
end
