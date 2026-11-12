"""
    bayesianupdating(
        prior::JointDistribution, likelihood,
        method::AbstractMarkovChainMonteCarlo, models = UQModel[]
    )

Sample a posterior using an MCMC method. The likelihood accepts a DataFrame
and returns one value per row, using log densities when `method.islog` is true.
The prior is always a distribution. Any supplied models are evaluated before
likelihood evaluation. Returns `(samples, rejection_fraction)`.
"""
function bayesianupdating(
        prior::JointDistribution,
        likelihood::Function,
        method::AbstractMarkovChainMonteCarlo,
        models::Vector{<:UQModel} = UQModel[],
    )
    issetequal(names(prior), Symbol.(names(method.x0))) || throw(ArgumentError("Starting-point columns must match the prior variables."))
    log_posterior = _log_posterior(prior, likelihood, models, method.islog)
    # The posterior helper always returns log densities, even for density input.
    return sample(log_posterior, method; islog = true)
end
