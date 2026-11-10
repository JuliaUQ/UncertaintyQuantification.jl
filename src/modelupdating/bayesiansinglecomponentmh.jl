"""
    bayesianupdating(prior, likelihood, models, mcmc)

Perform bayesian updating using the given `prior`, `likelihood`, `models`  and any MCMC sampler [`AbstractBayesianMethod`](@ref).

Alternatively the method can be called without `models`.

    bayesianupdating(prior, likelihood, mcmc)

When using [`TransitionalMarkovChainMonteCarlo`](@ref) the `prior` can automatically be constructed.

    bayesinupdating(likelihood, models, tmcmc)
    bayesianupdating(likelihood, tmcmc)


### Notes

`likelihood` is a Julia function which must be defined in terms of a `DataFrame` of samples, and must evaluate the likelihood for each row of the `DataFrame`

For example, a loglikelihood based on normal distribution using 'Data':

```julia
likelihood(df) = [sum(logpdf.(Normal.(df_i.x, 1), Data)) for df_i in eachrow(df)]
```

If a model evaluation is required to evaluate the likelihood, a vector of `UQModel`s must be passed to `bayesianupdating`. For example if the variable `x` above is the output of a numerical model.

"""
function bayesianupdating(
        prior::JointDistribution,
        likelihood::Function,
        mh::SingleComponentMetropolisHastings,
        models::Vector{<:UQModel} = UQModel[]
    )

    log_posterior = _log_posterior(prior, likelihood, models, mh.islog)
    samples, rejection = sample(log_posterior, mh)

    return samples, rejection
end

function bayesianupdating(
        prior::Function,
        likelihood::Function,
        models::UQModel,
        mh::SingleComponentMetropolisHastings,
    )
    return bayesianupdating(prior, likelihood, wrap(models), mh)
end

function bayesianupdating(
        prior::Function, likelihood::Function, mh::SingleComponentMetropolisHastings
    )
    return bayesianupdating(prior, likelihood, UQModel[], mh)
end
