"""
    SingleComponentMetropolisHastings(proposal, x0, n, burnin, islog)

Passed to [`bayesianupdating`](@ref) to run the single-component Metropolis-Hastings algorithm starting from `x0` with  univariate proposal distibution `proposal`. Will generate `n` samples *after* performing `burnin` steps of the Markov chain and discarding the samples. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms.

Alternative constructor

```julia
    SingleComponentMetropolisHastings(proposal, x0, n, burnin)  # `islog` = true
```

"""
struct SingleComponentMetropolisHastings <: AbstractBayesianMethod
    proposal::UnivariateDistribution
    x0::NamedTuple
    n::Int
    burnin::Int
    islog::Bool

    function SingleComponentMetropolisHastings(
        proposal::UnivariateDistribution,
        x0::NamedTuple,
        n::Int,
        burnin::Int,
        islog::Bool=true,
    )
        if n <= 0
            error("Number of samples `n` must be positive")
        end
        return new(proposal, x0, n, burnin, islog)
    end
end

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
    prior::Function,
    likelihood::Function,
    models::Vector{<:UQModel},
    mh::SingleComponentMetropolisHastings,
)
    number_of_dimensions = length(mh.x0)

    samples = DataFrame(collect(mh.x0)', collect(keys(mh.x0)))

    if !isempty(models)
        evaluate!(models, samples)
    end

    posterior = if mh.islog
        df -> likelihood(df) .+ prior(df)
    else
        df -> log.(likelihood(df)) .+ log.(prior(df))
    end

    rejection = 0.0

    for i in 1:(mh.n + mh.burnin - 1)
        current = DataFrame(samples[i, :])
        x = DataFrame(samples[i, :])

        for d in 1:number_of_dimensions
            x[1, d] += rand(mh.proposal)

            # safeguard for areas where the logprior is -Inf (prior = 0)
            while mh.islog ? isinf(prior(x)[1]) : isinf(log.(prior(x))[1])
                x[1, d] = current[1, d] + rand(mh.proposal)
            end

            if !isempty(models)
                evaluate!(models, x)
            end

            α = min(0, posterior(x)[1] - posterior(current)[1])

            if α >= log(rand())
                current[1, d] = x[1, d]
            else
                x[1, d] = current[1, d]
                rejection += 1
            end
        end

        push!(samples, x[1, :])
    end

    rejection /= ((mh.n + mh.burnin) * number_of_dimensions)

    # discard burnin samples during return
    return samples[(mh.burnin + 1):end, :], rejection
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

"""
    TransitionalMarkovChainMonteCarlo(prior, n, burnin, β, islog)

    Passed to [`bayesianupdating`](@ref) to run thetransitional Markov chain Monte Carlo algorithm  with [`RandomVariable'](@ref) vector `prior`. At each transitional level, one sample will be generated from `n` independent Markov chains after `burnin` steps have been discarded. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms.

Alternative constructors

```julia
    TransitionalMarkovChainMonteCarlo(prior, n, burnin, β)  # `islog` = true
     TransitionalMarkovChainMonteCarlo(prior, n, burnin)    # `β` = 0.2,  `islog` = true
```

# References

[chingTransitionalMarkovChain2007](@cite)

"""
struct TransitionalMarkovChainMonteCarlo <: AbstractBayesianMethod # Transitional Markov Chain Monte Carlo
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}
    n::Int
    burnin::Int
    β::Real
    islog::Bool

    function TransitionalMarkovChainMonteCarlo(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        n::Int,
        burnin::Int,
        β::Real=0.2,
        islog::Bool=true,
    )
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        return new(prior, n, burnin, β, islog)
    end
end

# TMCMC implementation
function bayesianupdating(
    prior::Function,
    likelihood::Function,
    models::Vector{<:UQModel},
    tmcmc::TransitionalMarkovChainMonteCarlo,
)
    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw)

    rv_names = names(tmcmc.prior)

    j = 0 # iteration
    βⱼ = 0.0 # tempering

    θⱼ = sample(tmcmc.prior, tmcmc.n) # prior samples

    if !isempty(models)
        evaluate!(models, θⱼ)
    end

    S = 0.0

    while βⱼ < 1
        j += 1

        likelihood_j = tmcmc.islog ? likelihood(θⱼ) : log.(likelihood(θⱼ))

        adjust = maximum(likelihood_j)

        βⱼ⁺, wⱼ = _beta_and_weights(βⱼ, likelihood_j .- adjust)

        @debug "βⱼ" βⱼ⁺

        S += (log(mean(wⱼ)) + (βⱼ⁺ - βⱼ) * adjust)

        weights = FrequencyWeights(wⱼ ./ sum(wⱼ))

        idx = StatsBase.sample(collect(1:(tmcmc.n)), weights, tmcmc.n; replace=true)

        θⱼ⁺ = θⱼ[idx, :]

        Σⱼ = tmcmc.β^2 * cov(covariance_method, Matrix(θⱼ⁺[:, rv_names]))

        # Run inner MH algorithm

        chain = Vector{DataFrame}(undef, tmcmc.burnin + 2)

        chain[1] = copy(θⱼ⁺)

        target = if tmcmc.islog
            df -> likelihood(df) .* βⱼ⁺ .+ prior(df)
        else
            df -> log.(likelihood(df)) .* βⱼ⁺ .+ log.(prior(df))
        end

        for i in 2:(tmcmc.burnin + 2)
            next = copy(chain[i - 1])

            for (j, x) in enumerate(eachrow(next[:, rv_names]))
                next[j, rv_names] = rand(MvNormal(collect(x), Σⱼ))
            end

            # safeguard for Inf in the prior
            # !TODO: Find a cleaner way to do this
            idx_inf = findall(isinf, prior(next))

            while !isempty(idx_inf)
                for (j, x) in zip(idx_inf, (eachrow(chain[i - 1][idx_inf, rv_names])))
                    next[j, rv_names] = rand(MvNormal(collect(x), Σⱼ))
                end

                idx_inf = findall(isinf, prior(next))
            end

            if !isempty(models)
                evaluate!(models, next)
            end

            α = min.(0, target(next) .- target(chain[i - 1]))

            accept = α .>= log.(rand(length(α)))

            reject = .!accept

            next[reject, :] .= chain[i - 1][reject, :]

            chain[i] = next
        end

        θⱼ⁺ = chain[end]

        βⱼ = βⱼ⁺
        θⱼ = θⱼ⁺
    end
    return θⱼ, S
end

function bayesianupdating(
    likelihood::Function,
    models::Vector{<:UQModel},
    tmcmc::TransitionalMarkovChainMonteCarlo,
)
    prior = if tmcmc.islog
        df -> vec(
            sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), tmcmc.prior)...); dims=2),
        )
    else
        df -> vec(prod(hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), tmcmc.prior)...); dims=2))
    end

    return bayesianupdating(prior, likelihood, models, tmcmc)
end

function bayesianupdating(likelihood::Function, tmcmc::TransitionalMarkovChainMonteCarlo)
    return bayesianupdating(likelihood, UQModel[], tmcmc)
end

function bayesianupdating(
    prior::Function, likelihood::Function, tmcmc::TransitionalMarkovChainMonteCarlo
)
    return bayesianupdating(prior, likelihood, UQModel[], tmcmc)
end

# Compute the next value for `β` and the nominal weights `w` using bisection.
function _beta_and_weights(β::Real, L::AbstractVector{<:Real})
    low = β
    high = 2

    local x, w # Declare variables so they are visible outside the loop

    while (high - low) / middle(low, high) > 1e-6 && high > eps()
        x = middle(low, high)
        w = exp.((x - β) .* L)

        if std(w) / mean(w) > 1
            high = x
        else
            low = x
        end
    end

    if x > 1
        x = 1
        w = exp.((x - β) .* L)
    end

    return x, w
end
