"""
    TransitionalMarkovChainMonteCarlo(n, burnin, γ = 0.2, islog = true)

Configure transitional MCMC for [`bayesianupdating`](@ref). Pass the prior
separately as `bayesianupdating(prior, likelihood, method, models=UQModel[])`.
At each level, retain one state from each of `n` chains after `burnin + 1`
transitions. `γ` scales the proposal standard deviation; `islog` specifies
whether the likelihood returns log densities.

# References

[chingTransitionalMarkovChain2007](@cite)
"""
struct TransitionalMarkovChainMonteCarlo <: AbstractBayesianMethod # Transitional Markov Chain Monte Carlo
    n::Int
    burnin::Int
    γ::Real
    islog::Bool

    function TransitionalMarkovChainMonteCarlo(
            n::Int,
            burnin::Int,
            γ::Real = 0.2,
            islog::Bool = true,
        )
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        burnin >= 0 || throw(ArgumentError("`burnin` must be nonnegative."))
        return new(n, burnin, γ, islog)
    end
end

# TMCMC implementation
function bayesianupdating(
        prior::JointDistribution,
        likelihood::Function,
        tmcmc::TransitionalMarkovChainMonteCarlo,
        models::Vector{<:UQModel} = UQModel[]
    )
    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw)

    rv_names = names(prior)
    n_rv = length(rv_names)

    j = 0 # iteration
    βⱼ = 0.0 # tempering

    θⱼ = sample(prior, tmcmc.n) # prior samples

    if !isempty(models)
        evaluate!(models, θⱼ)
    end

    model_calls = isempty(models) ? 0 : tmcmc.n
    # The helper evaluates models before calling the likelihood on supported rows.
    counted_likelihood = df -> begin
        if !isempty(models)
            model_calls += nrow(df)
        end
        return likelihood(df)
    end
    annealed_log_posterior = _annealed_log_posterior(prior, counted_likelihood, models, tmcmc.islog)
    S = 0.0

    while βⱼ < 1
        j += 1

        likelihood_j = tmcmc.islog ? likelihood(θⱼ) : log.(likelihood(θⱼ))

        adjust = maximum(likelihood_j)

        βⱼ⁺, wⱼ = _beta_and_weights(βⱼ, likelihood_j .- adjust)

        @debug "βⱼ" βⱼ⁺

        S += (log(mean(wⱼ)) + (βⱼ⁺ - βⱼ) * adjust)

        weights = FrequencyWeights(wⱼ ./ sum(wⱼ))

        idx = StatsBase.sample(collect(1:(tmcmc.n)), weights, tmcmc.n; replace = true)

        θⱼ⁺ = θⱼ[idx, :]

        Σⱼ = tmcmc.γ^2 * cov(covariance_method, Matrix(θⱼ⁺[:, rv_names]))

        # creating zero-means MvNormal, since cov stays constant along MCMC chains
        mv_normal = MvNormal(zeros(n_rv), Σⱼ)

        # The covariance is fixed within a level, shared by all particle chains.
        target = df -> annealed_log_posterior(df, βⱼ⁺)

        # M-H with multi-variate proposal ∼ N(0, Σⱼ)
        mh = MetropolisHastings(
            mv_normal, θⱼ⁺[:, rv_names], 1, tmcmc.burnin + 1, true,
        )
        updated, _ = sample(target, mh)
        # sample returns parameters only; restore model outputs for the next level
        # and for the returned posterior population.
        if !isempty(models)
            evaluate!(models, updated)
            model_calls += tmcmc.n
        end
        θⱼ⁺ = updated

        βⱼ = βⱼ⁺
        θⱼ = θⱼ⁺
    end

    @debug "Model Calls" model_calls

    return θⱼ, S
end

# Compute the next value for `β` and the nominal weights `w` using bisection.
function _beta_and_weights(β::Real, L::AbstractVector{<:Real})
    low = β
    high = 2

    local x, w # Declare variables so they are visible outside the loop

    while (high - low) / middle(low, high) > 1.0e-6 && high > eps()
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
