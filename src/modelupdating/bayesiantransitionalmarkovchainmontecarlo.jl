"""
    TransitionalMarkovChainMonteCarlo(prior, n, burnin, β, islog)

    Passed to [`bayesianupdating`](@ref) to run thetransitional Markov chain Monte Carlo algorithm  with [`RandomVariable`](@ref) vector `prior`. At each transitional level, one sample will be generated from `n` independent Markov chains after `burnin` steps have been discarded. The flag `islog` specifies whether the prior and likelihood functions passed to the  [`bayesianupdating`](@ref) method are already  given as logarithms.

Alternative constructors

```julia
    TransitionalMarkovChainMonteCarlo(prior, n, burnin, β)  # `islog` = true
TransitionalMarkovChainMonteCarlo(prior, n, burnin)    # `β` = 0.2,  `islog` = true
```

# References

[chingTransitionalMarkovChain2007](@cite)

"""
struct TransitionalMarkovChainMonteCarlo <: AbstractBayesianMethod # Transitional Markov Chain Monte Carlo
    n::Int
    burnin::Int
    β::Real
    islog::Bool

    function TransitionalMarkovChainMonteCarlo(
            n::Int,
            burnin::Int,
            β::Real = 0.2,
            islog::Bool = true,
        )
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        return new(n, burnin, β, islog)
    end
end

# TMCMC implementation
function bayesianupdating(
        prior::JointDistribution,
        likelihood::Function,
        tmcmc::TransitionalMarkovChainMonteCarlo,
        models::Vector{<:UQModel} = UQModel[]
    )
    log_prior = _log_prior(prior)

    covariance_method = LinearShrinkage(DiagonalUnitVariance(), :lw)

    rv_names = names(prior)
    n_rv = length(rv_names)

    j = 0 # iteration
    βⱼ = 0.0 # tempering

    θⱼ = sample(prior, tmcmc.n) # prior samples

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

        idx = StatsBase.sample(collect(1:(tmcmc.n)), weights, tmcmc.n; replace = true)

        θⱼ⁺ = θⱼ[idx, :]

        Σⱼ = tmcmc.β^2 * cov(covariance_method, Matrix(θⱼ⁺[:, rv_names]))

        # creating zero-means MvNormal, since cov stays constant along MCMC chains
        mv_normal = MvNormal(zeros(n_rv), Σⱼ)

        # Run inner MH algorithm

        chain = Vector{DataFrame}(undef, tmcmc.burnin + 2)

        chain[1] = copy(θⱼ⁺)

        target = if tmcmc.islog
            df -> likelihood(df) .* βⱼ⁺ .+ log_prior(df)
        else
            df -> log.(likelihood(df)) .* βⱼ⁺ .+ log_prior(df)
        end

        for i in 2:(tmcmc.burnin + 2)
            next = copy(chain[i - 1])

            next[:, rv_names] = Matrix(next[:, rv_names]) .+ transpose(rand(mv_normal, tmcmc.n))

            # safeguard for Inf in the prior
            idx_inf = findall(isinf, log_prior(next))

            while !isempty(idx_inf)
                current_means = Matrix(chain[i - 1][idx_inf, rv_names])

                next[idx_inf, rv_names] = current_means .+ transpose(rand(mv_normal, length(idx_inf)))

                # only re-checking idx_inf candidates
                still_inf = isinf.(log_prior(next[idx_inf, rv_names]))
                idx_inf = idx_inf[still_inf]
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

    model_calls = tmcmc.n * (1 + j * (1 + tmcmc.burnin))
    @debug "Model Calls" model_calls

    return θⱼ, S
end

function bayesianupdating(
        likelihood::Function,
        models::Vector{<:UQModel},
        tmcmc::TransitionalMarkovChainMonteCarlo,
    )
    prior = if tmcmc.islog
        df -> vec(
            sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), tmcmc.prior)...); dims = 2),
        )
    else
        df -> vec(prod(hcat(map(rv -> pdf.(rv.dist, df[:, rv.name]), tmcmc.prior)...); dims = 2))
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
