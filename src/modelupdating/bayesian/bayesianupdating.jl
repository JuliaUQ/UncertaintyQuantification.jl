# general functions for bayesian updating
function _log_prior(prior::JointDistribution)
    return df -> begin
        X = permutedims(Matrix(df[!, names(prior)]))
        map(eachcol(X)) do x
            supported = if prior.d isa Copulas.Copula
                all(i -> insupport(prior.m[i], x[i]), eachindex(prior.m))
            else
                insupport(prior.d, x)
            end
            supported ? logpdf(prior, x) : -Inf
        end
    end
end

function _log_prior(prior::Vector{<:RandomVariable{<:UnivariateDistribution}})
    return _log_prior(JointDistribution(prior))
end

function _log_posterior(
        prior::Union{JointDistribution, Vector{<:RandomVariable{<:UnivariateDistribution}}},
        likelihood::Function, models::Vector{<:UQModel}, islog::Bool,
    )
    annealed = _annealed_log_posterior(prior, likelihood, models, islog)
    return df -> annealed(df, 1.0)
end

function _annealed_log_posterior(
        prior::Union{JointDistribution, Vector{<:RandomVariable{<:UnivariateDistribution}}},
        likelihood::Function, models::Vector{<:UQModel}, islog::Bool,
    )
    return _annealed_log_posterior(_log_prior(prior), likelihood, models, islog)
end

function _annealed_log_posterior(
        log_prior::Function, likelihood::Function, models::Vector{<:UQModel}, islog::Bool,
    )
    return (df, β) -> begin
        lp = log_prior(df)
        # At β=0 the target is exactly the prior, even if the likelihood is zero.
        iszero(β) && return lp

        # Avoid model and likelihood evaluations outside the prior support.
        valid = findall(isfinite, lp)
        isempty(valid) && return lp
        supported = length(valid) == nrow(df) ? df : df[valid, :]
        if !isempty(models)
            evaluate!(models, supported)
        end
        ll = islog ? likelihood(supported) : log.(likelihood(supported))
        lp[valid] .+= β .* ll
        return lp
    end
end

"""
    bayesianupdating(prior, likelihood, method, models = UQModel[])

Perform Bayesian updating with MCMC, TMCMC, MAP, Laplace estimation, or a
transport map. The argument order is the same for every method. `prior` is a
`JointDistribution` or a vector of univariate `RandomVariable`s, which is
converted to an independent joint distribution. It is supplied separately from
`method`. A prior function is not accepted, since some methods must sample or
transform the prior.

The likelihood accepts a DataFrame and returns one value per row. `method.islog`
specifies whether these values are log densities; the prior always uses its
`logpdf`. `models` may be omitted, a single `UQModel`, or a vector of models.
All algorithm settings are stored on `method`.

The return value depends on the method: MCMC returns `(samples, rejection)`,
TMCMC returns `(samples, log_evidence)`, MAP returns a DataFrame of modes, and
Laplace estimation and transport maps return a `JointDistribution`.
"""
function bayesianupdating(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        likelihood::Function,
        method::Union{AbstractBayesianMethod, AbstractMarkovChainMonteCarlo, LaplaceEstimate},
        models::Vector{<:UQModel} = UQModel[],
    )
    return bayesianupdating(JointDistribution(prior), likelihood, method, models)
end

function bayesianupdating(
        prior::Union{JointDistribution, Vector{<:RandomVariable{<:UnivariateDistribution}}},
        likelihood::Function,
        method::Union{AbstractBayesianMethod, AbstractMarkovChainMonteCarlo, LaplaceEstimate},
        model::UQModel,
    )
    return bayesianupdating(prior, likelihood, method, [model])
end

# todo: Re-structure tests to be compatible with new file structure
# todo: Make same methods <: AbstractBayesianMethod ?
