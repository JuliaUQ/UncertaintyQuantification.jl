# general functions for bayesian updating
function _log_prior(prior::JointDistribution)
    return log_prior = df -> begin
        X = permutedims(Matrix(df[!, names(prior)]))
        [logpdf(prior, x) for x in eachcol(X)]
    end
end

function _log_prior(prior::Vector{<:RandomVariable{<:UnivariateDistribution}})
    return log_prior = df -> vec(
        sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), prior)...); dims = 2),
    )
end

function _log_posterior(prior::JointDistribution, likelihood::Function, models::Vector{<:UQModel}, islog::Bool)

    log_prior = _log_prior(prior)

    return df -> begin
        if !isempty(models)
            evaluate!(models, df)
        end
        if islog
            likelihood(df) .+ log_prior(df)
        else
            log.(likelihood(df)) .+ log_prior(df)
        end
    end
end

function _log_posterior(prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, likelihood::Function, models::Vector{<:UQModel}, islog::Bool)

    log_prior = _log_prior(prior)

    return df -> begin
        if !isempty(models)
            evaluate!(models, df)
        end
        if islog
            likelihood(df) .+ log_prior(df)
        else
            log.(likelihood(df)) .+ log_prior(df)
        end
    end
end

function _annealed_log_posterior(prior::Function, likelihood::Function, islog::Bool)

    log_prior = _log_prior(prior)

    return df -> begin
        if !isempty(models)
            evaluate!(models, df)
        end
        if islog
            β .* likelihood(df) .+ log_prior(df)
        else
            β .* log.(likelihood(df)) .+ log_prior(df)
        end
    end
end

function _annealed_log_posterior(prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, likelihood::Function, models::Vector{<:UQModel}, islog::Bool)

    log_prior = _log_prior(prior)

    return df, β -> begin
            if !isempty(models)
                evaluate!(models, df)
        end
            if islog
                β .* likelihood(df) .+ log_prior(df)
        else
                β .* log.(likelihood(df)) .+ log_prior(df)
        end
        end
end

# !TODO: Some general multiple dispatch functions for model updating, like casting UQModels into arrays etc.

# function bayesianupdating(
#         prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
#         likelihood::Function,
#         method::AbstractBayesianMethod,
#         models::Union{Nothing, UQModel, Vector{<:UQModel}} = nothing
#     )
#     joint_prior = JointDistribution(prior)
#     return bayesianupdating(joint_prior, likelihood, method, models)
# end
