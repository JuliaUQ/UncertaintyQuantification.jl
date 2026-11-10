# general functions for bayesian updating
function _log_posterior(prior::JointDistribution, likelihood::Function, models::Vector{<:UQModel}, islog::Bool)

    log_prior = df -> vec(
        sum(logpdf(prior, df[!, names(prior)]), dims = 2)
    )

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
    log_prior = df -> vec(
        sum(logpdf(prior, df[!, names(prior)]), dims = 2)
    )

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

# !TODO: Not sure if this is the right way to handle this
function bayesianupdating(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        likelihood::Function,
        method::AbstractBayesianMethod;
        models::Union{Nothing, UQModel, Vector{<:UQModel}} = nothing
    )
    joint_prior = JointDistribution(prior)
    return bayesianupdating(joint_prior, likelihood, method, models)
end
