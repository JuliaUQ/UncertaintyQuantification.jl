
"""
    UQTargetDensity

Target density for transport maps.
"""
struct UQTargetDensity <: TransportMaps.AbstractMapDensity
    logprior::Function
    loglikelihood::Function
    models::Vector{<:UQModel}
    names::Vector{Symbol}

    function UQTargetDensity(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        loglikelihood::Function,
        models::Vector{<:UQModel},
    )
        logprior =
            df -> vec(
                sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), prior)...); dims=2)
            )


        names = [rv.name for rv in prior]


        return new(logprior, loglikelihood, models, names)
    end
end

"""
    logpdf(target::UQTargetDensity, X::Matrix{<:Real})


"""
function logpdf(target::UQTargetDensity, X::Matrix{<:Real})

    df = DataFrame(X, target.names)

    if !isempty(target.models)
        evaluate!(target.models, df)
    end

    return target.logprior(df) + target.loglikelihood(df)
end

function logpdf(target::UQTargetDensity, x::Vector{<:Real})

    df = DataFrame(permutedims(x), target.names)

    if !isempty(target.models)
        evaluate!(target.models, df)
    end

    return target.logprior(df) + target.loglikelihood(df)
end

# todo: make use with autodiff
function logdensity(target::UQTargetDensity)
    return x -> TransportMaps.central_difference_gradient(x -> logpdf(target, x)[1], x)
    #ForwardDiff.gradient(target, logpdf(target, x))
end

#! overloaded `grad_logpdf` estimation
function grad_logpdf(target::UQTargetDensity, X::Matrix{<:Real})
    n, d = size(X)
    log_gradients = zeros(Float64, n, d)

    Threads.@threads for i in 1:n
        log_gradients[i, :] .= logdensity(target)(X[i, :])
    end

    return log_gradients
end


function bayesianupdating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, #! this should accept multiple different distributions
    loglikelihood::Function,
    models::Vector{<:UQModel},
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights,
)
    target = UQTargetDensity(prior, loglikelihood, models)

    optimize!(transportmap, target, quadrature)

    return transportmap
end
