


"""
    UQTargetDensity

Target density for transport maps.
"""
# todo: make parameterized, possibly rename?
struct UQTargetDensity <: TransportMaps.AbstractMapDensity
    logdensity::Function
    grad_logdensity::Function
    model::Union{UQModel,Nothing}
    names::Vector{Symbol}

    function UQTargetDensity(
        prior::Vector{<:RandomVariable{<:UnivariateDistribution}},
        loglikelihood::Function,
        model::UQModel, #! Mooncake breakes with Vector{<:UQModel}
    )
        function logprior(df::DataFrame)
            return vec(
                sum(hcat(map(rv -> logpdf.(rv.dist, df[:, rv.name]), prior)...); dims=2)
            )
        end

        names = [rv.name for rv in prior]

        function logposterior(x::AbstractVector{<:Real})
            df = _to_dataframe(x, names)

            evaluate!(model, df)

            return first(logprior(df) + loglikelihood(df))
        end

        function logposterior(X::AbstractMatrix{<:Real})
            df = _to_dataframe(X, names)

            evaluate!(model, df)

            return logprior(df) + loglikelihood(df)
        end

        # Gradient of log density
        if model isa Model
            @warn "Initializing gradient estimation with Mooncake. This may take a while."
            backend = AutoMooncake()
            prep = prepare_gradient(logposterior, backend, zeros(length(names)))
            gradient =
                x -> DifferentiationInterface.gradient(logposterior, prep, backend, x)
        elseif model isa ExternalModel
            @warn "Using Finite difference approximation for gradient."
            gradient = x -> TransportMaps.central_difference_gradient(logposterior, x)
        else
            error("ParallelModel not yet supported.")
        end

        return new(logposterior, gradient, model, names)
    end
end

function logpdf(density::UQTargetDensity, X::Matrix{<:Real})
    return density.logdensity(X)
end

function grad_logpdf(density::UQTargetDensity, X::Matrix{<:Real})
    n, d = size(X)
    log_gradients = zeros(Float64, n, d)

    for i in 1:n #! Mooncake breaks with Threads.@threads; somehow implement to use multi-threading with FD. Maybe use `pmap` or something?
        log_gradients[i, :] = density.grad_logdensity(X[i, :])
    end
    return log_gradients
end

function _to_dataframe(X::AbstractMatrix{<:Real}, names::Vector{Symbol})
    return DataFrame(X, names)
end

function _to_dataframe(x::AbstractVector{<:Real}, names::Vector{Symbol})
    return DataFrame(permutedims(x), names)
end

"""
    Struct for TransportMaps
"""
struct TransportMap # todo: possibly rename; add abstract type
    map::TransportMaps.PolynomialMap
    target::UQTargetDensity
    quadrature::AbstractQuadratureWeights
    names::Vector{Symbol}
end


function bayesianupdating(
    prior::Vector{<:RandomVariable{<:UnivariateDistribution}}, #! this should accept multiple different distributions
    loglikelihood::Function,
    model::UQModel,
    transportmap::TransportMaps.PolynomialMap,
    quadrature::TransportMaps.AbstractQuadratureWeights,
)
    target = UQTargetDensity(prior, loglikelihood, model)

    @warn "Starting Map Optimization..."

    optimize!(transportmap, target, quadrature)

    return TransportMap(transportmap, target, quadrature, target.names)
end

function sample(tm::TransportMap, n::Integer=1)
    Z = randn(n, numberdimensions(tm.map))
    X = TransportMaps.evaluate(tm.map, Z)
    return DataFrame(X, tm.names)
end

# todo: overload `evaluate!` and `inverse!` for TransportMap
function evaluate!(tm::TransportMap, Z::DataFrame)

end


# todo: add transformation between physical and standard normal space
function to_physical_space!(tm::TransportMap, Z::DataFrame)

end

function to_standard_normal_space!(tm::TransportMap, X::DataFrame)

end



# todo: overload display methods
