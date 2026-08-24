abstract type AbstractHyperparameterOptimization end

"""
    MaximumLikelihoodEstimation(
        optimizer=LBFGS(),
        options=Optim.Options(; iterations = 100, show_trace = false)),
        backend=AutoMooncake();
        restarts=5
    )

Represents a hyperparameter optimization strategy that maximizes the log marginal likelihood
of a Gaussian process model with random restarts of the optimization.

# Arguments
- `optimizer::Optim.AbstractOptimizer`: chosen optimizer (default: `LBFGS()`)
- `options:.Optim.Options`: options for the optimizer (default: `Optim.Options(; iterations=100, show_trace=false)`)
- `backend::AbstractADType`: automatic different backend (default: `Mooncake()`; can be `nothing` when using gradient-free optimization)

# Keyword Arguments
- `restarts`: Number of additional randomized optimization runs. Defaults to `5`.

# Note
You can choose from any optimizer and set of options provided by [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/),
such as `LBFGS()`, `Adam()`, or `ConjugateGradient()`.

# Examples

```jldoctest
julia> using Optim

julia> MaximumLikelihoodEstimation(Optim.Adam(alpha=0.01), Optim.Options(; iterations=1000, show_trace=false))
MaximumLikelihoodEstimation(Adam{Float64, Float64, Flat}(0.01, 0.9, 0.999, 1.0e-8, Flat()), Optim.Options(x_abstol = 0.0, x_reltol = 0.0, f_abstol = 0.0, f_reltol = 0.0, g_abstol = 1.0e-8, outer_x_abstol = 0.0, outer_x_reltol = 0.0, outer_f_abstol = 0.0, outer_f_reltol = 0.0, outer_g_abstol = 1.0e-8, f_calls_limit = 0, g_calls_limit = 0, h_calls_limit = 0, allow_f_increases = true, allow_outer_f_increases = true, successive_f_tol = 1, iterations = 1000, outer_iterations = 1000, store_trace = false, trace_simplex = false, show_trace = false, extended_trace = false, show_warnings = true, show_every = 1, time_limit = NaN, )
, AutoMooncake(), 5)
```
"""
struct MaximumLikelihoodEstimation <: AbstractHyperparameterOptimization
    optimizer::Optim.AbstractOptimizer
    options::Optim.Options
    backend::Union{AbstractADType, Nothing}
    restarts::Int

    function MaximumLikelihoodEstimation(
            optimizer::Optim.AbstractOptimizer = LBFGS(),
            options::Optim.Options = Optim.Options(; iterations = 100, show_trace = false),
            backend::Union{AbstractADType, Nothing} = AutoMooncake();
            restarts::Int = 5,
        )
        restarts >= 0 || throw(ArgumentError("restarts must be nonnegative"))

        if isa(optimizer, Optim.ZerothOrderOptimizer)
            return new(optimizer, options, nothing, restarts)
        end

        return new(optimizer, options, backend, restarts)
    end
end

objective(
    f::PriorGP,
    x::Union{RowVecs{<:Real}, Vector{<:Real}},
    y::Vector{<:Real},
    ::MaximumLikelihoodEstimation
) = -logpdf(f(x), y)

_initializer(θ) = θ .+ 0.5 .* randn(length(θ))

function optimize_hyperparameters(
        gp::PriorGP,
        x::Union{RowVecs{<:Real}, Vector{<:Real}},
        y::Vector{<:Real},
        mle::MaximumLikelihoodEstimation
    )
    model, θ₀ = parameterize(gp)
    θ₀_flat, unflatten = ParameterHandling.flatten(θ₀)
    obj = θ -> objective(model(unflatten(θ)), x, y, mle)

    best_gp = nothing
    best_objective = Inf

    for restart in 0:mle.restarts
        θ_init = restart == 0 ? copy(θ₀_flat) : _initializer(θ₀_flat)
        gp_opt = _optimize_hyperparameters(obj, model, unflatten, θ_init, mle)
        value = objective(gp_opt, x, y, mle)

        if value < best_objective
            best_objective = value
            best_gp = gp_opt
        end
    end

    return best_gp
end

function _optimize_hyperparameters(
        obj::Function,
        model,
        unflatten::Function,
        θ_init::AbstractVector,
        mle::MaximumLikelihoodEstimation,
    )
    if isa(mle.optimizer, Optim.FirstOrderOptimizer) ||
            isa(mle.optimizer, Optim.SecondOrderOptimizer)

        prep = DifferentiationInterface.prepare_gradient(obj, mle.backend, θ_init)

        function fg!(F, G, θ)
            value, gradient = DifferentiationInterface.value_and_gradient(
                obj, prep, mle.backend, θ
            )
            G !== nothing && (G .= gradient)
            return value
        end

        result = optimize(NLSolversBase.only_fg!(fg!), θ_init, mle.optimizer, mle.options)

        return model(unflatten(result.minimizer))

    elseif isa(mle.optimizer, Optim.ZerothOrderOptimizer)
        # Gradient-free optimizer
        result = optimize(obj, θ_init, mle.optimizer, mle.options)

        return model(unflatten(result.minimizer))
    else
        error("Optimizer of type $(typeof(mle.optimizer)) not supported.")
    end
end
