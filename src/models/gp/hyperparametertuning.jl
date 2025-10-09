using DifferentiationInterface

abstract type AbstractHyperparameterOptimization end

"""
    NoHyperparameterOptimization()

Creates a configuration that disables hyperparameter optimization for a Gaussian process model.

# Examples

```jldoctest
julia> NoHyperparameterOptimization()
NoHyperparameterOptimization()
```
"""
struct NoHyperparameterOptimization <: AbstractHyperparameterOptimization end

"""
    MaximumLikelihoodEstimation(optimizer::Optim.FirstOrderOptimizer, options::Optim.Options)

Represents a hyperparameter optimization strategy that maximizes the log marginal likelihood
of a Gaussian process model.

# Constructors
* `MaximumLikelihoodEstimation()` (default: optimizer = Optim.LBFGS(), options = Optim.Options(; iterations=10, show_trace=false))
* `MaximumLikelihoodEstimation(optimizer::Optim.FirstOrderOptimizer, options::Optim.Options)`

# Note
You can choose from any gradient-based optimizer and set of options provided by [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/), 
such as `LBFGS()`, `Adam()`, or `ConjugateGradient()`.

# Examples

```jldoctest
julia> MaximumLikelihoodEstimation(Optim.Adam(alpha=0.01), Optim.Options(; iterations=1000, show_trace=false))
MaximumLikelihoodEstimation(Adam{Float64, Float64, Flat}(0.01, 0.9, 0.999, 1.0e-8, Flat()), Optim.Options(x_abstol = 0.0, x_reltol = 0.0, f_abstol = 0.0, f_reltol = 0.0, g_abstol = 1.0e-8, outer_x_abstol = 0.0, outer_x_reltol = 0.0, outer_f_abstol = 0.0, outer_f_reltol = 0.0, outer_g_abstol = 1.0e-8, f_calls_limit = 0, g_calls_limit = 0, h_calls_limit = 0, allow_f_increases = true, allow_outer_f_increases = true, successive_f_tol = 1, iterations = 1000, outer_iterations = 1000, store_trace = false, trace_simplex = false, show_trace = false, extended_trace = false, show_warnings = true, show_every = 1, time_limit = NaN, )
)
```
"""
struct MaximumLikelihoodEstimation <: AbstractHyperparameterOptimization
    optimizer::Optim.FirstOrderOptimizer
    options::Optim.Options
end

MaximumLikelihoodEstimation() = MaximumLikelihoodEstimation(
    Optim.LBFGS(),
    Optim.Options(; iterations=10, show_trace=false)
)

function optimize_hyperparameters(
    gp::Union{AbstractGPs.GP, NoisyGP}, 
    ::Union{RowVecs{<:Real}, Vector{<:Real}}, 
    ::Vector{<:Real}, 
    ::NoHyperparameterOptimization
)
    return gp
end

objective(
    f::Union{AbstractGPs.GP, NoisyGP}, 
    x::Union{RowVecs{<:Real}, Vector{<:Real}}, 
    y::Vector{<:Real}, 
    ::MaximumLikelihoodEstimation
) = -logpdf(f(x), y)

function optimize_hyperparameters(
    gp::Union{AbstractGPs.GP, NoisyGP}, 
    x::Union{RowVecs{<:Real}, Vector{<:Real}}, 
    y::Vector{<:Real}, 
    mle::MaximumLikelihoodEstimation
)
    model, θ₀ = parameterize(gp)
    θ₀_flat, unflatten = ParameterHandling.flatten(θ₀)

    result = optimize(
        θ -> objective(model(unflatten(θ)), x, y, mle), 
        θ₀_flat, 
        mle.optimizer, mle.options; 
        autodiff= AutoZygote()
        )
    return model(unflatten(result.minimizer))
end