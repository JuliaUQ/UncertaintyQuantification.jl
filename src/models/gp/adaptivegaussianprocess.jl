"""
    AdaptiveGaussianProcess

An adaptive Gaussian-process model that stores a fitted [`GaussianProcess`](@ref)
and the settings needed to refine it repeatedly with [`refine!`](@ref).

The fields `gp`, `input`, `model`, `acquisition_function`, and
`candidate_sampling` store the current surrogate, input distributions and
model(s) to evaluate, candidate-selection rule, and candidate-sampling method.
The remaining fields store the hyperparameter-learning settings used when
refitting. Use `evaluate!`, `sample!`, and `logpdf` to access the corresponding
operations of the contained Gaussian process.
"""
mutable struct AdaptiveGaussianProcess
    gp::GaussianProcess
    input::Union{UQInput, Vector{<:UQInput}}
    model::Union{UQModel, Vector{<:UQModel}}
    acquisition_function::AbstractGaussianProcessAcquisitionFunction
    candidate_sampling::AbstractMonteCarlo
    learn_noise::Bool
    learn_hyperparameters::Bool
    optimizer::UncertaintyQuantification.AbstractHyperparameterOptimization
end

"""
    AdaptiveGaussianProcess(
        input, model, design, output::Symbol,
        acquisition_function = MaximumVariance(),
        input_names = wrap(names(input)); kwargs...
    )

Fit an initial [`GaussianProcess`](@ref) and store the configuration needed for
subsequent adaptive refinement with [`refine!`](@ref). Construction does not
perform adaptive iterations; call `refine!` on the returned object to add
training points. The initial input transformation is retained when the GP is
refitted, and training data and outputs remain in physical space and on their
original scale.

# Arguments
- `input`: A `UQInput` or vector of inputs defining the sampling distributions
  and parameters.
- `model`: A `UQModel` or vector of models evaluated at selected points.
- `design`: An `AbstractMonteCarlo` specifying how to sample the initial
  training design and its size.
- `output`: Name of the model output to approximate.
- `acquisition_function`: An `AbstractGaussianProcessAcquisitionFunction`
  used to select candidates during refinement. Defaults to `MaximumVariance()`.
- `input_names`: GP input columns. Defaults to all names from `input`.

# Keyword Arguments
- `candidate_sampling`: Monte Carlo method for sampling candidates during
  refinement. Defaults to `MonteCarlo(100_000)`.
- `mean`: Prior mean function. Defaults to `ZeroMean()`.
- `kernel`: Prior covariance kernel. Defaults to `SqExponentialKernel()`.
- `normalize`: Whether to map inputs to standard normal space using their
  distributions. Defaults to `true`; `false` uses physical-space inputs.
- `σ²`: Nonnegative observation-noise variance. Defaults to `1.0e-5`.
- `learn_noise`: Whether to optimize the noise variance along with the other
  hyperparameters. Defaults to `false`; only takes effect when
  `learn_hyperparameters = true`.
- `learn_hyperparameters`: Whether to optimize hyperparameters on each fit.
  Defaults to `true`.
- `optimizer`: Hyperparameter-optimization strategy. Defaults to
  `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))`.

# Examples
```jldoctest
julia> using QuasiMonteCarlo

julia> using DataFrames: nrow

julia> x = RandomVariable(Uniform(-2, 2), :x);

julia> model = Model(df -> sin.(df.x), :y);

julia> agp = AdaptiveGaussianProcess(
           x, model, QuasiMonteCarloSampling(6, LatinHypercubeSample()), :y;
           candidate_sampling = MonteCarlo(100), learn_hyperparameters = false,
       );

julia> UncertaintyQuantification.refine!(agp, 2);

julia> agp isa AdaptiveGaussianProcess && nrow(agp.gp.data) == 8
true
```
"""
function AdaptiveGaussianProcess(
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        design::AbstractMonteCarlo,
        output::Symbol,
        acquisition_function::AbstractGaussianProcessAcquisitionFunction=MaximumVariance(),
        input_names=wrap(names(input));
        candidate_sampling=MonteCarlo(100_000),
        σ²=1e-5,
        learn_noise=false,
        learn_hyperparameters=true,
        mean::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        normalize::Bool = true,
        optimizer::UncertaintyQuantification.AbstractHyperparameterOptimization=MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )

    gp = GaussianProcess(
        input,
        model,
        design,
        output,
        input_names;
        mean=mean,
        kernel=kernel,
        normalize=normalize,
        σ²=σ²,
        learn_noise=learn_noise,
        learn_hyperparameters=learn_hyperparameters,
        optimizer=optimizer
    )
    return AdaptiveGaussianProcess(gp, input, model, acquisition_function, candidate_sampling, learn_noise, learn_hyperparameters, optimizer)
end

"""
    refine!(agp::AdaptiveGaussianProcess, n::Int = 1)

Adaptively add up to `n` observations to `agp` and refit its Gaussian process.
Each iteration samples candidates using `agp.candidate_sampling`, selects a
candidate with `agp.acquisition_function`, evaluates `agp.model` there, and
updates `agp.gp`. Training rows that duplicate existing rows are not appended.
The fitted model is changed in place; the function returns `nothing`. Passing
`n = 0` leaves the model unchanged.

# Arguments
- `agp`: Adaptive Gaussian process to refine.
- `n`: Number of adaptive iterations. Defaults to `1`.
"""
function refine!(agp::AdaptiveGaussianProcess, n::Int=1)
    for i in 1:n
        candidates = UncertaintyQuantification.sample(agp.input, agp.candidate_sampling)
        next_point = _find_next_point(agp.gp, candidates, agp.acquisition_function)

        evaluate!(agp.model, next_point)
        agp.gp = _refit_gp(
            agp.gp, next_point, agp.optimizer, agp.gp.σ², agp.learn_noise, agp.learn_hyperparameters
        )

        @debug "added point" iteration = i point = NamedTuple(next_point[1, :])
    end
end

"""
    evaluate!(agp::AdaptiveGaussianProcess, data::DataFrame;
              mean::Bool = true, variance::Bool = false)

Evaluate the Gaussian process contained in `agp` at the input locations in
`data`, writing requested predictions to `data` and returning `nothing`.
Prediction columns are named `<output>_mean` and `<output>_var`. This is a
convenience wrapper for `evaluate!(agp.gp, data; mean, variance)`.

# Arguments
- `agp`: Adaptive Gaussian process whose current fitted GP is evaluated.
- `data`: Prediction locations containing the input columns used by the GP.

# Keyword Arguments
- `mean`: Whether to write the predictive mean. Defaults to `true`.
- `variance`: Whether to write the predictive variance. Defaults to `false`.
"""
function evaluate!(
        gp::AdaptiveGaussianProcess,
        data::DataFrame;
        mean::Bool = true,
        variance::Bool = false
    )

    return evaluate!(gp.gp, data; mean=mean, variance=variance)
end

"""
    sample!(agp::AdaptiveGaussianProcess, data::DataFrame, n::Int = 1)

Draw `n` independent realizations from the joint posterior predictive
distribution of the Gaussian process in `agp` at the locations in `data`.
This delegates to `sample!(agp.gp, data, n)` and returns its result.

# Arguments
- `agp`: Adaptive Gaussian process whose current fitted GP is sampled.
- `data`: Prediction locations containing the input columns used by the GP.
- `n`: Number of realizations to draw. Defaults to `1`.
"""
function sample!(
        gp::AdaptiveGaussianProcess,
        data::DataFrame,
        n::Int = 1
    )
    return sample!(gp.gp, data, n)
end

"""
    logpdf(agp::AdaptiveGaussianProcess, data::DataFrame)

Evaluate the joint posterior predictive log density of the output values in
`data` at the input locations in the same table. The calculation uses the
current Gaussian process stored in `agp.gp` and returns its `logpdf` result.

# Arguments
- `agp`: Adaptive Gaussian process whose current fitted GP is used.
- `data`: DataFrame containing the GP input columns and observed output column.
"""
function logpdf(gp::AdaptiveGaussianProcess, data::DataFrame)
    return logpdf(gp.gp, data)
end

function _refit_gp(
        gp::GaussianProcess,
        new_data::DataFrame,
        optimizer::AbstractHyperparameterOptimization,
        σ²::Float64,
        learn_noise::Bool,
        learn_hyperparameters::Bool,
    )
    # Only add points that are not already in the training data.
    unique_new_data = antijoin(new_data, gp.data; on = names(new_data))
    append!(gp.data, unique_new_data)

    x = transform(gp.data[:, gp.inputs], gp.transform)
    y = vec(gp.data[:, gp.output])
    posterior_gp, σ² = _fit_gp(
        gp.posterior.prior, x, y;
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer,
    )

    return GaussianProcess(posterior_gp, gp.output, gp.inputs, σ², gp.transform, gp.data)
end
