"""
    AdaptiveGaussianProcess(
        input, model, design, output::Symbol, acquisition_function, n_added_points::Int,
        input_names::Vector{Symbol} = wrap(names(input)); kwargs...
    )

Fit a [`GaussianProcess`](@ref) from an initial experimental design, then
adaptively enrich its training data. At each iteration, sample candidates from
`input`, select one with `acquisition_function`, evaluate `model`, and refit the
GP. Return the final fitted `GaussianProcess`.

The initial input transformation is retained during refitting. Training data
remain in physical space and outputs remain on their original scale. Duplicate
training rows are not added, so fewer than `n_added_points` rows may be appended.

# Arguments
- `input`: A `UQInput` or vector of inputs for the initial design and candidates.
- `model`: A `UQModel` or vector of models evaluated at selected points.
- `design`: An `AbstractMonteCarlo` or `AbstractDesignOfExperiments` specifying
  the sampling method and number of initial points.
- `output`: Name of the model output to approximate.
- `acquisition_function`: An `AbstractGaussianProcessAcquisitionFunction`,
  such as `MaximumVariance()`, used to select a candidate.
- `n_added_points`: Number of adaptive iterations.
- `input_names`: GP input columns. Defaults to all names from `input`.

# Keyword Arguments
- `mean`: Prior mean function. Defaults to `ZeroMean()`.
- `kernel`: Prior covariance kernel. Defaults to `SqExponentialKernel()`.
- `normalize`: Whether to map inputs to standard normal space using their
  distributions. Defaults to `true`; `false` uses physical-space inputs.
- `σ²`: Nonnegative observation-noise variance. Defaults to `1.0e-10`.
- `learn_noise`: Whether to optimize the noise variance along with the other
  hyperparameters. Defaults to `false`; only takes effect when
  `learn_hyperparameters = true`.
- `learn_hyperparameters`: Whether to optimize hyperparameters on each fit.
  Defaults to `true`.
- `optimizer`: Hyperparameter-optimization strategy. Defaults to
  `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))`.
- `candidate_sampling`: Monte Carlo method for sampling candidates at each
  adaptive iteration. Defaults to `MonteCarlo(100_000)`.

# Examples
```jldoctest
julia> using QuasiMonteCarlo

julia> x = RandomVariable(Uniform(-2, 2), :x);

julia> model = Model(df -> sin.(df.x), :y);

julia> gp = AdaptiveGaussianProcess(
           x, model, QuasiMonteCarloSampling(6, LatinHypercubeSample()), :y, MaximumVariance(), 2;
           mean = ZeroMean(), kernel = SqExponentialKernel(),
           candidate_sampling = MonteCarlo(100), learn_hyperparameters = false,
       );

julia> gp isa GaussianProcess && nrow(gp.data) == 8
true
```
"""
function AdaptiveGaussianProcess(
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments},
        output::Symbol,
        acquisition_function::AbstractGaussianProcessAcquisitionFunction,
        n_added_points::Int,
        input_names::Vector{Symbol} = wrap(names(input));
        mean::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        normalize::Bool = true,
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000),
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(
            Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)
        ),
    )
    gp_model = GaussianProcess(
        input, model, design, output, input_names;
        mean = mean,
        kernel = kernel,
        normalize = normalize,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer,
    )

    return AdaptiveGaussianProcess(
        gp_model, input, model, acquisition_function, n_added_points;
        candidate_sampling = candidate_sampling,
        optimizer = optimizer,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
    )
end

"""
    AdaptiveGaussianProcess(
        gp_model::GaussianProcess, input, model, acquisition_function, n_added_points::Int;
        kwargs...
    )

Refine an already-fitted [`GaussianProcess`](@ref) by selecting candidates
sampled from `input`, evaluating `model`, and refitting after each selection.
Return the final fitted GP, retaining the initial input names and transformation.
The training DataFrame in `gp_model.data` is mutated as points are appended;
use `deepcopy(gp_model)` to preserve the original model and its data.

# Arguments
- `gp_model`: Fitted GP supplying the initial training data and prior.
- `input`: A `UQInput` or vector of inputs used to sample candidates.
- `model`: A `UQModel` or vector of models supplying outputs at selected points.
- `acquisition_function`: An `AbstractGaussianProcessAcquisitionFunction`
  used to select a candidate.
- `n_added_points`: Number of adaptive iterations. Duplicate training rows are
  not added. With `0`, return `gp_model` unchanged.

# Keyword Arguments
- `candidate_sampling`: Candidate sampling method. Defaults to `MonteCarlo(100_000)`.
- `optimizer`: Hyperparameter-optimization strategy. Defaults to
  `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))`.
- `σ²`: Nonnegative observation-noise variance used when refitting. Defaults
  to `gp_model.σ²`.
- `learn_noise`: Whether to optimize the noise variance. Defaults to `false`;
  only takes effect when `learn_hyperparameters = true`.
- `learn_hyperparameters`: Whether to optimize hyperparameters on each refit.
  Defaults to `true`.

# Examples
```jldoctest
julia> using QuasiMonteCarlo

julia> x = RandomVariable(Uniform(-2, 2), :x);

julia> model = Model(df -> sin.(df.x), :y);

julia> initial_gp = GaussianProcess(
           x, model, QuasiMonteCarloSampling(6, LatinHypercubeSample()), :y; learn_hyperparameters = false,
       );

julia> refined_gp = AdaptiveGaussianProcess(
           deepcopy(initial_gp), x, model, MaximumVariance(), 2;
           candidate_sampling = MonteCarlo(100), learn_hyperparameters = false,
       );

julia> (nrow(initial_gp.data), nrow(refined_gp.data))
(6, 8)
```
"""
function AdaptiveGaussianProcess(
        gp_model::GaussianProcess,
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        acquisition_function::AbstractGaussianProcessAcquisitionFunction,
        n_added_points::Int;
        candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000),
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(
            Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)
        ),
        σ²::Float64 = gp_model.σ²,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
    )
    for i in 1:n_added_points
        candidates = sample(input, candidate_sampling)
        next_point = _find_next_point(gp_model, candidates, acquisition_function)

        evaluate!(model, next_point)
        gp_model = _refit_gp(
            gp_model, next_point, optimizer, σ², learn_noise, learn_hyperparameters
        )

        @debug "added point" iteration = i point = NamedTuple(next_point[1, :])
    end

    return gp_model
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
