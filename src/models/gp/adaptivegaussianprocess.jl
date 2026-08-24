"""
    AdaptiveGaussianProcess(
        gp::GP,
        input,
        acquisition_function,
        n_added_points,
        n_design_points=10,
        experimental_design=LatinHypercubeSampling(n_design_points);
        kwargs...
    )

Fit a Gaussian-process surrogate from an initial experimental design, then
adaptively enrich its training data with points selected by
`acquisition_function`.

At each adaptive iteration, candidate points are sampled from `input`, the
acquisition function selects one candidate, `model` is evaluated at that
point, and the Gaussian process is refitted using the enlarged training set.

# Arguments
- `gp`: Prior Gaussian process specifying the mean function and kernel.
- `input`: Input variable or variables used for the initial design and
  candidate sampling.
- `model`: Expensive model evaluated at selected adaptive points.
- `output`: Name of the model output approximated by the surrogate.
- `acquisition_function`: Learning function used to rank candidate points.
- `n_added_points`: Number of adaptively selected training points.
- `n_design_points`: Number of points in the initial experimental design.
- `experimental_design`: Sampling/design method for the initial training data.

# Keyword Arguments
- `input_transform`: Transformation applied to GP input features.
- `output_transform`: Transformation applied to the GP response.
- `σ²`: Observation-noise variance.
- `learn_noise`: Whether to infer the observation-noise variance.
- `learn_hyperparameters`: Whether to optimize GP hyperparameters on each fit.
- `candidate_sampling`: Monte Carlo sampling method used for acquisition candidates.
- `optimizer`: Hyperparameter-optimization strategy.

# Examples
```jldoctest
julia> x = RandomVariable(Uniform(-2, 2), :x);

julia> model = Model(df -> sin.(df.x), :y);

julia> prior = GP(ZeroMean(), SqExponentialKernel());

julia> surrogate = AdaptiveGaussianProcess(
           prior,
           x,
           model,
           :y,
           MaximumVariance(),
           5,
       );
```
"""
function AdaptiveGaussianProcess(
        gp::GP,
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol,
        acquisition_function::AbstractGaussianProcessAcquisitionFunction,
        n_added_points::Int,
        n_design_points::Int = 10,
        experimental_design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments} = LatinHypercubeSampling(
            n_design_points
        );
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000),
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(
            Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)
        ),
    )
    gp_model = GaussianProcess(
        gp,
        input,
        model,
        output;
        experimental_design = experimental_design,
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer,
    )

    return AdaptiveGaussianProcess(
        gp_model,
        input,
        model,
        acquisition_function,
        n_added_points,
        candidate_sampling = candidate_sampling,
        optimizer = optimizer,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters
    )
end

"""
    AdaptiveGaussianProcess(
        input,
        model,
        output,
        acquisition_function,
        n_added_points,
        n_design_points=10,
        experimental_design=LatinHypercubeSampling(n_design_points);
        mean_fct=ZeroMean(),
        kernel=SqExponentialKernel(),
        kwargs...
    )

Construct and adapt a Gaussian-process surrogate using a zero-mean,
squared-exponential prior by default. Use `mean_fct` and `kernel` to choose a
different prior.
"""
function AdaptiveGaussianProcess(
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol,
        acquisition_function::AbstractGaussianProcessAcquisitionFunction,
        n_added_points::Int,
        n_design_points::Int = 10,
        experimental_design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments} = LatinHypercubeSampling(
            n_design_points
        );
        mean_fct::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000),
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(
            Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)
        ),
    )

    return AdaptiveGaussianProcess(
        GP(mean_fct, kernel),
        input,
        model,
        output,
        acquisition_function,
        n_added_points,
        n_design_points,
        experimental_design;
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        candidate_sampling = candidate_sampling,
        optimizer = optimizer,
    )
end

"""
    AdaptiveGaussianProcess(
        gp_model,
        input,
        model,
        acquisition_function,
        n_added_points;
        kwargs...
    )

Adapt an already-fitted `GaussianProcess` by evaluating `model` at
`n_added_points` selected from candidates sampled from `input`.
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

"""
    AdaptiveGaussianProcess(
        gp,
        data,
        input,
        model,
        output,
        acquisition_function,
        n_added_points;
        kwargs...
    )

Fit the initial surrogate from `data` with the specified prior `gp`, then
adaptively add points selected from candidates sampled from `input`. The
`model` supplies the expensive response at each selected point.
"""
function AdaptiveGaussianProcess(
        gp::GP,
        data::DataFrame,
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol,
        acquisition_function::AbstractGaussianProcessAcquisitionFunction,
        n_added_points::Int;
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000),
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(
            Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)
        ),
    )
    gp_model = GaussianProcess(
        gp,
        data,
        output;
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer,
    )

    return AdaptiveGaussianProcess(
        gp_model,
        input,
        model,
        acquisition_function,
        n_added_points;
        candidate_sampling = candidate_sampling,
        optimizer = optimizer,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
    )
end

"""
    AdaptiveGaussianProcess(
        data,
        input,
        model,
        output,
        acquisition_function,
        n_added_points;
        mean_fct=ZeroMean(),
        kernel=SqExponentialKernel(),
        kwargs...
    )

Fit the initial surrogate from `data`, then adaptively add points selected
from candidates sampled from `input`. Use `mean_fct` and `kernel` to specify
the initial GP prior; `model` is required to evaluate newly selected points.
"""
function AdaptiveGaussianProcess(
        data::DataFrame,
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol,
        acquisition_function::AbstractGaussianProcessAcquisitionFunction,
        n_added_points::Int;
        mean_fct::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000),
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(
            Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)
        ),
    )

    return AdaptiveGaussianProcess(
        GP(mean_fct, kernel),
        data,
        input,
        model,
        output,
        acquisition_function,
        n_added_points;
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        candidate_sampling = candidate_sampling,
        optimizer = optimizer,
    )
end

function _refit_gp(
        gp::GaussianProcess,
        new_data::DataFrame,
        optimizer::AbstractHyperparameterOptimization,
        σ²::Float64,
        learn_noise::Bool,
        learn_hyperparameters::Bool,
    )
    # fit Gaussian process to new data
    σ² = check_gp_input(σ², learn_noise)

    # Only add point if it is not in df
    unique_new_data = antijoin(new_data, gp.training_data; on = names(new_data))
    append!(gp.training_data, unique_new_data)

    # transform data
    x = transform(gp.training_data, gp.input_transformer)
    y = transform(gp.training_data, gp.output_transformer)

    if learn_hyperparameters
        _gp = optimize_hyperparameters(
            PriorGP(gp.posterior.prior, σ², learn_noise), x, y, optimizer
        )
        σ² = _gp.σ²

        posterior_gp = posterior(_gp(x), y)
    else
        posterior_gp = posterior(gp.posterior.prior(x, σ²), y)
    end

    return GaussianProcess(
        posterior_gp,
        gp.output,
        σ²,
        gp.input_transformer,
        gp.output_transformer,
        gp.training_data,
    )
end
