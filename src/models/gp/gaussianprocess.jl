struct GaussianProcess <: UQModel
    posterior::AbstractGPs.PosteriorGP
    output::Symbol
    inputs::Vector{Symbol}
    σ²::Float64
    transform::Union{ZScoreTransform, Vector{<:UQInput}, Nothing}
    data::DataFrame
end

function Base.show(io::IO, gp::GaussianProcess)
    print(io, "GaussianProcess(")
    print(io, "mean=$(gp.posterior.prior.mean), ")
    print(io, "kernel=$(gp.posterior.prior.kernel), ")
    print(io, "input=$(gp.transform), ")
    print(io, "output=$(gp.output), ")
    print(io, "n_datapoints=$(size(gp.data, 1))")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", gp::GaussianProcess)
    println(io, "GaussianProcess")
    println(io, "   mean: $(gp.posterior.prior.mean)")
    println(io, "   kernel: $(gp.posterior.prior.kernel)")
    println(io, "   input: $(gp.transform)")
    println(io, "   output: $(gp.output)")
    print(io, "   n_datapoints: $(size(gp.data, 1))")
    return nothing
end

# function to check the inputs to a GaussianProcess constructor
function check_gp_input(σ²::Float64, learn_noise::Bool)
    # check if σ² is ≥0, not using @assert because apparently it can be turned off and shouldn't be used for function input checking (https://discourse.julialang.org/t/efficient-use-of-test-or-assert/75895/4)
    if σ² < 0.0
        throw(DomainError(σ², "σ² < 0"))
    end

    # σ² should be >0, otherwise the parameterization throws an error
    if learn_noise && σ² < eps()
        σ² = 1.0e-5
        @warn "learn_noise was set but σ² is too small, setting σ² = $(σ²)"
    end

    if !learn_noise && σ² < eps()
        @warn "using small σ² < eps() might lead to numerical instabilities"
    end
    return σ²
end

"""
    GaussianProcess(
        data::DataFrame, output::Symbol,
        inputs::Vector{Symbol} = propertynames(data[:, Not(output)]); kwargs...
    )

Fit a Gaussian-process surrogate to the input and output columns in `data`.
The returned model stores the fitted posterior, selected input names, noise
variance, input transformation, and training data in `posterior`, `inputs`,
`σ²`, `transform`, and `data`, respectively. The output is kept on its original
scale. The training DataFrame is stored by reference.

# Arguments
- `data`: Training data containing the input columns and observed output.
- `output`: Name of the output column to approximate.
- `inputs`: Input columns, in the order used by the GP. Defaults to all columns
  except `output`; specify this argument to exclude metadata or other outputs.

# Keyword Arguments
- `mean`: Prior mean function. Defaults to `ZeroMean()`.
- `kernel`: Prior covariance kernel. Defaults to `SqExponentialKernel()`.
- `normalize`: Whether to standardize each input column using its training-data
  mean and standard deviation. Defaults to `true`. The fitted transformation is
  reused for predictions; `false` uses the inputs without transformation.
- `σ²`: Nonnegative observation-noise variance. Defaults to `1.0e-10`.
- `learn_noise`: Whether to optimize the noise variance along with the other
  hyperparameters. Defaults to `false`; only takes effect when
  `learn_hyperparameters = true`.
- `learn_hyperparameters`: Whether to optimize the prior hyperparameters.
  Defaults to `true`.
- `optimizer`: Hyperparameter-optimization strategy. Defaults to
  `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))`.

# Examples
```jldoctest
julia> data = DataFrame(x = [0.0, 1.0, 2.0], y = [0.0, 1.0, 4.0], id = 1:3);

julia> gp = GaussianProcess(
           data, :y, [:x]; mean = ConstMean(0.0), kernel = SqExponentialKernel(),
           σ² = 1.0e-3, learn_hyperparameters = false,
       );

julia> gp.inputs == [:x] && gp.output == :y && nrow(gp.data) == 3
true
```
"""
function GaussianProcess(
        data::DataFrame,
        output::Symbol,
        inputs::Vector{Symbol} = propertynames(data[:, Not(output)]);
        mean::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        normalize::Bool = true,
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )

    gp = GP(mean, kernel)

    x = permutedims(Matrix(data[:, inputs]))
    y = vec(data[:, output])

    transform = nothing

    if normalize
        transform = fit(
            StatsBase.ZScoreTransform,
            x;
            dims = 2
        )
        StatsBase.transform!(transform, x)
    end

    gp, σ² = _fit_gp(
        gp, x, y;
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer
    )

    return GaussianProcess(gp, output, inputs, σ², transform, data)

end

function _fit_gp(
        gp::GP,
        x::AbstractMatrix{<:Real},
        y::AbstractVector{<:Real};
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )
    # force learn_noise to false if learn_hyperparameters is false
    if !learn_hyperparameters && learn_noise
        @warn "learn_hyperparameters is false, setting learn_noise to false"
        learn_noise = false
    end
    σ² = check_gp_input(σ², learn_noise)


    # optimize hyperparameters
    if learn_hyperparameters
        _gp = optimize_hyperparameters(PriorGP(gp, σ², learn_noise), x, y, optimizer)
        σ² = _gp.σ²
        # _gp is a PriorGP object, calling it directly involves the noise, so no need to add σ² again
        return posterior(_gp(x), y), σ²
    else
        # gp has to be called with noise since it is an AbstractGPs.GP object, not a PriorGP object
        return posterior(gp(x, σ²), y), σ²
    end
end

"""
    GaussianProcess(
        inputs, models, design, output::Symbol,
        input_names::Vector{Symbol} = wrap(names(inputs)); kwargs...
    )

Sample an initial experimental design, evaluate `models` at those points, and
fit a Gaussian-process surrogate for `output`. Training data are stored in
physical space in the returned model's `data` field. The output is kept on its
original scale.

# Arguments
- `inputs`: A `UQInput` or vector of inputs defining the sampling distributions
  and parameters.
- `models`: A `UQModel` or vector of models evaluated on the sampled data.
- `design`: An `AbstractMonteCarlo` or `AbstractDesignOfExperiments` specifying
  the sampling method and number of initial points, e.g. `LatinHypercubeSampling(10)`.
- `output`: Name of the model output to approximate.
- `input_names`: Input columns, in the order used by the GP. Defaults to all
  names from `inputs`.

# Keyword Arguments
- `mean`: Prior mean function. Defaults to `ZeroMean()`.
- `kernel`: Prior covariance kernel. Defaults to `SqExponentialKernel()`.
- `normalize`: Whether to map the inputs to standard normal space using their
  distributions. Defaults to `true`. The same transformation is applied during
  prediction; `false` uses physical-space inputs directly.
- `σ²`: Nonnegative observation-noise variance. Defaults to `1.0e-10`.
- `learn_noise`: Whether to optimize the noise variance along with the other
  hyperparameters. Defaults to `false`; only takes effect when
  `learn_hyperparameters = true`.
- `learn_hyperparameters`: Whether to optimize the prior hyperparameters.
  Defaults to `true`.
- `optimizer`: Hyperparameter-optimization strategy. Defaults to
  `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))`.

# Examples
```jldoctest
julia> x = RandomVariable(Uniform(0, 5), :x);

julia> model = Model(df -> sin.(df.x), :y);

julia> gp = GaussianProcess(
           x, model, LatinHypercubeSampling(10), :y;
           mean = ConstMean(0.0), kernel = SqExponentialKernel(),
           learn_hyperparameters = false,
       );

julia> gp.inputs == [:x] && nrow(gp.data) == 10
true
```
"""
function GaussianProcess(
        inputs::Union{<:UQInput, Vector{<:UQInput}},
        models::Union{<:UQModel, Vector{<:UQModel}},
        design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments},
        output::Symbol,
        input_names::Vector{Symbol} = wrap(names(inputs));
        mean::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        normalize::Bool = true,
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )

    inputs = wrap(inputs)

    gp = GP(mean, kernel)

    data = sample(inputs, design)

    evaluate!(models, data)

    transform = nothing
    if normalize
        to_standard_normal_space!(inputs, data)
        transform = inputs
    end

    x = permutedims(Matrix(data[:, input_names]))
    y = vec(data[:, output])

    if normalize
        to_physical_space!(inputs, data)
    end

    gp, σ² = _fit_gp(
        gp, x, y;
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer
    )

    return GaussianProcess(gp, output, input_names, σ², transform, data)
end

"""
    evaluate!(gp::GaussianProcess, data::DataFrame; mode::Symbol = :mean)

Evaluate a fitted [`GaussianProcess`](@ref) at the input locations in `data`,
append or replace prediction columns in `data`, and return `nothing`. Input
columns are selected using `gp.inputs` and transformed using `gp.transform`.
Predictions are on the original output scale; predictive variances include
`gp.σ²` observation noise.

# Arguments
- `gp`: Fitted Gaussian-process model.
- `data`: Prediction locations containing the columns named in `gp.inputs`.

# Keyword Arguments
- `mode`: Prediction columns to write, using `gp.output` as the name prefix:
  - `:mean`: Predictive mean in `<output>_mean` (default).
  - `:var`: Predictive variance in `<output>_var`.
  - `:mean_and_var`: Both prediction columns.
  Other modes raise an `ArgumentError`.

Use [`sample!`](@ref) to draw joint samples from the predictive distribution.

# Examples
```jldoctest
julia> data = DataFrame(x = [0.0, 1.0, 2.0], y = [0.0, 1.0, 4.0]);

julia> gp = GaussianProcess(data, :y; σ² = 1.0e-3, learn_hyperparameters = false);

julia> predictions = DataFrame(x = [0.5, 1.5]);

julia> evaluate!(gp, predictions; mode = :mean_and_var);

julia> propertynames(predictions) == [:x, :y_mean, :y_var]
true

julia> all(isfinite, predictions.y_mean) && all(>=(0), predictions.y_var)
true
```
"""
function evaluate!(
        gp::GaussianProcess,
        data::DataFrame;
        mode::Symbol = :mean,
    )
    x = transform(data[:, gp.inputs], gp.transform)
    finite_projection = gp.posterior(x, gp.σ²)

    if mode === :mean
        μ = mean(finite_projection)
        col = Symbol(string(gp.output, "_mean"))
        data[!, col] = μ
    elseif mode === :var
        σ² = var(finite_projection)
        col = Symbol(string(gp.output, "_var"))
        data[!, col] = σ²
    elseif mode === :mean_and_var
        μ = mean(finite_projection)
        σ² = var(finite_projection)
        col_mean = Symbol(string(gp.output, "_mean"))
        col_var = Symbol(string(gp.output, "_var"))
        data[!, col_mean] = μ
        data[!, col_var] = σ²
    else
        throw(ArgumentError("Unknown `GaussianProcess` evaluation mode: $mode"))
    end
    return nothing
end

"""
    sample!(gp::GaussianProcess, data::DataFrame, n_samples::Int = 1)

Draw `n_samples` independent realizations from the fitted GP's joint posterior
predictive distribution at the input locations in `data`. Each realization
contains correlated values across the rows of `data` and is written to a column
named `<output>_sample_<i>`, where `<output>` is `gp.output` and `i` starts at 1.
Return `nothing`.

Inputs are selected in `gp.inputs` order and transformed using `gp.transform`.
Samples are on the original output scale and include observation noise with
variance `gp.σ²`. Existing sample columns with the same names are replaced;
other columns are left unchanged. The fitted posterior is not modified.

# Arguments
- `gp`: Fitted [`GaussianProcess`](@ref).
- `data`: Prediction locations containing the columns named in `gp.inputs`.
- `n_samples`: Nonnegative number of realizations to draw. Defaults to `1`;
  `0` adds no columns.

Sampling uses Julia's default random-number generator; use `Random.seed!` for
reproducibility. Drawing joint samples requires factoring a covariance matrix
whose size is the number of rows in `data`. Very close or repeated locations
can cause numerical difficulties when the observation-noise variance is too small.

# Examples
```jldoctest
julia> training = DataFrame(x = [0.0, 1.0, 2.0], y = [0.0, 1.0, 4.0]);

julia> gp = GaussianProcess(training, :y; σ² = 1.0e-3, learn_hyperparameters = false);

julia> draws = DataFrame(x = [0.5, 1.5]);

julia> sample!(gp, draws, 2);

julia> propertynames(draws) == [:x, :y_sample_1, :y_sample_2]
true

julia> all(isfinite, Matrix(draws[:, [:y_sample_1, :y_sample_2]]))
true
```
"""
function sample!(
        gp::GaussianProcess,
        data::DataFrame,
        n_samples::Int = 1
    )

    x = transform(data[:, gp.inputs], gp.transform)
    finite_projection = gp.posterior(x, gp.σ²)

    samples = rand(finite_projection, n_samples)
    cols = [Symbol(string(gp.output, "_sample_", i)) for i in 1:n_samples]
    foreach(
        (col, sample) -> data[!, col] = sample,
        cols, eachcol(samples)
    )

    return nothing
end

function transform(data::DataFrame, dt::ZScoreTransform)
    return StatsBase.transform(dt, permutedims(Matrix(data)))
end

function transform(data::DataFrame, dt::Vector{<:UQInput})
    df = copy(data)
    to_standard_normal_space!(dt, df)
    return permutedims(Matrix(df))
    return StatsBase.transform(dt, permutedims(Matrix(data)))
end

function transform(data::DataFrame, ::Nothing)
    return permutedims(Matrix(data))
end
