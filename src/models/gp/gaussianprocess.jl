struct GaussianProcess <: UQModel
    posterior::AbstractGPs.PosteriorGP
    output::Symbol
    σ²::Float64
    input_transformer::GaussianProcessInputTransformer
    output_transformer::GaussianProcessOutputTransformer
    training_data::DataFrame
end

function Base.show(io::IO, gp::GaussianProcess)
    print(io, "GaussianProcess(")
    print(io, "mean=$(gp.posterior.prior.mean), ")
    print(io, "kernel=$(gp.posterior.prior.kernel), ")
    print(io, "input=$(gp.input_transformer.input)), ")
    print(io, "output=$(gp.output), ")
    print(io, "n_datapoints=$(size(gp.training_data, 1))")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", gp::GaussianProcess)
    println(io, "GaussianProcess")
    println(io, "   mean: $(gp.posterior.prior.mean)")
    println(io, "   kernel: $(gp.posterior.prior.kernel)")
    println(io, "   input: $(gp.input_transformer.input)")
    println(io, "   output: $(gp.output)")
    print(io, "   n_datapoints: $(size(gp.training_data, 1))")
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
    GaussianProcess(data::DataFrame, output::Symbol; kwargs...)

Constructs a `GaussianProcess` model with the specified data and output variable.

# Arguments
- `data`: A `DataFrame` containing the input and output data.
- `output`: The output variable for the Gaussian process.

# Keyword Arguments
- `mean_fct`: The mean function for the Gaussian process. Defaults to `ZeroMean`.
- `kernel`: The kernel for the Gaussian process. Defaults to `SqExponentialKernel`.
- `input_transform`: The transformation to apply to the input variables. Defaults to `IdentityTransformChoice`.
- `output_transform`: The transformation to apply to the output variables. Defaults to `IdentityTransformChoice`.
- `σ²`: The noise variance. Defaults to 0.0.
- `learn_noise`: Whether to learn the noise variance. Defaults to `false`.
- `learn_hyperparameters`: Whether to learn the hyperparameters. Defaults to `true`.
- `optimizer`: The optimization algorithm used to learn the hyperparameters. Defaults to `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations=100, show_trace=false))`.

# Examples
```jldoctest
julia> mean_fct = ConstMean(0.0);

julia> kernel = SqExponentialKernel();

julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101]);

julia> gp_model = GaussianProcess(data, :y; mean_fct = mean_fct, kernel = kernel, σ² = 1.0e-3);
```
"""
function GaussianProcess(
        data::DataFrame,
        output::Symbol;
        mean_fct::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )

    gp = GP(mean_fct, kernel)

    return GaussianProcess(
        gp, data, output;
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer
    )
end

"""
    GaussianProcess(
        gp::GP,
        data::DataFrame,
        output::Symbol;
        kwargs...
    )

Constructs a Gaussian process model for the given data and output variable using a pre-defined Gaussian process.

# Arguments
- `gp`: A Gaussian process object, typically from `AbstractGPs`, defining the kernel and mean.
- `data`: A `DataFrame` containing the input and output data.
- `output`: The name of the output (as a `Symbol`) to be modeled as the response variable.

# Keyword Arguments
- `input_transform`: Choice of transformation that is applied to input features before fitting.
  Defaults to [`IdentityTransformChoice()`](@ref).
- `output_transform`: Choice of transformation that is applied to output data before fitting.
  Defaults to [`IdentityTransformChoice()`](@ref).
- `σ²`: The noise variance. Defaults to 0.0.
- `learn_noise`: Whether to learn the noise variance. Defaults to false.
- `learn_hyperparameters`: Whether to learn the hyperparameters. Defaults to true.
- `optimizer`: The optimizer for hyperparameter optimization. Defaults to `MaximumLikelihoodEstimation` with `Optim.LBFGS()` and `Optim.Options(; iterations=100, show_trace=false)`.

# Examples
```jldoctest
julia> gp = GP(0.0, SqExponentialKernel());

julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101]);

julia> gp_model = GaussianProcess(gp, data, :y);
```
"""
function GaussianProcess(
        gp::GP,
        data::DataFrame,
        output::Symbol;
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
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

    input = propertynames(data[:, Not(output)]) # Is this always the case?

    # build in- and output transforms
    input_transformer = fit_input_transform(data, input, input_transform)
    output_transformer = fit_output_transform(data, output, output_transform)

    # transform data
    x = transform(data, input_transformer)
    y = transform(data, output_transformer)

    posterior_gp = nothing

    # optimize hyperparameters
    if learn_hyperparameters
        _gp = optimize_hyperparameters(PriorGP(gp, σ², learn_noise), x, y, optimizer)
        σ² = _gp.σ²
        # _gp is a PriorGP object, callig it directly involves the noise, so no need to add σ² again
        posterior_gp = posterior(_gp(x), y)
    else
        # gp has to be called with noise since it is an AbstrctGPs.GP object, not a PriorGP object
        posterior_gp = posterior(gp(x, σ²), y)
    end

    return GaussianProcess(
        posterior_gp,
        output,
        σ²,
        input_transformer,
        output_transformer,
        data
    )
end

"""
    GaussianProcess(input::Union{UQInput, Vector{<:UQInput}}, model::Union{UQModel, Vector{<:UQModel}}, output::Symbol; kwargs...)

Constructs a `GaussianProcess` model with the specified input, model, and output.

# Arguments
- `input`: The input variable(s) for the Gaussian process.
- `model`: The model(s) to be used for the Gaussian process.
- `output`: The output variable for the Gaussian process.

# Keyword Arguments
- `n_design_points`: Number of design points to sample from the input space. Defaults to 10.
- `experimental_design`: The strategy utilized for sampling the input variables. Defaults to `LatinHypercubeSampling`.
- `mean_fct`: The mean function for the Gaussian process. Defaults to `ZeroMean`.
- `kernel`: The kernel for the Gaussian process. Defaults to `SqExponentialKernel`.
- `input_transform`: The transformation to apply to the input variables. Defaults to `IdentityTransformChoice`.
- `output_transform`: The transformation to apply to the output variables. Defaults to `IdentityTransformChoice`.
- `σ²`: The noise variance. Defaults to 0.0.
- `learn_noise`: Whether to learn the noise variance. Defaults to `false`.
- `learn_hyperparameters`: Whether to learn the hyperparameters. Defaults to `true`.
- `optimizer`: The optimization algorithm used to learn the hyperparameters. Defaults to `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations=100, show_trace=false))`.

# Examples
```jldoctest
julia> begin # hide
           mean_fct = ConstMean(0.0)
           kernel = SqExponentialKernel()
           x = RandomVariable(Uniform(0, 5), :x)
           model = Model(df -> sin.(df.x), :y)
           design = LatinHypercubeSampling(10)
           gp_model = GaussianProcess(x, model, :y; experimental_design = design, mean_fct = mean_fct, kernel = kernel)
           nothing # hide
       end # hide
```
"""
function GaussianProcess(
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol;
        n_design_points::Int = 10,
        experimental_design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments} = LatinHypercubeSampling(n_design_points),
        mean_fct::AbstractGPs.MeanFunction = ZeroMean(),
        kernel::Kernel = SqExponentialKernel(),
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )

    gp = GP(mean_fct, kernel)
    return GaussianProcess(
        gp, input, model, output;
        experimental_design = experimental_design,
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer
    )

end

"""
    GaussianProcess(
        gp::GP,
        input::Vector{<:UQInput},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol;
        kwargs...
    )

Constructs a Gaussian process model for the given input and model. Evaluates the model using specified experimental design.

# Arguments
- `gp`: A Gaussian process object, typically from `AbstractGPs`, defining the kernel and mean.
- `input`: Single input or vector of inputs. The Gaussian process will only consider inputs of type [`RandomVariable`](@ref) as input features.
- `model`: Single model or vector of models of supertype [`UQModel`](@ref) that the Gaussian process is supposed to model.
- `output`: The name of the output (as a `Symbol`) to be modeled as the response variable.

# Keyword Arguments
- `n_design_points`: Number of design points to sample from the input space. Defaults to 10.
- `experimental_design`: The strategy utilized for sampling the input variables.
- `input_transform`: Choice of transformation that is applied to input features before fitting.
  Defaults to [`IdentityTransformChoice()`](@ref).
- `output_transform`: Choice of transformation that is applied to output data before fitting.
  Defaults to [`IdentityTransformChoice()`](@ref).
- `σ²`: The noise variance. Defaults to 0.0.
- `learn_noise`: Whether to learn the noise variance. Defaults to `false`.
- `learn_hyperparameters`: Whether to learn the hyperparameters. Defaults to `true`.
- `optimizer`: The optimization algorithm used to learn the hyperparameters. Defaults to `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations=100, show_trace=false))`.

# Examples
```jldoctest
julia> begin # hide
           gp = GP(0.0, SqExponentialKernel())
           x = RandomVariable(Uniform(0, 5), :x)
           model = Model(df -> sin.(df.x), :y)
           design = LatinHypercubeSampling(10)
           gp_model = GaussianProcess(gp, x, model, :y; experimental_design = design)
           nothing # hide
       end # hide
```
"""
function GaussianProcess(
        gp::GP,
        input::Vector{<:UQInput},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol;
        n_design_points::Int = 10,
        experimental_design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments} = LatinHypercubeSampling(n_design_points),
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )
    # build DataFrame
    data = sample(input, experimental_design)
    evaluate!(model, data)

    # Repeated deterministic input will break the GP kernel
    random_input = names(filter(i -> isa(i, RandomVariable), input))

    return GaussianProcess(
        gp, data[!, [random_input..., output]], output;
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer
    )
end

# Helper constructor to wrap `input` into a Vector
function GaussianProcess(
        gp::GP,
        input::UQInput,
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol;
        n_design_points::Int = 10,
        experimental_design::Union{AbstractMonteCarlo, AbstractDesignOfExperiments} = LatinHypercubeSampling(n_design_points),
        input_transform::AbstractTransformChoice = IdentityTransformChoice(),
        output_transform::AbstractTransformChoice = IdentityTransformChoice(),
        σ²::Float64 = 1.0e-10,
        learn_noise::Bool = false,
        learn_hyperparameters::Bool = true,
        optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))
    )
    return GaussianProcess(
        gp, [input], model, output;
        experimental_design = experimental_design,
        input_transform = input_transform,
        output_transform = output_transform,
        σ² = σ²,
        learn_noise = learn_noise,
        learn_hyperparameters = learn_hyperparameters,
        optimizer = optimizer
    )
end

"""
    evaluate!(gp::GaussianProcess, data::DataFrame; mode::Symbol = :mean, n_samples::Int = 1)

Evaluates a fitted [`GaussianProcess`](@ref) model at the specified input locations.

# Arguments
- `gp`: Trained Gaussian process model to be evaluated.
- `data`: A `DataFrame` containing the input locations at which predictions are computed.

# Keyword Arguments
- `mode`: A `Symbol` specifying the type of output to return.
    Supported options are:
    - `:mean` - predictive mean (default)
    - `:var` - predictive variance
    - `:mean_and_var` - both mean and variance
    - `:sample` - random samples from the predictive distribution
- `n_samples`: Number of samples to draw when `mode = :sample`. Ignored otherwise.
    (Note: Sampling can be unstable when input locations are very close together, leading to numerical issues in the covariance matrix.)

# Examples
```jldoctest
julia> gp = GP(0.0, SqExponentialKernel());

julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101]);

julia> gp_model = GaussianProcess(gp, data, :y; σ² = 1.0e-3);

julia> df = DataFrame(x = [0.5, 1.5, 2.5, 5.5, 8.5]);

julia> evaluate!(gp_model, df; mode = :mean_and_var);
```
"""
function evaluate!(
        gp::GaussianProcess,
        data::DataFrame;
        mode::Symbol = :mean,
        n_samples::Int = 1
    )
    x = transform(data, gp.input_transformer)
    finite_projection = gp.posterior(x, gp.σ²)

    if mode === :mean
        μ = mean(finite_projection)
        col = Symbol(string(gp.output, "_mean"))
        data[!, col] = inverse_transform(μ, gp.output_transformer)
    elseif mode === :var
        σ² = var(finite_projection)
        col = Symbol(string(gp.output, "_var"))
        data[!, col] = variance_inverse_transform(σ², gp.output_transformer)
    elseif mode === :mean_and_var
        μ = mean(finite_projection)
        σ² = var(finite_projection)
        col_mean = Symbol(string(gp.output, "_mean"))
        col_var = Symbol(string(gp.output, "_var"))
        data[!, col_mean] = inverse_transform(μ, gp.output_transformer)
        data[!, col_var] = variance_inverse_transform(σ², gp.output_transformer)
    elseif mode === :sample
        samples = rand(finite_projection, n_samples)
        cols = [Symbol(string(gp.output, "_sample_", i)) for i in 1:n_samples]
        foreach(
            (col, sample) -> data[!, col] = inverse_transform(sample, gp.output_transformer),
            cols, eachcol(samples)
        )
    else
        throw(ArgumentError("Unknown `GaussianProcess` evaluation mode: $mode"))
    end

    return nothing
end
