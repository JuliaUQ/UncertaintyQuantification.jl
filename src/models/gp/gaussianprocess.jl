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
- `σ²`: The noise variance. Defaults to 1.0e-10.
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

function transform(data::DataFrame, dt::ZScoreTransform)
    return StatsBase.transform(dt, permutedims(Matrix(data)))
end

function transform(data::DataFrame, dt::Vector{<:UQInput})
    df = copy(data)
    to_standard_normal_space!(dt, df)
    return permutedims(Matrix(df))
    return StatsBase.transform(dt, permutedims(Matrix(data)))
end
