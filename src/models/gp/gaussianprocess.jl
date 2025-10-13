struct GaussianProcess <: UQModel
    gp::AbstractGPs.PosteriorGP
    input::Union{Symbol, Vector{Symbol}}
    output::Symbol
    standardizer::DataStandardizer
end

"""
    GaussianProcess(
        gp::Union{AbstractGPs.GP, NoisyGP}, 
        data::DataFrame, 
        output::Symbol; 
        input_transform::AbstractDataTransform = IdentityTransform(), 
        output_transform::AbstractDataTransform = IdentityTransform(),
        optimization::AbstractHyperparameterOptimization = NoHyperparameterOptimization()
    )

Constructs a Gaussian process model for the given data and output variable.

# Arguments
- `gp`: A Gaussian process object, typically from `AbstractGPs`, defining the kernel and mean.
- `data`: A `DataFrame` containing the input and output data.
- `output`: The name of the output (as a `Symbol`) to be modeled as the response variable.

# Keyword Arguments
- `input_transform`: Transformation applied to input features before fitting.
  Defaults to [`IdentityTransform()`](@ref).
- `output_transform`: Transformation applied to output data before fitting.
  Defaults to [`IdentityTransform()`](@ref).
- `optimization`: Strategy for hyperparameter optimization.
  Defaults to `NoHyperparameterOptimization()`.

# Examples
```jldoctest
julia> using AbstractGPs

julia> gp = with_gaussian_noise(GP(0.0, SqExponentialKernel()), 1e-3);

julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101]);

julia> gp_model = GaussianProcess(gp, data, :y);
```
"""
function GaussianProcess(
    gp::Union{AbstractGPs.GP, NoisyGP},
    data::DataFrame,
    output::Symbol;
    input_transform::AbstractDataTransform=IdentityTransform(),
    output_transform::AbstractDataTransform=IdentityTransform(),
    optimization::AbstractHyperparameterOptimization=NoHyperparameterOptimization()
) 
    input = propertynames(data[:, Not(output)]) # Is this always the case?

    # build in- and output transforms
    dts = DataStandardizer(
        data, input, output, 
        InputTransform(input_transform), 
        OutputTransform(output_transform)
    )

    # transform data
    x = dts.fᵢ(data)
    y = dts.fₒ(data)

    # build posterior gp
    optimized_gp = optimize_hyperparameters(gp, x, y, optimization)
    posterior_gp = posterior(optimized_gp(x), y)
    return GaussianProcess(
        posterior_gp,
        input,
        output,
        dts
    )
end

"""
    GaussianProcess(
        gp::Union{AbstractGPs.GP, NoisyGP}, 
        input::Union{UQInput, Vector{<:UQInput}},
        model::Union{UQModel, Vector{<:UQModel}},
        output::Symbol,
        experimentaldesign::Union{AbstractMonteCarlo, AbstractDesignOfExperiments}; 
        input_transform::AbstractDataTransform = IdentityTransform(), 
        output_transform::AbstractDataTransform = IdentityTransform(),
        optimization::AbstractHyperparameterOptimization = NoHyperparameterOptimization()
    )

Constructs a Gaussian process model for the given input and model. Evaluates the model using specified experimental design.

# Arguments
- `gp`: A Gaussian process object, typically from `AbstractGPs`, defining the kernel and mean.
- `input`: Single input or vector of inputs. The Gaussian process will only consider inputs of type ['RandomVariable](@ref) as input features.
- `model`: Single model or vector of models of supertype [`UQModel`](@ref) that the Gaussian process is supposed to model.
- `output`: The name of the output (as a `Symbol`) to be modeled as the response variable.
- `experimentaldesign`: The strategy utilized for sampling the input variables.

# Keyword Arguments
- `input_transform`: Transformation applied to input features before fitting.
  Defaults to [`IdentityTransform()`](@ref).
- `output_transform`: Transformation applied to output data before fitting.
  Defaults to [`IdentityTransform()`](@ref).
- `optimization`: Strategy for hyperparameter optimization.
  Defaults to `NoHyperparameterOptimization()`.

# Examples
```jldoctest
julia> using AbstractGPs

julia> gp = with_gaussian_noise(GP(0.0, SqExponentialKernel()), 1e-3);

julia> input = RandomVariable(Uniform(0, 5), :x);
        
julia> model = Model(df -> sin.(df.x), :y);

julia> design = LatinHypercubeSampling(10);

julia> gp_model = GaussianProcess(gp, input, model, :y, design);
```
"""
function GaussianProcess(
    gp::Union{AbstractGPs.GP, NoisyGP},
    input::Union{UQInput, Vector{<:UQInput}},
    model::Union{UQModel, Vector{<:UQModel}},
    output::Symbol,
    experimentaldesign::Union{AbstractMonteCarlo, AbstractDesignOfExperiments};
    input_transform::AbstractDataTransform=IdentityTransform(),
    output_transform::AbstractDataTransform=IdentityTransform(),
    optimization::AbstractHyperparameterOptimization=NoHyperparameterOptimization()
)
    # build DataFrame
    data = sample(input, experimentaldesign)
    evaluate!(model, data)

    # Repeated deterministic input will break the GP kernel
    filter!(i -> isa(i, RandomVariable), input)

    # build in- and output transforms
    dts = DataStandardizer(
        data, input, output, 
        InputTransform(input_transform), 
        OutputTransform(output_transform)
    )

    # transform data
    x = dts.fᵢ(data)
    y = dts.fₒ(data)

    # build posterior gp
    optimized_gp = optimize_hyperparameters(gp, x, y, optimization)
    posterior_gp = posterior(optimized_gp(x), y)

    return GaussianProcess(
        posterior_gp,
        names(input),
        output,
        dts
    )
end

# what should this calculate? Calculates only mean for now
function evaluate!(
    gp::GaussianProcess, 
    data::DataFrame;
    mode::Symbol = :mean,
    n_samples::Int = 1
)
    x = gp.standardizer.fᵢ(data)
    finite_projection = gp.gp(x)

    if mode === :mean
        μ = mean(finite_projection)
        col = Symbol(string(gp.output, "_mean"))
        data[!, gp.output] = gp.standardizer.fₒ⁻¹(μ)
    elseif mode === :var
        σ² = var(finite_projection)
        col = Symbol(string(gp.output, "_var"))
        data[!, col] = gp.standardizer.var_fₒ⁻¹(σ²)
    elseif mode === :mean_and_var
        μ = mean(finite_projection)
        σ² = var(finite_projection)
        col_mean = Symbol(string(gp.output, "_mean"))
        col_var = Symbol(string(gp.output, "_var"))
        data[!, col_mean] = gp.standardizer.fₒ⁻¹(μ)
        data[!, col_var] = gp.standardizer.var_fₒ⁻¹(σ²)
    elseif mode === :sample
        samples = rand(finite_projection, n_samples)
        cols = [Symbol(string(gp.output, "_sample_", i)) for i in 1:n_samples]
        foreach(
            (colᵢ, sampleᵢ) -> data[!, colᵢ] = gp.standardizer.fₒ⁻¹(sampleᵢ), 
            cols, eachcol(samples)
        )
    else
        throw(ArgumentError("Unknown `GaussianProcess` evaluation mode: $mode"))
    end

    return nothing
end