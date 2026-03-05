abstract type AbstractTransformChoice end

"""
    IdentityTransformChoice()

A standardization choice that specifies the application of an identity transform to data.

Used as an input or output transformation in a [`GaussianProcess`](@ref). 
Internally, the `DataStandardizer` constructs the functions required for evaluation.

# Examples
```jldoctest
julia> id = IdentityTransformChoice()
IdentityTransformChoice()
```
"""
struct IdentityTransformChoice <: AbstractTransformChoice end

"""
    ZScoreTransformChoice()

A standardization choice that specifies the application of a Z-score-transformation to data.

Used as an input or output transformation in a [`GaussianProcess`](@ref). 
Internally, the `DataStandardizer` constructs the functions required for evaluation.

# Examples
```jldoctest
julia> zscore = ZScoreTransformChoice()
ZScoreTransformChoice()
```
"""
struct ZScoreTransformChoice <: AbstractTransformChoice end

"""
    UnitRangeTransformChoice()

A standardization choice that specifies the application of a unit range transform to data.

Used as an input or output transformation in a [`GaussianProcess`](@ref). 
Internally, the `DataStandardizer` constructs the functions required for evaluation.

# Examples
```jldoctest
julia> unitrange = UnitRangeTransformChoice()()
UnitRangeTransformChoice()()
```
"""
struct UnitRangeTransformChoice <: AbstractTransformChoice end

"""
    StandardNormalTransformChoice()

A standardization choice that specifies the application of a standard normal transformation to data. 

Can only be used as an input transformation in a [`GaussianProcess`](@ref) for inputs of type [`RandomVariable`](@ref). 
Internally, the `DataStandardizer` constructs the function required for evaluation.

# Examples
```jldoctest
julia> sns = StandardNormalTransformChoice()
StandardNormalTransformChoice()
```
"""
struct StandardNormalTransformChoice <: AbstractTransformChoice end


# ---------------- Utility ----------------
to_gp_format(x::Vector) = x
to_gp_format(x::Matrix) = RowVecs(x)
dataframe_to_array(df::DataFrame, name::Symbol) = df[:, name]
dataframe_to_array(df::DataFrame, names::Vector{<:Symbol}) = length(names) == 1 ? x = dataframe_to_array(df, only(names)) : x = Matrix(df[:, names])

# Internal transform types for dispatching
struct NoTransform end
struct StandardNormalTransform end


# ---------------- Input transformation ----------------
# # Developer Note
# Gaussian process regression inputs are always transformed to Vector or RowVecs in the multivariate input case
struct GaussianProcessInputTransformer{T}
    transform::T
    input::Union{Symbol, Vector{<:Symbol}, UQInput, Vector{<:UQInput}}
end

# Fitting
fit_input_transform(
    ::DataFrame, 
    input::Union{Symbol, Vector{<:Symbol}}, 
    ::IdentityTransformChoice
) = GaussianProcessInputTransformer(NoTransform(), input)

function fit_input_transform(
    data::DataFrame, 
    input::Union{Symbol, Vector{<:Symbol}}, 
    ::ZScoreTransformChoice
) 
    transform = fit(
        StatsBase.ZScoreTransform, 
        dataframe_to_array(data, input); 
        dims=1
    )
    return GaussianProcessInputTransformer(transform, input)
end

function fit_input_transform(
    data::DataFrame, 
    input::Union{Symbol, Vector{<:Symbol}}, 
    ::UnitRangeTransformChoice
) 
    transform = fit(
        StatsBase.UnitRangeTransform, 
        dataframe_to_array(data, input); 
        dims=1
    )
    return GaussianProcessInputTransformer(transform, input)
end

fit_input_transform(
    ::DataFrame, 
    input::Union{Symbol, Vector{<:Symbol}}, 
    ::StandardNormalTransformChoice
) = throw(ArgumentError("Standard normal input transform is only valid for inputs of type UQInput"))

fit_input_transform(
    data::DataFrame, 
    input::Union{UQInput, Vector{<:UQInput}}, 
    choice::Union{IdentityTransformChoice, ZScoreTransformChoice, UnitRangeTransformChoice}
) = fit_input_transform(data, names(input), choice)

fit_input_transform(
    ::DataFrame, 
    input::Union{UQInput, Vector{<:UQInput}}, 
    ::StandardNormalTransformChoice
) = GaussianProcessInputTransformer(StandardNormalTransform(), input)

# Transforms
transform(
    data::DataFrame, 
    transformer::GaussianProcessInputTransformer{<:NoTransform}
) = to_gp_format(dataframe_to_array(data, transformer.input))

transform(
    data::DataFrame,
    transformer::Union{GaussianProcessInputTransformer{<:ZScoreTransform}, GaussianProcessInputTransformer{<:UnitRangeTransform}}
) = to_gp_format(
    StatsBase.transform(
            transformer.transform,
            dataframe_to_array(data, transformer.input)
        )
    )

function transform(
    data::DataFrame, 
    transformer::GaussianProcessInputTransformer{<:StandardNormalTransform}
)
    data_copy = copy(data)
    to_standard_normal_space!(transformer.input, data_copy)
    return to_gp_format(dataframe_to_array(data_copy, names(transformer.input)))
end 


# ---------------- Output transformation ----------------
# # Developer Note
# Gaussian process regression target outputs are always extracted from DataFrame and transformed to Vector
# Inverse transforms return Vectors that later get inserted in the provided DataFrame that is used in evaluate! method for GaussianProcess
struct GaussianProcessOutputTransformer{T}
    transform::T
    output::Symbol
end

# Fitting
fit_output_transform(
    ::DataFrame, 
    output::Symbol, 
    ::IdentityTransformChoice
) = GaussianProcessOutputTransformer(NoTransform(), output)

function fit_output_transform(
    data::DataFrame, 
    output::Symbol, 
    ::ZScoreTransformChoice
) 
    transform = fit(
        StatsBase.ZScoreTransform, 
        dataframe_to_array(data, output); 
        dims=1
    )
    return GaussianProcessOutputTransformer(transform, output)
end

function fit_output_transform(
    data::DataFrame, 
    output::Symbol, 
    ::UnitRangeTransformChoice
) 
    transform = fit(
        StatsBase.UnitRangeTransform, 
        dataframe_to_array(data, output); 
        dims=1
    )
    return GaussianProcessOutputTransformer(transform, output)
end

fit_output_transform(
    ::DataFrame, 
    ::Symbol, 
    ::StandardNormalTransformChoice
) = throw(ArgumentError("Standard normal transform for outputs is not possible"))

# Transforms
transform(
    data::DataFrame, 
    transformer::GaussianProcessOutputTransformer{NoTransform}
) = to_gp_format(dataframe_to_array(data, transformer.output))

transform(
    data::DataFrame,
    transformer::Union{GaussianProcessOutputTransformer{<:ZScoreTransform}, GaussianProcessOutputTransformer{<:UnitRangeTransform}}
) = to_gp_format(
    StatsBase.transform(
            transformer.transform,
            dataframe_to_array(data, transformer.output)
        )
    )

# Inverse transforms
# # Developer Note
# Gaussian process regression requires two distinct inverse transformations for the output:
# one for the mean predictions (this same transformation can also be applied to function samples) and one for the variance predictions.

# Consider a z-score transformation of output ``y``:
#     ```math
#     \tilde{y} = \frac{y - μ}{σ}.
#     ```
# To recover the mean of the untransformed output, we can simply apply the inverse transformation:
#     ```math
#     E[y] = E[σ\tilde{y} + μ] = σE[\tilde{y}] + μ.
#     ```
# Analogously, sampled functions ``\tilde{y}_s`` from the Gaussian process regression model can be transformed back:
#     ```math
#     y_s = σ\tilde{y}_s + μ.
#     ```
# The variance, however, is untransformed as follows:
#     ```math
#     Var[y] = E[(σ\tilde{y} + μ - E[σ\tilde{y} + μ])^2] = E[(σ^2(\tilde{y} - E[\tilde{y}])^2] = σ^2 Var[\tilde{y}]
#     ```
inverse_transform(
    data::AbstractArray, 
    ::GaussianProcessOutputTransformer{NoTransform}
) = data

inverse_transform(
    data::AbstractArray,
    transformer::Union{GaussianProcessOutputTransformer{<:ZScoreTransform}, GaussianProcessOutputTransformer{<:UnitRangeTransform}}
) = StatsBase.reconstruct(transformer.transform, data)

variance_inverse_transform(
    data::AbstractArray,
    ::GaussianProcessOutputTransformer{NoTransform}
) = data

variance_inverse_transform(
    data::AbstractArray,
    transformer::Union{GaussianProcessOutputTransformer{<:ZScoreTransform}, GaussianProcessOutputTransformer{<:UnitRangeTransform}}
) = only(transformer.transform.scale)^2 * data