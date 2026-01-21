abstract type AbstractDataTransform end

# ---------------- Input/Output transforms ----------------
"""
    IdentityTransform()

A standardization transform that applies an identity transform to data.

Used as an input or output transformation in a [`GaussianProcess`](@ref). 
Internally, the `DataStandardizer` constructs the functions required for evaluation.

# Examples
```jldoctest
julia> id = UncertaintyQuantification.IdentityTransform()
IdentityTransform()
```
"""
struct IdentityTransform <: AbstractDataTransform end

"""
    ZScoreTransform()

A standardization transform that rescales data to zero mean and unit variance. 

Used as an input or output transformation in a [`GaussianProcess`](@ref). 
Internally, the `DataStandardizer` constructs the functions required for evaluation.

# Examples
```jldoctest
julia> zscore = UncertaintyQuantification.ZScoreTransform()
ZScoreTransform()
```
"""
struct ZScoreTransform <: AbstractDataTransform end

"""
    UnitRangeTransform()

A standardization transform that rescales data to the [0, 1] range.

Used as an input or output transformation in a [`GaussianProcess`](@ref). 
Internally, the `DataStandardizer` constructs the functions required for evaluation.

# Examples
```jldoctest
julia> unitrange = UncertaintyQuantification.UnitRangeTransform()
UnitRangeTransform()
```
"""
struct UnitRangeTransform <: AbstractDataTransform end

"""
    StandardNormalTransform()

A normalization transform that transforms data to the standard normal space. 

Can only be used as an input transformation in a [`GaussianProcess`](@ref) for inputs of type [`RandomVariable`](@ref). 
Internally, the `DataStandardizer` constructs the function required for evaluation.

# Examples
```jldoctest
julia> sns = StandardNormalTransform()
StandardNormalTransform()
```
"""
struct StandardNormalTransform <: AbstractDataTransform end

struct InputTransform{T <: AbstractDataTransform} end
InputTransform(::Type{T}) where {T <: AbstractDataTransform} = InputTransform{T}()
InputTransform(x::AbstractDataTransform) = InputTransform(typeof(x))

struct OutputTransform{T <: AbstractDataTransform} end
OutputTransform(::Type{T}) where {T <: AbstractDataTransform} = OutputTransform{T}()
OutputTransform(x::AbstractDataTransform) = OutputTransform(typeof(x))

# ---
# # Developer Note

# DataStandardizer bundles input and output transformation functions for Gaussian process models.

# # Fields

# - `fᵢ` - function applied to input data.
# - `fₒ` - function applied to output data.
# - `fₒ⁻¹` - inverse function for the output transformation.
# - `var_fₒ⁻¹` - function for transforming output variances.

# !!! note "Inverse output transformations"

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
# Hence, `fₒ⁻¹` and `var_fₒ⁻¹` must be implemented separately.

# # Constructor

#     DataStandardizer(
#         data::DataFrame, 
#         input::Union{Symbol, Vector{<:Symbol}, UQInput, Vector{<:UQInput}}, 
#         output::Symbol, 
#         input_transform::InputTransform, 
#         output_transform::OutputTransform
#     )

# Constructs a set of transformation functions from the provided data and user-specified input/output transforms. 
# Internally, it uses `build_datatransform` to create the actual functions.

# # Purpose

# This struct allows [`GaussianProcess`](@ref) models to consistently apply input and output transformations 
#     (like `ZScoreTransform` or `IdentityTransform`) while keeping the API simple for end-users. 
# The `AbstractDataTransform` structs signal the desired behavior, and `DataStandardizer` converts them into callable functions for internal use.
# ---
struct DataStandardizer
    fᵢ::Function
    fₒ::Function
    fₒ⁻¹::Function
    var_fₒ⁻¹::Function
end

function DataStandardizer(
    data::DataFrame,
    input::Union{Symbol, Vector{<:Symbol}, UQInput, Vector{<:UQInput}},
    output::Symbol,
    input_transform::InputTransform, 
    output_transform::OutputTransform
)
    fᵢ = build_datatransform(data, input, input_transform)
    fₒ, fₒ⁻¹, var_fₒ⁻¹ = build_datatransform(data, output, output_transform)
    return DataStandardizer(fᵢ, fₒ, fₒ⁻¹, var_fₒ⁻¹)
end


# ---
# build_datatransform(data, input/output, transform)
#
# Returns a function (or pair of functions for outputs) that applies the specified 
# transformation to a dataframe.
# ---

# ---------------- Input ----------------
# No input transformation
function build_datatransform(
    ::DataFrame, 
    input::Union{Symbol, Vector{<:Symbol}}, 
    ::InputTransform{IdentityTransform}
)
    f(df::DataFrame) = to_gp_format(
        dataframe_to_array(df, input)
    )
    return f
end

build_datatransform(
    data::DataFrame, 
    input::Union{UQInput, Vector{<:UQInput}}, 
    transform::InputTransform{IdentityTransform}
 ) = build_datatransform(data, names(input), transform)

 # ZScore input transformation
function build_datatransform(
    data::DataFrame,
    input::Union{Symbol, Vector{<:Symbol}},
    ::InputTransform{ZScoreTransform}
)
    zscore_transform = fit(
        StatsBase.ZScoreTransform, 
        dataframe_to_array(data, input); 
        dims=1
    )
    f(df::DataFrame) = to_gp_format(
        StatsBase.transform(
            zscore_transform, 
            dataframe_to_array(df, input)
        )
    )
    return f
end

build_datatransform(
    data::DataFrame, 
    input::Union{UQInput, Vector{<:UQInput}}, 
    transform::InputTransform{ZScoreTransform}
 ) = build_datatransform(data, names(input), transform)

# UnitRange input transformation
function build_datatransform(
    data::DataFrame,
    input::Union{Symbol, Vector{<:Symbol}},
    ::InputTransform{UnitRangeTransform}
)
    unitrange_transform = fit(
        StatsBase.UnitRangeTransform, 
        dataframe_to_array(data, input); 
        dims=1
    )
    f(df::DataFrame) = to_gp_format(
        StatsBase.transform(
            unitrange_transform, 
            dataframe_to_array(df, input)
        )
    )
    return f
end

build_datatransform(
    data::DataFrame, 
    input::Union{UQInput, Vector{<:UQInput}}, 
    transform::InputTransform{UnitRangeTransform}
 ) = build_datatransform(data, names(input), transform)

# SNS input transform
function build_datatransform(
    ::DataFrame,
    input::Union{UQInput, Vector{<:UQInput}},
    ::InputTransform{StandardNormalTransform}
)
    function f(df::DataFrame)
        df_copy = copy(df)
        to_standard_normal_space!(input, df_copy)
        return to_gp_format(
            dataframe_to_array(df_copy, names(input))    
        )
    end
    return f
end

# ---------------- Output ----------------
# No output transformation
function build_datatransform(
    ::DataFrame, 
    output::Symbol, 
    ::OutputTransform{IdentityTransform}
)
    f(df::DataFrame) = to_gp_format(
        dataframe_to_array(df, output)
    )
    f⁻¹(Y::AbstractArray) = Y
    var_f⁻¹(Y::AbstractArray) = Y
    return (f, f⁻¹, var_f⁻¹)
end

# ZScore output transformation
function build_datatransform(
    data::DataFrame, 
    output::Symbol,
    ::OutputTransform{ZScoreTransform}
)
    zscore_transform = fit(
        StatsBase.ZScoreTransform, 
        dataframe_to_array(data, output); 
        dims=1
    )
    f(df::DataFrame) = to_gp_format(
        StatsBase.transform(
            zscore_transform, 
            dataframe_to_array(df, output)
        )
    )
    f⁻¹(Y::AbstractArray) = StatsBase.reconstruct(zscore_transform, Y)
    var_f⁻¹(Y::AbstractArray) = only(zscore_transform.scale)^2 * Y 
    return (f, f⁻¹, var_f⁻¹)
end

# UnitRange output transformation
function build_datatransform(
    data::DataFrame, 
    output::Symbol,
    ::OutputTransform{UnitRangeTransform}
)
    unitrange_transform = fit(
        StatsBase.UnitRangeTransform, 
        dataframe_to_array(data, output); 
        dims=1
    )
    f(df::DataFrame) = to_gp_format(
        StatsBase.transform(
            unitrange_transform, 
            dataframe_to_array(df, output)
        )
    )
    f⁻¹(Y::AbstractArray) = StatsBase.reconstruct(unitrange_transform, Y)
    var_f⁻¹(Y::AbstractArray) = only(unitrange_transform.scale)^2 * Y 
    return (f, f⁻¹, var_f⁻¹)
end

function build_datatransform(
    ::DataFrame, 
    ::Symbol, 
    ::OutputTransform{StandardNormalTransform}
)
    throw(ArgumentError(
        "StandardNormalTransform is only valid for input transforms."
    ))
end

# ---------------- Utility ----------------
to_gp_format(x::Vector) = x
to_gp_format(x::Matrix) = RowVecs(x)

dataframe_to_array(df::DataFrame, name::Symbol) = df[:, name]
dataframe_to_array(df::DataFrame, names::Vector{<:Symbol}) = length(names) == 1 ? x = dataframe_to_array(df, only(names)) : x = Matrix(df[:, names])