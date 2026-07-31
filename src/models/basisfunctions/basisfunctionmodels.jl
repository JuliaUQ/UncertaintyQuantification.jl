"""
    LinearBasisFunctionModel(
        df::DataFrame, out::Symbol, b:<AbstractBasis, inputs::Vector{Symbol}=propertynames(df[:, Not(out)])
    )

Construct a linear basis function model for the data in `df` using the basis `b`.By default the input
variables are assumed to be all columns of the `DataFrame` except for `out`.
"""
struct LinearBasisFunctionModel{T<:AbstractBasis} <: UQModel
    b::T
    β::Vector{<:Real}
    inputs::Vector{Symbol}
    out::Symbol
end

function LinearBasisFunctionModel(
    df::DataFrame, out::Symbol, b::T, inputs::Vector{Symbol}=propertynames(df[:, Not(out)])
) where {T<:AbstractBasis}
    X = permutedims(Matrix(df[:, inputs]))
    y = df[:, out]

    β = b(X)' \ y
    return LinearBasisFunctionModel(b, β, inputs, out)
end

function evaluate!(bfm::LinearBasisFunctionModel, df::DataFrame)
    x = permutedims(Matrix{Float64}(df[:, bfm.inputs])) # convert to matrix, sort by bfm.inputs
    out = map(x -> dot(x, bfm.β), eachcol(bfm.b(x)))

    df[!, bfm.out] = out
    return nothing
end

name(bfm::LinearBasisFunctionModel) = bfm.out
