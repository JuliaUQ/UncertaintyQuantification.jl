struct BasisFunctionModel{T<:AbstractBasis} <: UQModel
    b::T
    β::Vector{<:Real}
    inputs::Vector{Symbol}
    out::Symbol
    function BasisFunctionModel(
        df::DataFrame,
        out::Symbol,
        b::T,
        inputs::Vector{Symbol}=propertynames(df[:, Not(out)]),
    ) where {T<:AbstractBasis}
        X = permutedims(Matrix(df[:, inputs]))
        y = df[:, out]

        β = b(X)' \ y
        return new{T}(b, β, inputs, out)
    end
end

function evaluate!(bfm::BasisFunctionModel, df::DataFrame)
    x = permutedims(Matrix{Float64}(df[:, bfm.inputs])) # convert to matrix, sort by bfm.inputs
    out = map(x -> dot(x, bfm.β), eachcol(bfm.b(x)))

    df[!, bfm.out] = out
    return nothing
end

