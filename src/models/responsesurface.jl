"""
    ResponseSurface(data::DataFrame, dependendVarName::Symbol, deg::Int, dim::Int)

Creates a response surface using polynomial least squares regression with given degree.

# Examples
```jldoctest; filter = r\"(\\d*)\\.(\\d{12})\\d+\" => s\"\\1.\\2***\"
julia> data = DataFrame(x = 1:10, y = [1, 4, 10, 15, 24, 37, 50, 62, 80, 101]);

julia> rs = ResponseSurface(data, :y, 2)
ResponseSurface([0.48333333333332457, -0.23863636363636026, 1.0189393939393936], :y, [:x], 2, Monomials.Monomial[1, x1, x1²])
```
"""
const ResponseSurface = LinearBasisFunctionModel{MonomialBasis}

function ResponseSurface(
    data::DataFrame,
    output::Symbol,
    p::Int,
    inputs::Vector{Symbol}=propertynames(data[:, Not(output)]),
)
    if p < 0
        error("Degree(p) of ResponseSurface must be non-negative.")
    end

    basis = MonomialBasis(length(inputs), p; include_zero=true)
    return LinearBasisFunctionModel(data, output, basis, inputs)
end
