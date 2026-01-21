"""
    TransportMap

A transport map used to transform between standard normal space Z and physical space X.

# Fields

- `map::AbstractTriangularMap`: The optimized triangular transport map.
- `target::MapTargetDensity`: The target density.
- `quadrature::AbstractQuadratureWeights`: The quadrature weights used in optimization.
- `names::Vector{Symbol}`: The names of the variables.
"""
struct TransportMap <: AbstractTransportMap
    map::AbstractTriangularMap
    target::MapTargetDensity
    quadrature::AbstractQuadratureWeights
    names::Vector{Symbol}
end

function Base.show(io::IO, tm::TransportMap)
    print(io, "TransportMap(")
    print(io, "map=$(tm.map), ")
    print(io, "target=$(tm.target), ")
    print(io, "quadrature=$(tm.quadrature), ")
    print(io, "names=$(tm.names)")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tm::TransportMap)
    println(io, "Transport Map:")
    println(io, "  Map: $(tm.map)")
    println(io, "  Target: $(tm.target)")
    println(io, "  Quadrature: $(tm.quadrature)")
    println(io, "  Names: $(tm.names)")
    return nothing
end

function _to_dataframe(X::AbstractMatrix{<:Real}, names::Vector{Symbol})
    return DataFrame(X, names)
end

function _to_dataframe(x::AbstractVector{<:Real}, names::Vector{Symbol})
    return DataFrame(permutedims(x), names)
end

function sample(tm::TransportMap, n::Integer=1)
    Z = randn(n, numberdimensions(tm.map))
    X = evaluate(tm, Z)
    return _to_dataframe(X, tm.names)
end

function to_physical_space!(tm::TransportMap, Z::DataFrame)
    X = evaluate(tm, Matrix(Z[!, tm.names]))
    Z[!, tm.names] .= X
    return nothing
end

function to_standard_normal_space!(tm::TransportMap, X::DataFrame)
    Z = inverse(tm, Matrix(X[!, tm.names]))
    X[!, tm.names] .= Z
    return nothing
end

# variance diagnostic to estimate the quality of the fit
function variancediagnostic(tm::TransportMap, Z::DataFrame)
    return variance_diagnostic(tm.map, tm.target, Matrix(Z[!, tm.names]))
end

# Fit a transport map from density
function mapfromdensity(
    transportmap::AbstractTriangularMap,
    target::MapTargetDensity,
    quadrature::AbstractQuadratureWeights,
    names::Vector{Symbol},
)
    optimize!(transportmap, target, quadrature)

    return TransportMap(transportmap, target, quadrature, names)
end

# todo: adaptive map construction (from density)



"""
    TransportMapFromSamples

A transport map constructed from samples in the physical space X which are mapped to the standard normal space Z.

# Fields
- `map::ComposedMap`: The optimized composed map.
- `samples::DataFrame`: The DataFrame of samples in the physical space X used to fit the map.
"""
struct TransportMapFromSamples <: AbstractTransportMap
    map::ComposedMap{LinearMap}
    samples::DataFrame
    names::Vector{Symbol}
end

# Fit a transport map from samples (no target, no auto-diff required)
function mapfromsamples(transportmap::AbstractTriangularMap, X::DataFrame)
    target_samples = Matrix(X)

    # First, fit a linear map
    linear_map = LinearMap(target_samples)
    # Optimize transportmap
    optimize!(transportmap, target_samples, linear_map)
    # define ComposedMap
    composed_map = ComposedMap(linear_map, transportmap)

    return TransportMapFromSamples(composed_map, X, Symbol.(names(X)))
end

# The direction are reversed
function to_physical_space!(tm::TransportMapFromSamples, Z::DataFrame)
    X = inverse(tm, Matrix(Z[!, tm.names]))
    Z[!, tm.names] .= X
    return nothing
end

function to_standard_normal_space!(tm::TransportMapFromSamples, X::DataFrame)
    Z = evaluate(tm, Matrix(X[!, tm.names]))
    X[!, tm.names] .= Z
    return nothing
end

function sample(tm::TransportMapFromSamples, n::Integer=1)
    Z = randn(n, numberdimensions(tm.map))
    X = inverse(tm, Z)
    return _to_dataframe(X, tm.names)
end

function Base.show(io::IO, tm::TransportMapFromSamples)
    print(io, "TransportMapFromSamples(")
    print(io, "map=$(tm.map), ")
    print(io, "names=$(tm.names) ")
    print(io, "number_samples=$(nrow(tm.samples))")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tm::TransportMapFromSamples)
    println(io, "Transport Map (from samples):")
    println(io, "  Map: $(tm.map)")
    println(io, "  Names: $(tm.names)")
    println(io, "  Samples: $(tm.samples)")
    return nothing
end

# todo: adaptive map construction (from samples)


#* General methods that work for both "ways"

names(tm::AbstractTransportMap) = names(tm.names)

function evaluate(tm::AbstractTransportMap, Z::Matrix{<:Real})
    return TransportMaps.evaluate(tm.map, Z)
end

function inverse(tm::AbstractTransportMap, X::Matrix{<:Real})
    return TransportMaps.inverse(tm.map, X)
end

"""
    pdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})

PDF of the transport map in the physical space X.
"""
function pdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
    return TransportMaps.pullback(tm.map, x)
end

function logpdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
    return log.(pdf(tm, x))
end
