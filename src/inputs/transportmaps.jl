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
    transform_density::Union{Nothing,Vector{<:RandomVariable{<:UnivariateDistribution}}}
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

"""
    sample(tm::TransportMapFromSamples, n::Integer=1)

Generate samples in the physical space `X` using the transport map `tm`.
"""
function sample(tm::TransportMap, n::Integer=1)
    Z = randn(n, numberdimensions(tm.map))
    X = evaluate(tm, Z)

    df = _to_dataframe(X, tm.names)

    if !isnothing(tm.transform_density)
        to_physical_space!(tm.transform_density, df)
    end

    return df
end

function to_physical_space!(tm::TransportMap, Z::DataFrame)
    X = evaluate(tm, Matrix(Z[!, tm.names]))
    Z[!, tm.names] .= X
    if !isnothing(tm.transform_density)
        to_physical_space!(tm.transform_density, Z)
    end
    return nothing
end

function to_standard_normal_space!(tm::TransportMap, X::DataFrame)
    Z = inverse(tm, Matrix(X[!, tm.names]))
    X[!, tm.names] .= Z
    if !isnothing(tm.transform_density)
        to_physical_space!(tm.transform_density, X)
    end
    return nothing
end

"""
    pdf(tm::TransportMap, x::AbstractVector{<:Real})

Probability density function of the transport map in the physical space X.
"""
function pdf(tm::TransportMap, x::AbstractVector{<:Real})
    if !isnothing(tm.transform_density)
        # transform from physical space x to auxilary space ξ
        ξ = _to_standard_normal(tm.transform_density, x)

        if any(isinf.(ξ))
            # when input x is outside the bounds of the RandomVariable
            return 0
        else
            J = jacobian(tm.transform_density, x, ξ)
            return TransportMaps.pullback(tm.map, ξ) * J
        end
    else
        return TransportMaps.pullback(tm.map, x)
    end
end

function pdf(tm::TransportMap, X::AbstractMatrix{<:Real})
    return [pdf(tm, xᵢ) for xᵢ in eachrow(X)]
end

# Helper function to transform vector to standard normal
function _to_standard_normal(
    inputs::Vector{<:RandomVariable{<:UnivariateDistribution}}, x::AbstractVector{<:Real}
)
    return [quantile(Normal(), cdf(rv.dist, x[i])) for (i, rv) in enumerate(inputs)]
end

# Helper function to get Jacobian of densitry transformation from standard normal to physical
function jacobian(
    inputs::Vector{<:RandomVariable{<:UnivariateDistribution}},
    x::AbstractVector{<:Real},
    ξ::AbstractVector{<:Real},
)
    return abs(
        prod(pdf(Normal(), ξ[i]) / pdf(rv.dist, x[i]) for (i, rv) in enumerate(inputs))
    )
end

"""
    variancediagnostic(tm::TransportMap, Z::DataFrame)

Evaluate the variance-based diagnostic for assessing the quality of a transport map.
"""
function variancediagnostic(tm::TransportMap, Z::DataFrame)
    return variance_diagnostic(tm.map, tm.target, Matrix(Z[!, tm.names]))
end

"""
    mapfromdensity(transportmap::AbstractTriangularMap, target::MapTargetDensity, quadrature::AbstractQuadratureWeights, names::Vector{Symbol})

Optimize a transport map from the given target density by minimizing the KL-divergence.
The KL-divergence is evaluated at the given quadrature points.
"""
function mapfromdensity(
    transportmap::AbstractTriangularMap,
    target::MapTargetDensity,
    quadrature::AbstractQuadratureWeights,
    names::Vector{Symbol},
    transform_density::Union{Nothing,Vector{<:RandomVariable{<:UnivariateDistribution}}},
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options(),
)
    optimize!(transportmap, target, quadrature; optimizer=optimizer, options=options)

    return TransportMap(transportmap, target, quadrature, transform_density, names)
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
    println(io, "  Number of Samples: $(nrow(tm.samples))")
    return nothing
end

"""
    mapfromsamples(transportmap::AbstractTriangularMap, X::DataFrame)

Fit a transportmap from samples.
"""
function mapfromsamples(
    transportmap::AbstractTriangularMap,
    X::DataFrame,
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options(),
)
    target_samples = Matrix(X)

    # First, fit a linear map
    linear_map = LinearMap(target_samples)
    # Optimize transportmap
    optimize!(
        transportmap, target_samples, linear_map; optimizer=optimizer, options=options
    )
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

"""
    sample(tm::TransportMapFromSamples, n::Integer=1)

Generate samples in the physical space `X` using the transport map `tm`.
"""
function sample(tm::TransportMapFromSamples, n::Integer=1)
    Z = randn(n, numberdimensions(tm.map))
    X = inverse(tm, Z)
    return _to_dataframe(X, tm.names)
end

"""
    pdf(tm::TransportMapFromSamples, x::AbstractVecOrMat{<:Real})

Probability density function of the transport map in the physical space X.
"""
function pdf(tm::TransportMapFromSamples, x::AbstractVecOrMat{<:Real})
    return TransportMaps.pullback(tm.map, x)
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
    logpdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})

Log-Probability density function of the transport map in the physical space X.
"""
function logpdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
    return log.(pdf(tm, x))
end

# Helper functions
function _to_dataframe(X::AbstractMatrix{<:Real}, names::Vector{Symbol})
    return DataFrame(X, names)
end

function _to_dataframe(x::AbstractVector{<:Real}, names::Vector{Symbol})
    return DataFrame(permutedims(x), names)
end
