"""
    TransportMap(map, target, transform_density, names)

A transport map used to transform between standard normal space Z and physical space X. The `map` is the optimized triangular transport map, `target` is the target density, and `transform_density` optionally specifies random variables for density transformation, and `names` is a vector of variable names.

"""
struct TransportMap <: AbstractTransportMap
    map::AbstractTriangularMap
    target::MapTargetDensity
    transform_density::Union{Nothing,Vector{<:RandomVariable{<:UnivariateDistribution}}}
    names::Vector{Symbol}
end

function Base.show(io::IO, tm::TransportMap)
    print(io, "TransportMap(")
    print(io, "map=$(tm.map), ")
    print(io, "target=$(tm.target), ")
    print(io, "names=$(tm.names)")
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", tm::TransportMap)
    println(io, "Transport Map:")
    println(io, "  Map: $(tm.map)")
    println(io, "  Target: $(tm.target)")
    println(io, "  Names: $(tm.names)")
    return nothing
end

"""
    to_physical_space!(tm::TransportMap, Z::DataFrame)

Transforms samples in `Z` from standard normal space to physical space using transport map `tm`.
"""
function to_physical_space!(tm::TransportMap, Z::DataFrame)
    X = evaluate(tm, Matrix(Z[!, tm.names]))
    Z[!, tm.names] .= X
    if !isnothing(tm.transform_density)
        to_physical_space!(tm.transform_density, Z)
    end
    return nothing
end

"""
    to_standard_normal_space!(tm::TransportMap, X::DataFrame)

Transforms samples in `X` from physical space to standard normal space using transport map `tm`.
"""
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

Evaluate the probability density function (pdf) of the transport map in the physical space, i.e., the pushforward density. `x` is a vector, representing a point in the M-dimensional target space. Returns a `Float64`.
"""
function pdf(tm::TransportMap, x::AbstractVector{<:Real})
    if !isnothing(tm.transform_density)
        # transform from physical space x to auxilary space ξ
        ξ = _to_standard_normal(tm.transform_density, x)

        if any(isinf.(ξ))
            # when input x is outside the bounds of the RandomVariable
            return 0
        else
            J = _jacobian(tm.transform_density, x, ξ)
            return TransportMaps.pullback(tm.map, ξ) * J
        end
    else
        return TransportMaps.pullback(tm.map, x)
    end
end

"""
    pdf(tm::TransportMap, x::AbstractMatrix{<:Real})

Evaluate the probability density function (pdf) of the transport map in the physical space, i.e., the pushforward density. `X` is a matrix of points in the physical space for which the pdf is evaluated. Returns a `Vector{Float64}` of evaluated pdf values for each row of the matrix.
"""
function pdf(tm::TransportMap, X::AbstractMatrix{<:Real})
    return [pdf(tm, xᵢ) for xᵢ in eachrow(X)]
end

"""
    mean(tm::TransportMap)

Get the mean value of the density approximated by the transport map.
"""
function mean(tm::TransportMap)
    if !isnothing(tm.transform_density)
        ξ = evaluate(tm, zeros(length(tm)))
        return _to_physical(tm.transform_density, ξ)
    else
        return evaluate(tm, zeros(length(tm)))
    end
end

# Helper function to transform vector to standard normal
function _to_standard_normal(
    inputs::Vector{<:RandomVariable{<:UnivariateDistribution}}, x::AbstractVector{<:Real}
)
    return [quantile(Normal(), cdf(rv.dist, x[i])) for (i, rv) in enumerate(inputs)]
end

# Helper function to transform vector to physical
function _to_physical(
    inputs::Vector{<:RandomVariable{<:UnivariateDistribution}}, ξ::AbstractVector{<:Real}
)
    return [quantile(rv.dist, cdf(Normal(), ξ[i])) for (i, rv) in enumerate(inputs)]
end

# Helper function to get Jacobian of densitry transformation from standard normal to physical
function _jacobian(
    inputs::Vector{<:RandomVariable{<:UnivariateDistribution}},
    x::AbstractVector{<:Real},
    ξ::AbstractVector{<:Real},
)
    return abs(
        prod(pdf(rv.dist, x[i]) / pdf(Normal(), ξ[i]) for (i, rv) in enumerate(inputs))
    )
end

"""
    variancediagnostic(tm::TransportMap, Z::DataFrame)

Evaluate the variance-based diagnostic for assessing the quality of a transport map. The diagnostic measures the variance of the log-ratio between the pushforward density and the reference density. A smaller variance indicates a better approximation of the target density by the transport map. The argument `Z` is a DataFrame of samples in the standard normal space. Returns the evaluated variance diagnostic.
"""
function variancediagnostic(tm::TransportMap, Z::DataFrame)
    return variance_diagnostic(tm.map, tm.target, Matrix(Z[!, tm.names]))
end

"""
    mapfromdensity(transportmap, target, quadrature, names, transform_density, optimizer, options)

Optimize a transport map from the given target density by minimizing the KL-divergence. The `transportmap` is the transport map to be optimized, `target` is the target density in the physical space, and `quadrature` specifies quadrature points in the standard normal space. The KL-divergence is evaluated at the given quadrature points. The `names` is a vector of variable names, and `transform_density` optionally specifies random variables for transformation. The `optimizer` specifies the optimization method from Optim.jl (default: `LBFGS()`), and `options` allows passing options to the optimizer (default: `Optim.Options()`). Returns the optimized [`TransportMap`](@ref).

Alternative calls

```julia
    mapfromdensity(transportmap, target, quadrature, names, transform_density)  # optimizer = LBFGS(), options = Optim.Options()
```
"""
function mapfromdensity(
    transportmap::AbstractTriangularMap,
    target::MapTargetDensity,
    quadrature::AbstractQuadratureWeights,
    names::Vector{Symbol},
    transform_density::Union{Nothing,Vector{<:RandomVariable{<:UnivariateDistribution}}}=nothing,
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options(),
)
    optimize!(transportmap, target, quadrature; optimizer=optimizer, options=options)

    return TransportMap(transportmap, target, transform_density, names)
end

"""
    TransportMapFromSamples(map, samples, names)

A transport map constructed from samples in the physical space X which are mapped to the standard normal space Z. The `map` is the optimized composed map, `samples` is a DataFrame of samples in the physical space X used to fit the map, and `names` is a vector of variable names.
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
    print(io, "  Number of Samples: $(nrow(tm.samples))")
    return nothing
end

"""
    mapfromsamples(transportmap, X, optimizer, options)

Fit a transport map from samples. The `transportmap` is the transport map to be optimized, and `X` is a DataFrame with samples in the physical space. The `optimizer` specifies the optimization method from Optim.jl (default: `LBFGS()`), and `options` allows passing options to the optimizer (default: `Optim.Options()`). Returns the optimized [`TransportMapFromSamples`](@ref).

Alternative calls

```julia
    mapfromsamples(transportmap, X)  # optimizer = LBFGS(), options = Optim.Options()
```
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

"""
    to_physical_space!(tm::TransportMapFromSamples, Z::DataFrame)

Transforms samples in `Z` from standard normal space to physical space using transport map `tm`.
"""
function to_physical_space!(tm::TransportMapFromSamples, Z::DataFrame)
    X = inverse(tm, Matrix(Z[!, tm.names]))
    Z[!, tm.names] .= X
    return nothing
end

"""
    to_standard_normal_space!(tm::TransportMapFromSamples, X::DataFrame)

Transforms samples in `X` from physical space to standard normal space using transport map `tm`.
"""
function to_standard_normal_space!(tm::TransportMapFromSamples, X::DataFrame)
    Z = evaluate(tm, Matrix(X[!, tm.names]))
    X[!, tm.names] .= Z
    return nothing
end

"""
    sample(tm::AbstractTransportMap, n::Integer=1)

Generate `n` samples in the physical space `X` using the transport map `tm`.
"""
function sample(tm::AbstractTransportMap, n::Integer=1)
    X = permutedims(rand(tm, n))
    return _to_dataframe(X, tm.names)
end

"""
    pdf(tm::TransportMapFromSamples, x)

Evaluate the probability density function (pdf) of the transport map in the physical space, i.e., the pullback density. The argument `x` can be either a vector or matrix of values where the pdf is evaluated. Returns a `Float64` for a vector input or `Vector{Float64}` for a matrix input.
"""
function pdf(tm::TransportMapFromSamples, x::AbstractVecOrMat{<:Real})
    return TransportMaps.pullback(tm.map, x)
end

"""
    mean(tm::TransportMapFromSamples)

Get the mean value of the density approximated by the transport map.
"""
function mean(tm::TransportMapFromSamples)
    return inverse(tm, zeros(length(tm)))
end

# General methods that work for both "ways"
names(tm::AbstractTransportMap) = names(tm.names)

function evaluate(tm::AbstractTransportMap, Z::AbstractVecOrMat{<:Real})
    return TransportMaps.evaluate(tm.map, Z)
end

function inverse(tm::AbstractTransportMap, X::AbstractVecOrMat{<:Real})
    return TransportMaps.inverse(tm.map, X)
end

# Methods for compatibility with `MultivariateDistribution`
length(tm::AbstractTransportMap) = length(tm.names)
eltype(tm::AbstractTransportMap) = Float64
sampler(tm::AbstractTransportMap) = tm

function Distributions._rand!(rng::AbstractRNG, tm::TransportMap, x::AbstractVector{<:Real})
    if !isnothing(tm.transform_density)
        randn!(rng, x)
        return _to_physical(tm.transform_density, evaluate(tm, x))
    else
        randn!(rng, x)
        return evaluate(tm, x)
    end
end

function Distributions._rand!(rng::AbstractRNG, tm::TransportMap, x::AbstractMatrix{<:Real})
    if !isnothing(tm.transform_density)
        randn!(rng, x)
        ξ = permutedims(evaluate(tm, permutedims(x)))
        for (i, ξi) in enumerate(eachcol(ξ))
            x[:, i] .= _to_physical(tm.transform_density, ξi)
        end
        return x
    else
        randn!(rng, x)
        return permutedims(evaluate(tm, permutedims(x)))
    end
end

function Distributions._rand!(
    rng::AbstractRNG, tm::TransportMapFromSamples, x::AbstractVector{<:Real}
)
    randn!(rng, x)
    return inverse(tm, x)
end

function Distributions._rand!(
    rng::AbstractRNG, tm::TransportMapFromSamples, x::AbstractMatrix{<:Real}
)
    randn!(rng, x)
    return permutedims(inverse(tm, permutedims(x)))
end

function Distributions._logpdf(tm::AbstractTransportMap, x::AbstractArray)
    return log.(pdf(tm, x))
end

"""
    logpdf(tm::AbstractTransportMap, x)

Log-probability density function of the transport map in the physical space X. The argument `x` can be either a vector or matrix of values where the log-pdf is evaluated. Returns a `Float64` for a vector input or `Vector{Float64}` for a matrix input.
"""
function logpdf(tm::AbstractTransportMap, x::AbstractVecOrMat{<:Real})
    return Distributions._logpdf(tm, x)
end

# Helper functions
function _to_dataframe(X::AbstractMatrix{<:Real}, names::Vector{Symbol})
    return DataFrame(X, names)
end

function _to_dataframe(x::AbstractVector{<:Real}, names::Vector{Symbol})
    return DataFrame(permutedims(x), names)
end

function minimum(tm::TransportMap)
    if !isnothing(tm.transform_density)
        return [minimum(dens) for dens in tm.transform_density]
    else
        # Without transformation: support domain is ℝ
        return fill(-Inf, length(tm))
    end
end

function maximum(tm::TransportMap)
    # support domain is ℝ
    return fill(Inf, length(tm))
end

function minimum(tm::TransportMapFromSamples)
    if !isnothing(tm.transform_density)
        return [minimum(dens) for dens in tm.transform_density]
    else
        # Without transformation: support domain is ℝ
        return fill(-Inf, length(tm))
    end
end

function maximum(tm::TransportMapFromSamples)
    # support domain is ℝ
    return fill(Inf, length(tm))
end

function insupport(tm::AbstractTransportMap, x::Vector{<:Real})
    if all(x .>= minimum(tm)) && all(x .<= maximum(tm))
        return true
    else
        return false
    end
end

function var(tm::AbstractTransportMap)
    # maybe can be calculated using numerical integration ?
    return error("Variance not defined for $(typeof(tm)).")
end


# JointDistribution overlay for transport maps

#! make this work for transport map!
JointDistribution(tm::AbstractTransportMap) = JointDistribution(tm, tm.names)

function to_standard_normal_space!(
    jd::JointDistribution{<:AbstractTransportMap, Symbol}, df::DataFrame
)
    return to_standard_normal_space!(jd.d, df)
end

function to_physical_space!(
    jd::JointDistribution{<:AbstractTransportMap, Symbol}, df::DataFrame
)
    return to_physical_space!(jd.d, df)
end
