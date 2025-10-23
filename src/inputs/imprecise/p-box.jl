"""
    ProbabilityBox{T}(parameters::Dict{Symbol, Union{Real, Interval}}, lb::Real, ub::Real)

Represents a (optionally truncated) probability box (p-box) for a univariate distribution `T`, where parameters may be specified as precise values (`Real`) or intervals ([`Interval`](@ref)). The support of the distribution is bounded by `lb` (lower bound) and `ub` (upper bound).

To use the `ProbabilityBox` in an analysis it has to be wrapped in a [`RandomVariable`](@ref).

# Fields
- `parameters::Dict{Symbol, Union{Real, Interval}}`: Dictionary mapping parameter names (as symbols) to their values or intervals.
- `lb::Real`: Lower bound of the distribution's support.
- `ub::Real`: Upper bound of the distribution's support.

# Constructors
- `ProbabilityBox{T}(parameters::Dict{Symbol, Union{Real, Interval}}, lb::Real, ub::Real)`: Specify all parameters and support bounds explicitly.
- `ProbabilityBox{T}(parameters::Dict{Symbol, Union{Real, Interval}})`: Support bounds are inferred from the distribution type `T`.
- `ProbabilityBox{T}(parameter::Interval)`: For univariate distributions with a single parameter, construct from a single interval.

For convenience the first two constructors can also be called with a `Vector{Pair{Symbol,Union{Real,Interval}}}` to automatically create the `Dict`.

# Examples

```jldoctest
julia> ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => Interval(0.1, 1)), 0.0, Inf)
ProbabilityBox{Normal}(Dict{Symbol, Union{Real, Interval}}(:μ => [0, 1], :σ => [0.1, 1]), 0.0, Inf)

julia> ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => Interval(0.1, 1)))
ProbabilityBox{Normal}(Dict{Symbol, Union{Real, Interval}}(:μ => [0, 1], :σ => [0.1, 1]), -Inf, Inf)

julia> ProbabilityBox{Exponential}(Interval(0.1, 0.5))
ProbabilityBox{Exponential}(Dict{Symbol, Union{Real, Interval}}(:θ => [0.1, 0.5]), 0.0, Inf)

julia> ProbabilityBox{Normal}([:μ => Interval(0, 1), :σ => Interval(0.1, 1)])
ProbabilityBox{Normal}(Dict{Symbol, Union{Real, Interval}}(:μ => [0, 1], :σ => [0.1, 1]), -Inf, Inf)
```
"""
struct ProbabilityBox{T<:UnivariateDistribution}
    parameters::Dict{Symbol,Union{Real,Interval}}
    lb::Real
    ub::Real

    function ProbabilityBox{T}(
        p::Dict{Symbol,<:Any}, lb::Real, ub::Real
    ) where {T<:UnivariateDistribution}
        # Make sure all required parameters for the distribution are present
        if !issetequal(keys(p), fieldnames(T))
            error(
                "Parameter mismatch for ProbabilityBox $(keys(p)) != $([fieldnames(T)...])."
            )
        end
        # If someone only passes Parameters, return a RandomVariable instead.
        if all(isa.(values(p), Real))
            @warn "ProbabilityBox() returns a UnivariateDistribution if no intervals are passed"
            return T(getindex.(Ref(p), fieldnames(T))...)
        end

        # attempt construction of all distributions
        parameters = collect(getindex.(Ref(p), fieldnames(T)))
        parameter_bounds = map(x -> isa(x, Interval) ? bounds(x) : x, parameters)
        for pars in Iterators.product(parameter_bounds...)
            try
                T(pars...)
            catch e
                rethrow(
                    ArgumentError(
                        "Invalid $T distribution for parameter combination $(pars)"
                    ),
                )
            end
        end

        return new(convert(Dict{Symbol,Union{Real,Interval}}, p), lb, ub)
    end
end

function ProbabilityBox{T}(p::Dict{Symbol,<:Any}) where {T<:UnivariateDistribution}
    domain = support(T())
    return ProbabilityBox{T}(p, domain.lb, domain.ub)
end

function ProbabilityBox{T}(p::Dict{Symbol,<:Any}) where {T<:Uniform}
    if !issetequal(keys(p), fieldnames(T))
        error("Parameter mismatch for ProbabilityBox $(keys(p)) != $([fieldnames(T)...]).")
    end
    # p-boxes with Uniform distribution as parameter must be treated separately since their support changes with p-box lower and upper bounds.
    parameters = collect(getindex.(Ref(p), fieldnames(T)))
    values = vcat(
        [
            isa(p, Interval) ? collect(UncertaintyQuantification.bounds(p)) : p for
            p in parameters
        ]...,
    )
    return ProbabilityBox{T}(p, minimum(values), maximum(values))
end

function ProbabilityBox{T}(parameter::Interval) where {T<:UnivariateDistribution}
    @assert length(fieldnames(T)) == 1
    return ProbabilityBox{T}(
        Dict{Symbol,Union{Real,Interval}}(fieldnames(T)[1] => parameter)
    )
end

function ProbabilityBox{T}(
    parameters::Vector{<:Pair{Symbol,<:Any}}
) where {T<:UnivariateDistribution}
    return ProbabilityBox{T}(Dict(parameters))
end

function ProbabilityBox{T}(
    parameters::Vector{<:Pair{Symbol,<:Any}}, lb::Real, ub::Real
) where {T<:UnivariateDistribution}
    return ProbabilityBox{T}(Dict(parameters), lb, ub)
end

function map_to_precise(
    x::AbstractVector{<:Real}, pbox::ProbabilityBox{T}
) where {T<:UnivariateDistribution}
    parameters = collect(getindex.(Ref(pbox.parameters), fieldnames(T)))
    intervals = filter(x -> isa(x, Interval), parameters)
    if !all(in.(x, intervals))
        error("Values outside of parameter intervals for ProbabilityBox")
    end

    _x = copy(x)

    p = [
        if isa(par, Interval)
            popfirst!(_x)
        else
            par
        end for par in parameters
    ]

    dist_support = support(T())

    if pbox.lb == dist_support.lb && pbox.ub == dist_support.ub
        return T(p...)
    end

    return truncated(T(p...), pbox.lb, pbox.ub)
end

function quantile(pbox::ProbabilityBox{T}, u::Real) where {T<:UnivariateDistribution}
    quantiles = map(
        par -> quantile(map_to_precise([par...], pbox), u),
        Iterators.product([[a, b] for (a, b) in zip(bounds(pbox)...)]...),
    )

    return Interval(minimum(quantiles), maximum(quantiles))
end

rand(pbox::ProbabilityBox, n::Integer=1) = quantile.(Ref(pbox), rand(n))

function bounds(pbox::ProbabilityBox{T}) where {T<:UnivariateDistribution}
    intervals = filter(
        x -> isa(x, Interval), collect(getindex.(Ref(pbox.parameters), fieldnames(T)))
    )
    lb = getproperty.(intervals, :lb)
    ub = getproperty.(intervals, :ub)

    return lb, ub
end

function cdf(pbox::ProbabilityBox{T}, x::Real) where {T<:UnivariateDistribution}
    cdfs = map(
        par -> cdf(map_to_precise([par...], pbox), x),
        Iterators.product([[a, b] for (a, b) in zip(bounds(pbox)...)]...),
    )

    return Interval(minimum(cdfs), maximum(cdfs))
end

# Does the inverse of quantile, not cdf, which would return an interval
function reverse_quantile(
    pbox::ProbabilityBox{T}, x::Interval
) where {T<:UnivariateDistribution}
    lb, ub = bounds(pbox)

    cdfs_lo = map(
        par -> cdf(map_to_precise([par...], pbox), x.lb),
        Iterators.product([[a, b] for (a, b) in zip(lb, ub)]...),
    )

    cdfs_hi = map(
        par -> cdf(map_to_precise([par...], pbox), x.ub),
        Iterators.product([[a, b] for (a, b) in zip(lb, ub)]...),
    )

    u_lo = maximum(cdfs_lo)
    u_hi = minimum(cdfs_hi)

    error = abs(u_hi - u_lo)
    if error > 10^-6
        @warn(
            "When inverting the quantile function for a p-box, the error was $(error), greater than the allowed tolerance of 10^-6"
        )
    end
    return middle(u_lo, u_hi)   # Return midpoint
end

Base.broadcastable(pbox::ProbabilityBox) = Ref(pbox)

length(::ProbabilityBox{<:UnivariateDistribution}) = 1

function mean(pbox::ProbabilityBox{<:UnivariateDistribution})
    cdfs = map(
        par -> mean(map_to_precise([par...], pbox)),
        Iterators.product([[a, b] for (a, b) in zip(bounds(pbox)...)]...),
    )

    return Interval(minimum(cdfs), maximum(cdfs))
end

function var(pbox::ProbabilityBox{<:UnivariateDistribution})
    cdfs = map(
        par -> var(map_to_precise([par...], pbox)),
        Iterators.product([[a, b] for (a, b) in zip(bounds(pbox)...)]...),
    )

    return Interval(minimum(cdfs), maximum(cdfs))
end
