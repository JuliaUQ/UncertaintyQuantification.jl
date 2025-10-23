"""
    JointDistribution{D<:Union{Copula,MultivariateDistribution}, M<:Union{RandomVariable,Symbol}}(d, m)

Represents a joint probability distribution, either via a copula and a vector of marginal random variables,
or a multivariate distribution and a vector of variable names.

# Constructors

- JointDistribution(d::Copula, m::Vector{RandomVariable}):
    - Use a copula `d` to combine the marginal distributions in `m` into a joint distribution.
    - The copula's dimension must match the length of `m`.
    - `m` must be a vector of `RandomVariable`.

- JointDistribution(d::MultivariateDistribution, m::Vector{Symbol}):
    - Use a multivariate distribution `d` with named components specified by `m`.
    - The distribution's dimension (number of variables) must match the length of `m`.
    - `m` must be a vector of `Symbol`.

# Examples

```jldoctest
julia> JointDistribution(GaussianCopula([1.0 0.71; 0.71 1.0]), [RandomVariable(Normal(), :x), RandomVariable(Uniform(), :y)])
JointDistribution{Copula, RandomVariable}(GaussianCopula([1.0 0.71; 0.71 1.0]), RandomVariable[RandomVariable{Normal{Float64}}(Normal{Float64}(μ=0.0, σ=1.0), :x), RandomVariable{Uniform{Float64}}(Uniform{Float64}(a=0.0, b=1.0), :y)])
```

```jldoctest
julia> JointDistribution(MultivariateNormal([1.0 0.71; 0.71 1.0]), [:x, :y])
JointDistribution{MultivariateDistribution, Symbol}(ZeroMeanFullNormal(
dim: 2
μ: Zeros(2)
Σ: [1.0 0.71; 0.71 1.0]
)
, [:x, :y])
```
"""
struct JointDistribution{
    D<:Union{Copula,MultivariateDistribution},M<:Union{RandomVariable,Symbol}
} <: RandomUQInput
    d::D
    m::Vector{<:M}

    # Copula + RandomVariable
    function JointDistribution(d::Copula, m::Vector{<:RandomVariable})
        dimensions(d) == length(m) ||
            throw(ArgumentError("Dimension mismatch between copula and marginals."))
        return new{Copula,RandomVariable}(d, m)
    end

    # MultivariateDistribution + Symbol
    function JointDistribution(d::MultivariateDistribution, m::Vector{Symbol})
        length(d) == length(m) ||
            throw(ArgumentError("Dimension mismatch between distribution and names."))
        return new{MultivariateDistribution,Symbol}(d, m)
    end
end

function sample(jd::JointDistribution{<:Copula,<:RandomVariable}, n::Integer=1)
    u = sample(jd.d, n)

    samples = DataFrame()

    for (i, rv) in enumerate(jd.m)
        samples[!, rv.name] = quantile.(rv.dist, u[:, i])
    end

    return samples
end

function sample(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, n::Integer=1)
    return DataFrame(permutedims(rand(jd.d, n)), jd.m)
end

function to_physical_space!(jd::JointDistribution{<:Copula,<:RandomVariable}, x::DataFrame)
    correlated_cdf = to_copula_space(jd.d, Matrix{Float64}(x[:, names(jd)]))
    for (i, rv) in enumerate(jd.m)
        x[!, rv.name] = quantile.(rv.dist, correlated_cdf[:, i])
    end
    return nothing
end

function to_standard_normal_space!(
    jd::JointDistribution{<:Copula,<:RandomVariable}, x::DataFrame
)
    for rv in jd.m
        if isa(rv.dist, ProbabilityBox)
            x[!, rv.name] = reverse_quantile.(rv.dist, x[:, rv.name])
        else
            x[!, rv.name] = cdf.(rv.dist, x[:, rv.name])
        end
    end
    uncorrelated_stdnorm = to_standard_normal_space(jd.d, Matrix{Float64}(x[:, names(jd)]))
    for (i, rv) in enumerate(jd.m)
        x[!, rv.name] = uncorrelated_stdnorm[:, i]
    end
    return nothing
end

function to_standard_normal_space!(jd::JointDistribution{D,M}, _::DataFrame) where {D,M}
    return error("Cannot map $(typeof(jd.d)) to standard normal space.")
end

function to_physical_space!(jd::JointDistribution{D,M}, _::DataFrame) where {D,M}
    return error("Cannot map $(typeof(jd.d)) to physical space.")
end

function names(jd::JointDistribution{<:Copula,<:RandomVariable})
    return vec(map(x -> x.name, jd.m))
end

function names(jd::JointDistribution{<:MultivariateDistribution,<:Symbol})
    return jd.m
end

mean(jd::JointDistribution{<:Copula,<:RandomVariable}) = mean.(jd.m)

function mean(jd::JointDistribution{<:MultivariateDistribution,<:Symbol})
    return mean(jd.d)
end

dimensions(jd::JointDistribution{<:Copula,<:RandomVariable}) = dimensions(jd.d)

dimensions(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = length(jd.d)

function bounds(
    jd::JointDistribution{
        <:Copula,<:RandomVariable{<:Union{UnivariateDistribution,ProbabilityBox}}
    },
)
    b = map(bounds, filter(isimprecise, jd.m))

    return vcat(getindex.(b, 1)...), vcat(getindex.(b, 2)...)
end

var(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = var(jd.d)

pdf(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, x::Union{Vector{<:Real}, <:Real}) = pdf(jd.d, x)

logpdf(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, x::Union{Vector{<:Real}, <:Real}) = logpdf(jd.d, x)

minimum(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = minimum(jd.d)

maximum(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = maximum(jd.d)

insupport(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, x::Union{Vector{<:Real}, <:Real}) = insupport(jd.d, x)
