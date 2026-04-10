"""
	RandomVariable(dist::UnivariateDistribution, name::Symbol)

Defines a random variable, with a univariate distribution from Distributions.jl and a name.

# Examples

```jldoctest
julia> RandomVariable(Normal(), :x)
RandomVariable{Normal{Float64}}(Normal{Float64}(μ=0.0, σ=1.0), :x)

julia> RandomVariable(Exponential(1), :x)
RandomVariable{Exponential{Float64}}(Exponential{Float64}(θ=1.0), :x)
```
"""
struct RandomVariable{T<:Union{UnivariateDistribution,ProbabilityBox}} <: RandomUQInput
    dist::T
    name::Symbol
end

# promote precise rvs to common supertype
function Base.promote_rule(
    ::Type{RandomVariable{T1}}, ::Type{RandomVariable{T2}}
) where {T1<:UnivariateDistribution,T2<:UnivariateDistribution}
    if T1 == T2
        return RandomVariable{T1}
    else
        return RandomVariable{<:UnivariateDistribution}
    end
end

"""
	sample(rv::RandomVariable, n::Integer=1)

Generates n samples from a random variable. Returns a DataFrame.

# Examples

See also: [`RandomVariable`](@ref)
"""
function sample(rv::RandomVariable, n::Integer=1)
    return DataFrame(rv.name => rand(rv.dist, n))
end

dimensions(rv::RandomVariable) = 1

logpdf(rv::RandomVariable, x::Real) = logpdf(rv.dist, x)
pdf(rv::RandomVariable, x::Real) = pdf(rv.dist, x)
cdf(rv::RandomVariable, x::Real) = cdf(rv.dist, x)
quantile(rv::RandomVariable, q::Real) = quantile(rv.dist, q)
minimum(rv::RandomVariable) = minimum(rv.dist)
maximum(rv::RandomVariable) = maximum(rv.dist)
insupport(rv::RandomVariable, x::Real) = insupport(rv.dist, x)
mean(rv::RandomVariable) = mean(rv.dist)
var(rv::RandomVariable) = var(rv.dist)

include("parametric/distribution.jl")
include("parametric/pbox.jl")
