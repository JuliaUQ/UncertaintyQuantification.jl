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
JointDistribution{GaussianCopula{2, Matrix{Float64}}, RandomVariable}(GaussianCopula{2, Matrix{Float64}}(Σ = [1.0 0.71; 0.71 1.0])), RandomVariable{<:UnivariateDistribution}[RandomVariable{Normal{Float64}}(Normal{Float64}(μ=0.0, σ=1.0), :x), RandomVariable{Uniform{Float64}}(Uniform{Float64}(a=0.0, b=1.0), :y)])
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
    D<:Union{<:Copulas.Copula,MultivariateDistribution},M<:Union{RandomVariable,Symbol}
} <: RandomUQInput
    d::D
    m::Vector{<:M}

    # Copula + RandomVariable
    function JointDistribution(c::Copulas.Copula, m::Vector{<:RandomVariable})
        if length(c) != length(m)
            throw(ArgumentError("Dimension mismatch between copula and marginals."))
        end
        if unique(names(m)) != names(m)
            throw(ArgumentError("Marginal names must be unique."))
        end
        return new{typeof(c),RandomVariable}(c, m)
    end

    # MultivariateDistribution + Symbol
    function JointDistribution(d::MultivariateDistribution, m::Vector{Symbol})
        if length(d) != length(m)
            throw(ArgumentError("Dimension mismatch between distribution and names."))
        end
        if unique(m) != m
            throw(ArgumentError("Marginal names must be unique."))
        end
        return new{MultivariateDistribution,Symbol}(d, m)
    end
end

function sample(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, n::Integer=1)
    u = rand(jd.d, n)
    # ensure that u is a Matrix
    if n == 1
        u = reshape(u, (length(jd.d), 1))
    end

    samples = DataFrame()

    for (i, rv) in enumerate(jd.m)
        samples[!, rv.name] = quantile.(rv.dist, u[i, :])
    end

    return samples
end

function sample(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, n::Integer=1)
    return DataFrame(permutedims(rand(jd.d, n)), jd.m)
end

function sample(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, conditions::AbstractDict{<:Symbol,<:Real}, n::Integer=1)
    dist, rv_names = ([rv.dist for rv in jd.m], [rv.name for rv in jd.m])
    D = Copulas.SklarDist(jd.d, Tuple(dist))

    pairs = [(findfirst(==(name), rv_names), value) for (name, value) in conditions if findfirst(==(name), rv_names) !== nothing]
    indices = first.(pairs)
    values = last.(pairs)

    # map unconditioned indices (in original order) to rows of cond_samples
    uncond_indices = sort(setdiff(1:length(jd.m), indices))

    Dc = Copulas.condition(D, Tuple(indices), Tuple(values))
    cond_samples = rand(Dc, n)

    if length(uncond_indices) == 1 # ensure Matrix shape for vector output
        cond_samples = reshape(cond_samples, 1, size(cond_samples, 1))
    end

    samples = DataFrame()
    for (i, rv) in enumerate(jd.m)
        if haskey(conditions, rv.name)
            samples[!, rv.name] = fill(conditions[rv.name], n)
        else
            # find position of unconditioned index in reduced conditioned sample rows
            pos = findfirst(==(i), uncond_indices)
            samples[!, rv.name] = cond_samples[pos, :]
        end
    end

    return samples
end

function sample(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, existing_samples::DataFrame)
    dist, rv_names = ([rv.dist for rv in jd.m], [rv.name for rv in jd.m])
    D = Copulas.SklarDist(jd.d, Tuple(dist))

    if size(existing_samples, 2) >= length(rv_names) || size(existing_samples, 2) == 0
        throw(ArgumentError("DataFrame has more columns than the number of variables in the joint distribution or is empty"))
    end
    existing_cols = Symbol.(DataFrames.names(existing_samples))
    if !issubset(Set(existing_cols), Set(rv_names))
        throw(ArgumentError("DataFrame columns must be a subset of the variable names in the joint distribution"))
    end

    indices = map(name -> findfirst(==(name), rv_names), existing_cols)
    cond_indices = [findfirst(==(name), rv_names) for name in existing_cols]
    uncond_indices = sort(setdiff(1:length(jd.m), cond_indices))

    rows = map(1:nrow(existing_samples)) do i
        cond_values = Tuple(convert(Float64, existing_samples[i, c]) for c in existing_cols)

        Dc = Copulas.condition(D, Tuple(cond_indices), cond_values)
        cs = rand(Dc, 1)
        cs_mat = isa(cs, AbstractMatrix) ? cs : reshape(cs, length(cs), 1)

        # assemble row as Dict{Symbol,Any} preserving jd.m order via keys
        Dict(
            (rv.name => (
                rv.name in existing_cols ? existing_samples[i, rv.name] :
                begin
                    pos = findfirst(==(j), uncond_indices)
                    cs_mat[pos, 1]
                end
            )) for (j, rv) in enumerate(jd.m)
        )
    end

    samples = DataFrame(rows)
    return samples[:, Symbol.(rv_names)]
end

function to_physical_space!(
    jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, x::DataFrame
)
    # correlated cdf space
    U = inverse_rosenblatt(
        jd.d, permutedims(cdf.(Normal(), Matrix{Float64}(x[:, names(jd)])))
    )
    # inverse transform for marginals
    for (i, rv) in enumerate(jd.m)
        x[!, rv.name] = quantile.(rv.dist, U[i, :])
    end
    return nothing
end

function to_standard_normal_space!(
    jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, x::DataFrame
)
    for rv in jd.m
        if isa(rv.dist, ProbabilityBox)
            x[!, rv.name] = reverse_quantile.(rv.dist, x[:, rv.name])
        else
            x[!, rv.name] = cdf.(rv.dist, x[:, rv.name])
        end
    end
    U = quantile.(Normal(), rosenblatt(jd.d, permutedims(Matrix{Float64}(x[:, names(jd)]))))
    for (i, rv) in enumerate(jd.m)
        x[!, rv.name] = U[i, :]
    end
    return nothing
end

function to_standard_normal_space!(jd::JointDistribution{D,M}, df::DataFrame) where {D,M}
    if isa(jd.d, AbstractTransportMap)
        return to_standard_normal_space!(jd.d, df)
    else
        return error("Cannot map $(typeof(jd.d)) to standard normal space.")
    end
end

function to_physical_space!(jd::JointDistribution{D,M}, df::DataFrame) where {D,M}
    if isa(jd.d, AbstractTransportMap)
        return to_physical_space!(jd.d, df)
    else
        return error("Cannot map $(typeof(jd.d)) to physical space.")
    end
end

function names(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable})
    return vec(map(x -> x.name, jd.m))
end

function names(jd::JointDistribution{<:MultivariateDistribution,<:Symbol})
    return jd.m
end

mean(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}) = mean.(jd.m)

function mean(jd::JointDistribution{<:MultivariateDistribution,<:Symbol})
    return mean(jd.d)
end

dimensions(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}) = length(jd.d)

dimensions(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = length(jd.d)

function bounds(
    jd::JointDistribution{
        <:Copulas.Copula,<:RandomVariable{<:Union{UnivariateDistribution,ProbabilityBox}}
    },
)
    b = map(bounds, filter(isimprecise, jd.m))

    return vcat(getindex.(b, 1)...), vcat(getindex.(b, 2)...)
end

var(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = var(jd.d)

function pdf(
    jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, x::AbstractVector{<:Real}
)
    return pdf(jd.d, [cdf(jd.m[i], x[i]) for i in 1:length(jd.d)]) * prod([pdf(jd.m[i], x[i]) for i in 1:length(jd.d)])
end

function cdf(
    jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, x::AbstractVector{<:Real}
)
    return cdf(jd.d, [cdf(jd.m[i], x[i]) for i in 1:length(jd.d)])
end

function pdf(
    jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, x::AbstractVector{<:Real}
)
    return pdf(jd.d, x)
end

function logpdf(
    jd::JointDistribution{<:MultivariateDistribution,<:Symbol}, x::AbstractVector{<:Real}
)
    return logpdf(jd.d, x)
end

minimum(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = minimum(jd.d)

maximum(jd::JointDistribution{<:MultivariateDistribution,<:Symbol}) = maximum(jd.d)

function insupport(
    jd::JointDistribution{<:MultivariateDistribution,<:Symbol},
    x::Union{Vector{<:Real},<:Real},
)
    return insupport(jd.d, x)
end