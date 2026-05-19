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

function condition(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, existing_sample::DataFrameRow)
    dists, rv_names = ([rv.dist for rv in jd.m], [rv.name for rv in jd.m])

    if any(x -> x isa ProbabilityBox, dists)
        throw(ArgumentError("Conditional sampling using JointDistribution with P-box marginals is currently not supported"))
    end

    existing_cols = Symbol.(propertynames(existing_sample))
    jd_cols = intersect(existing_cols, rv_names)
    if isempty(jd_cols)
        throw(ArgumentError("DataFrameRow must contain at least one column from the variable names in the joint distribution"))
    end

    D = Copulas.SklarDist(jd.d, Tuple(dists))

    pairs = [
        (findfirst(==(name), rv_names), convert(Float64, existing_sample[name])) for name in jd_cols
    ]
    indices = first.(pairs)
    values = last.(pairs)

    uncond_indices = sort(setdiff(1:length(jd.m), indices))

    Dc = Copulas.condition(D, Tuple(indices), Tuple(values))

    copula_cond = nothing
    marginals_cond = nothing
    
    if hasproperty(Dc, :C)
        copula_cond = Dc.C
        if hasproperty(Dc, :m)
            marginals_cond = Dc.m
        end
    elseif hasproperty(Dc, :X)
        copula_cond = Dc.X
    elseif Distributions.isa(Dc, Union{Distributions.UnivariateDistribution, Distributions.ContinuousUnivariateDistribution})
        copula_cond = Dc
    else
        error("Cannot extract copula from conditioned distribution of type $(typeof(Dc)); unknown type")
    end

    # build conditional JointDistribution if there are unconditioned variables
    fixed_params = [Parameter(values[i], rv_names[indices[i]]) for i in eachindex(indices)]
    jd_cond = nothing
    
    if length(uncond_indices) > 1
        if marginals_cond !== nothing
            uncond_rvs = [RandomVariable(marginals_cond[j], jd.m[uncond_indices[j]].name) for j in eachindex(uncond_indices)]
            jd_cond = JointDistribution(copula_cond, uncond_rvs)
        else
            jd_cond = JointDistribution(copula_cond, jd.m[uncond_indices])
        end
    elseif length(uncond_indices) == 1  # ensuring that a single variable left still returns a JD 
        idx = uncond_indices[1]
        cond_rv = RandomVariable(copula_cond, jd.m[idx].name)
        copula_1d = Copulas.IndependentCopula(1)
        jd_cond = JointDistribution(copula_1d, [cond_rv])
    end

    inputs = UQInput[]
    if jd_cond !== nothing
        push!(inputs, jd_cond)
    end
    append!(inputs, fixed_params)

    return inputs
end

function sample(jd::JointDistribution{<:Copulas.Copula,<:RandomVariable}, existing_samples::DataFrame)
    dist, rv_names = ([rv.dist for rv in jd.m], [rv.name for rv in jd.m])

    if any(x -> x isa ProbabilityBox, dist)
        throw(ArgumentError("Conditional sampling using JointDistribution with P-box marginals is currently not supported"))
    end

    existing_cols = Symbol.(propertynames(existing_samples))
    jd_cols = intersect(Set(existing_cols), Set(rv_names))
    if isempty(jd_cols)
        throw(ArgumentError("DataFrame must contain at least one column from the joint distribution variables"))
    end

    # Columns that aren't in the JointDistribution
    other_cols = setdiff(existing_cols, rv_names)

    samples = vcat(map(i -> begin
        conditioned_inputs = condition(jd, existing_samples[i, existing_cols])

        sample_row = Dict{Symbol, Any}()

        for input in conditioned_inputs
            if input isa JointDistribution
                sampled = sample(input, 1)
                for name in names(input)
                    sample_row[name] = sampled[1, name]
                end
            elseif input isa Parameter
                sample_row[input.name] = input.value
            end
        end

        if length(sample_row) != length(jd.m)
            @warn "incomplete sample_row at row $i"
        end

        result_dict = Dict(rv.name => [sample_row[rv.name]] for rv in jd.m)
        
        for col in other_cols
            result_dict[col] = [existing_samples[i, col]]
        end

        DataFrame(result_dict)
    end, 1:nrow(existing_samples))...)
    
    return samples
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

function to_standard_normal_space!(jd::JointDistribution{D,M}, _::DataFrame) where {D,M}
    return error("Cannot map $(typeof(jd.d)) to standard normal space.")
end

function to_physical_space!(jd::JointDistribution{D,M}, _::DataFrame) where {D,M}
    return error("Cannot map $(typeof(jd.d)) to physical space.")
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