"""
GaussianLikelihood(data::DataFrame, σ::Float64) -> Function
GaussianLikelihood(data::DataFrame, σ::Vector{Float64}) -> Function
GaussianLikelihood(data::DataFrame, Σ::Matrix{Float64}) -> Function

Convenience functions to create a Gaussian likelihood function for Bayesian updating. The likelihood function accepts a DataFrame of parameter samples and returns the log-likelihood for each sample based on the provided observed data.
- `data`: A DataFrame containing the observed data. Each column corresponds to an observed variable
- `σ`: A scalar or vector of standard deviations for the Gaussian likelihood. If a scalar is provided, it is used for all observed variables. If a vector is provided, its length must match the number of observed variables.
- `Σ`: A covariance matrix for the multivariate Gaussian likelihood. Its dimensions must match the number of observed variables.
"""
function GaussianLikelihood(data::DataFrame, σ::Float64)
    obs_names = names(data)
    return function likelihood(df::DataFrame)
        all(contains.(obs_names, names(df))) || throw(ArgumentError("DataFrame must contain columns $(obs_names)"))
        lkl = zeros(size(df, 1))
        for _n in obs_names
            lkl .+= sum([logpdf.(Normal.(df[!, _n], σ), d[_n]) for d in eachrow(data)])
        end
        return lkl
    end
end

function GaussianLikelihood(data::DataFrame, σ::Vector{Float64})
    obs_names = names(data)
    @assert length(σ) == length(obs_names) "Length of σ must match number of observed variables"

    return function likelihood(df::DataFrame)
        all(contains.(obs_names, names(df))) || throw(ArgumentError("DataFrame must contain columns $(obs_names)"))
        lkl = zeros(size(df, 1))
        for (_n, _σ) in zip(obs_names, σ)
            lkl .+= sum([logpdf.(Normal.(df[!, _n], _σ), d[_n]) for d in eachrow(data)])
        end
        return lkl
    end
end

function GaussianLikelihood(data::DataFrame, Σ::Matrix{Float64})
    obs_names = names(data)
    @assert size(Σ, 1) == length(obs_names) && size(Σ, 2) == length(obs_names) "Size of Σ must match number of observed variables"

    return function likelihood(df::DataFrame)
        all(contains.(obs_names, names(df))) || throw(ArgumentError("DataFrame must contain columns $(obs_names)"))
        lkl = logpdf.([MvNormal(Vector(s_), Σ) for s_ in eachrow(df)], Vector.(eachrow(data)))
        return lkl
    end
end
