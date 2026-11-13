"""
    gaussian_likelihood(data::DataFrame, σ::Float64) -> Function

Create a Gaussian likelihood function for Bayesian parameter updating using scalar standard deviation.

This returns a likelihood function that computes log-likelihood for parameter samples based on 
observed data, assuming independent Gaussian distributions with a common standard deviation across 
all observed variables.

# Arguments
- `data::DataFrame`: Observed data where each column represents an observed variable.
- `σ::Float64`: Common standard deviation for the Gaussian likelihood across all variables.

# Returns
A function `likelihood(df::DataFrame)` that:
- Accepts a DataFrame of parameter samples (columns must match `data` column names)
- Returns a vector of log-likelihoods (one per row)

# Example
```julia
observed = DataFrame(x = [1.0, 2.0], y = [3.0, 4.0])
lik = gaussian_likelihood(observed, 0.5)
samples = DataFrame(x = [0.9, 2.1], y = [3.1, 3.9])
loglik = lik(samples)  # Returns log-likelihoods for each sample
```
"""
function gaussian_likelihood(data::DataFrame, σ::Float64)
    obs_names = names(data)
    return function likelihood(df::DataFrame)
        @assert all(x -> x in names(df), obs_names) "DataFrame must contain columns $(obs_names)"
        lkl = zeros(size(df, 1))
        for _n in obs_names
            lkl .+= sum([logpdf.(Normal.(df[!, _n], σ), d[_n]) for d in eachrow(data)])
        end
        return lkl
    end
end

"""
    gaussian_likelihood(data::DataFrame, σ::Vector{Float64}) -> Function

Create a Gaussian likelihood function for Bayesian parameter updating using variable standard deviations.

This returns a likelihood function that computes log-likelihood for parameter samples based on 
observed data, assuming independent Gaussian distributions with potentially different standard 
deviations for each observed variable.

# Arguments
- `data::DataFrame`: Observed data where each column represents an observed variable.
- `σ::Vector{Float64}`: Standard deviations for each observed variable. Length must match 
  the number of columns in `data`.

# Returns
A function `likelihood(df::DataFrame)` that:
- Accepts a DataFrame of parameter samples (columns must match `data` column names)
- Returns a vector of log-likelihoods (one per row)

# Throws
- `AssertionError`: If length of `σ` does not match the number of observed variables.

# Example
```julia
observed = DataFrame(x = [1.0, 2.0], y = [3.0, 4.0])
lik = gaussian_likelihood(observed, [0.5, 1.0])  # Different σ for x and y
samples = DataFrame(x = [0.9, 2.1], y = [3.1, 3.9])
loglik = lik(samples)  # Returns log-likelihoods for each sample
```
"""
function gaussian_likelihood(data::DataFrame, σ::Vector{Float64})
    obs_names = names(data)
    @assert length(σ) == length(obs_names) "Length of σ must match number of observed variables"

    return function likelihood(df::DataFrame)
        @assert all(x -> x in names(df), obs_names) "DataFrame must contain columns $(obs_names)"
        lkl = zeros(size(df, 1))
        for (_n, _σ) in zip(obs_names, σ)
            lkl .+= sum([logpdf.(Normal.(df[!, _n], _σ), d[_n]) for d in eachrow(data)])
        end
        return lkl
    end
end

"""
    gaussian_likelihood(data::DataFrame, Σ::Matrix{Float64}) -> Function

Create a Gaussian likelihood function for Bayesian parameter updating using a covariance matrix.

This returns a likelihood function that computes log-likelihood for parameter samples based on 
observed data, assuming a multivariate Gaussian distribution with the specified covariance matrix.
This allows modeling correlations between observed variables.

# Arguments
- `data::DataFrame`: Observed data where each column represents an observed variable.
- `Σ::Matrix{Float64}`: Covariance matrix. Must be square with dimensions matching the number 
  of observed variables, and must be positive definite.

# Returns
A function `likelihood(df::DataFrame)` that:
- Accepts a DataFrame of parameter samples (columns must match `data` column names)
- Returns a vector of log-likelihoods (one per row)

# Throws
- `AssertionError`: If `Σ` dimensions do not match the number of observed variables.

# Example
```julia
observed = DataFrame(x = [1.0, 2.0], y = [3.0, 4.0])
Σ = [0.5 0.1; 0.1 0.8]  # Covariance matrix with correlation
lik = gaussian_likelihood(observed, Σ)
samples = DataFrame(x = [0.9, 2.1], y = [3.1, 3.9])
loglik = lik(samples)  # Returns log-likelihoods for each sample
```
"""
function gaussian_likelihood(data::DataFrame, Σ::Matrix{Float64})
    obs_names = names(data)
    @assert size(Σ, 1) == length(obs_names) && size(Σ, 2) == length(obs_names) "Size of Σ must match number of observed variables"

    return function likelihood(df::DataFrame)
        @assert all(x -> x in names(df), obs_names) "DataFrame must contain columns $(obs_names)"
        all(x -> x in names(df), obs_names) || throw(ArgumentError("DataFrame must contain columns $(obs_names)"))
        lkl = logpdf.([MvNormal(Vector(s_), Σ) for s_ in eachrow(df)], Vector.(eachrow(data)))
        return lkl
    end
end

"""
    abc_likelihood(data::DataFrame, ε::Float64, distance_metric::Function) -> Function

Create an Approximate Bayesian Computation (ABC) likelihood function.

This returns a likelihood function that computes log-likelihood for parameter samples based on 
observed data, using the Approximate Bayesian Computation (ABC) [biBhattacharyyaDistanceEnriching2019](@cite) approach with a specified distance metric. The function computes

```math
\\mathcal{L} = -\\frac{d}{\\varepsilon}
```

The distance function receives a single row of the DataFrame and the enire observed data, so for a single-data distance the distance function needs to loop over the data. For example, for a single-data Euclidean distance, the distance function could be defined as:
```julia
distance_metric = (p, d) -> begin
    sqrt(sum([(p .- d_) .^ 2 for d_ in d]))
end
```

For applications that compare model outputs to the entire observed dataset, the distance function can be defined as:

```julia
distance_metric = (a, b) = sqrt((mean(a) - mean(b))^2)
```

i.e., the Euclidean distance between the means of the two datasets.

# Arguments
- `data::DataFrame`: Observed data where each column represents an observed variable.
- `ε::Float64`: Tolerance threshold for the distance metric.
- `distance_metric::Function`: A function that computes the distance between two values.

# Returns
A function `likelihood(df::DataFrame)` that:
- Accepts a DataFrame of parameter samples (columns must match `data` column names)
- Returns a vector of log-likelihoods (one per row)

# Example
```julia
observed = DataFrame(x = [1.0, 2.0], y = [3.0, 4.0])
ε = 0.1
distance_metric = (a, b) -> abs(a - b)
lik = abc_likelihood(observed, ε, distance_metric)
samples = DataFrame(x = [0.9, 2.1], y = [3.1, 3.9])
loglik = lik(samples)  # Returns log-likelihoods for each sample
```
"""
function abc_likelihood(data::DataFrame, ε::Float64, distance_metric::Function)
    obs_names = names(data)
    return function likelihood(df::DataFrame)
        @assert all(x -> x in names(df), obs_names) "DataFrame must contain columns $(obs_names)"
        lkl = zeros(size(df, 1))
        for _n in obs_names
            lkl .-= [distance_metric(row[_n], data[!, _n]) ./ ε for row in eachrow(df)]
        end
        return lkl
    end
end
