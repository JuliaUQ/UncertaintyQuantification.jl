"""
    GaussianMixtureModel(
        data::DataFrame,
        number_components::Integer;
        maximum_iterations::Integer = 100,
        tolerance::Number = 1.0e-4,
    )

Fits a Gaussian mixture model to the given data using the Expectation-Maximization (EM) algorithm.

# Arguments
- `data::DataFrame`: The input data as a DataFrame, where each column represents a variable.
- `number_components::Integer`: The number of Gaussian components to fit.
- `maximum_iterations::Integer=100`: (Optional) The maximum number of EM iterations. Default is 100.
- `tolerance::Number=1e-4`: (Optional) The convergence tolerance for the EM algorithm. Default is 1e-4.

# Returns
- `JointDistribution`: A joint distribution object representing the fitted Gaussian mixture model, with variable names corresponding to the columns of `data`.

See also: [`JointDistribution`](@ref)

"""
function GaussianMixtureModel(
        data::DataFrame,
        number_components::Integer;
        maximum_iterations::Integer = 100,
        tolerance::Number = 1.0e-4,
    )
    # Fit mixture model to the data using EM algorithm
    mixture = _gaussian_mixture_expectation_maximization(
        number_components,
        Matrix(data);
        maximum_iterations = maximum_iterations,
        tolerance = tolerance,
    )

    return JointDistribution(mixture, Symbol.(names(data)))
end

function _gaussian_mixture_expectation_maximization(
        number_components::Integer,
        data::Matrix;
        maximum_iterations::Integer = 100,
        tolerance::Number = 1.0e-4,
    )
    # Check inputs
    if number_components <= 0
        throw(ArgumentError("Number of components must be a positive integer."))
    end
    if size(data, 2) < 2
        throw(ArgumentError("Input must be at least two-dimensional to fit GMM."))
    end

    # Initialize parameters
    number_samples, _ = size(data)
    μ, Σ, π = _initialize_gaussian_mixture_model(data, number_components)
    log_likelihood_old = -Inf

    for _ in 1:maximum_iterations
        γ = _expectation_step(data, μ, Σ, π)
        μ, Σ, π = _maximization_step(data, γ)

        # Log-likelihood (for convergence check)
        log_likelihood = sum(
            log(
                    sum(
                        π[k] * pdf(MvNormal(μ[k, :], Σ[k]), data[i, :]) for
                        k in 1:number_components
                    ),
                ) for i in 1:number_samples
        )

        abs(log_likelihood - log_likelihood_old) < tolerance ? break : continue
        log_likelihood_old = log_likelihood
    end

    return MixtureModel([MvNormal(μ[k, :], Σ[k]) for k in 1:number_components], π)
end

function _initialize_gaussian_mixture_model(
        data::Matrix{Float64}, number_components::Integer
    )
    # Randomly initialize parameters for GMM
    number_samples, dimensions = size(data)
    μ = data[rand(1:number_samples, number_components), :]  # random initialization of means
    Σ = [I(dimensions) for _ in 1:number_components]        # identity covariance matrices
    π = fill(1 / number_components, number_components)      # equal mixing weights
    return μ, Σ, π
end

function _expectation_step(
        data::Matrix{Float64}, μ::Matrix{Float64}, Σ::Vector, π::Vector{Float64}
    )
    number_samples, number_components = size(data, 1), length(π)
    # Compute responsibilities and normalize
    γ = [
        π[k] * pdf(MvNormal(μ[k, :], Σ[k]), data[i, :]) for i in 1:number_samples,
            k in 1:number_components
    ]
    γ = γ ./ sum(γ; dims = 2)
    return γ
end

function _maximization_step(data::Matrix{Float64}, γ::Matrix{Float64})
    number_samples, dimensions = size(data)
    number_components = size(γ, 2)
    Nₖ = sum(γ; dims = 1)             # effective number of points in cluster 𝑘
    π = vec(Nₖ ./ number_samples)   # weights: relative number of points in each cluster
    μ = γ' * data ./ Nₖ'            # means: weighted average of points in each cluster
    Σ = Vector{Matrix{Float64}}(undef, number_components)
    for k in 1:number_components
        data_centered = data .- μ[k:k, :]
        Σₖ = zeros(dimensions, dimensions)
        for i in 1:number_samples
            Σₖ += γ[i, k] * (data_centered[i, :] * data_centered[i, :]')
        end
        Σ[k] = Σₖ / Nₖ[k]
    end
    return μ, Σ, π
end
