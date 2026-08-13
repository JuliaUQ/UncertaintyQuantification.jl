function distribution_parameters(mean::Real, std::Real, _::Type{Distributions.Beta})
    if 0 < mean < 1
        α = ((1 - mean) / std^2 - (1 / mean)) * mean^2
        β = α * ((1 / mean) - 1)
    else
        error(
            "provided mean value $mean is not compatible with Beta distribution support: (0; 1)",
        )
    end
    return α, β
end

function distribution_parameters(mean::Real, std::Real, _::Type{Distributions.Gamma})
    α = (mean / std)^2
    θ = std^2 / mean
    return α, θ
end

function distribution_parameters(mean::Real, std::Real, _::Type{Distributions.Gumbel})
    β = std * sqrt(6) / π
    μ = mean - MathConstants.γ * β
    return μ, β
end

function distribution_parameters(mean::Real, std::Real, _::Type{Distributions.Logistic})
    θ = std / (π * sqrt(1 / 3))
    return mean, θ
end

function distribution_parameters(mean::Real, std::Real, _::Type{Distributions.LogNormal})
    μ = log(mean^2 / sqrt(std^2 + mean^2))
    σ = sqrt(log(std^2 / mean^2 + 1))
    return μ, σ
end

function distribution_parameters(
        val::Real, d::Type{Distributions.Rayleigh}; std::Bool = false
    )
    if !std
        σ = val / sqrt(π / 2)
    else
        σ = val / sqrt(2 - π / 2)
    end
    return σ
end

function distribution_parameters(mean::Real, std::Real, _::Type{Distributions.Uniform})
    a = mean - sqrt(3) * std
    b = mean + sqrt(3) * std
    return a, b
end
