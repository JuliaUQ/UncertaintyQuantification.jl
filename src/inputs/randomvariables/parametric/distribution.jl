function to_standard_normal_space!(
    rv::RandomVariable{<:UnivariateDistribution}, x::DataFrame
)
    x[!, rv.name] = quantile.(Normal(), cdf.(rv.dist, x[:, rv.name]))
    return nothing
end

function to_physical_space!(rv::RandomVariable{<:UnivariateDistribution}, x::DataFrame)
    x[!, rv.name] = quantile.(rv.dist, cdf.(Normal(), x[:, rv.name]))
    return nothing
end

function to_standard_normal_space!(rv::RandomVariable{<:Normal}, x::DataFrame)
    μ, σ = params(rv.dist)
    # do nothing for standard normal rv
    if μ == 0.0 && σ == 1.0
        return nothing
    else
        x[!, rv.name] = (x[:, rv.name] .- μ) ./ σ
        return nothing
    end
end

function to_physical_space!(rv::RandomVariable{<:Normal}, x::DataFrame)
    μ, σ = params(rv.dist)
    # do nothing for standard normal rv
    if μ == 0.0 && σ == 1.0
        return nothing
    else
        x[!, rv.name] = x[:, rv.name] .* σ .+ μ
        return nothing
    end
end
