# standard gaussian pdf
function ϕ(x::Real)
    return exp(-x^2 / 2) / sqrt(2 * π)
end
# 4th derivative of the standard gaussian pdf
function ϕ4(x::Real)
    return (exp(-x^2 / 2) * (x^4 - 6x^2 + 3)) / sqrt(2 * π)
end
# 6th derivative of the standard gaussian pdf
function ϕ6(x::Real)
    return (exp(-x^2 / 2) * (x^6 - 15x^4 + 45x^2 - 15)) / sqrt(2 * π)
end

function TD(b::Real, x::Vector{<:Real})
    n = length(x)

    val = zero(eltype(x))

    @inbounds @simd for i in 1:(n - 1)
        for j in (i + 1):n
            val += @views 2 * ϕ6(b^(-1) * (x[i] - x[j]))
        end
    end

    val += n * ϕ6(zero(eltype(x)))

    return -1 * (n * (n - 1))^(-1) * b^(-7) * val
end

function TD(b::Real, data::BinnedData)
    n = sum(data.weights)
    nbins = length(data.grid)

    val = zero(eltype(data.grid))

    @inbounds @simd for i in 1:(nbins - 1)
        for j in (i + 1):nbins
            val += @views 2 *
                data.weights[i] *
                data.weights[j] *
                ϕ6(b^(-1) * (data.grid[i] - data.grid[j]))
        end
    end

    val += sum(data.weights .^ 2) * ϕ6(zero(eltype(data.grid)))

    return -1 * (n * (n - 1))^(-1) * b^(-7) * val
end

function SD(α::Real, x::AbstractVector{<:Real})
    n = length(x)

    val = zero(eltype(x))

    @inbounds @simd for i in 1:(n - 1)
        for j in (i + 1):n
            val += @views 2 * ϕ4(α^(-1) * (x[i] - x[j]))
        end
    end

    val += n * ϕ4(zero(eltype(x)))

    return (n * (n - 1))^(-1) * α^(-5) * val
end
function SD(α::Real, data::BinnedData)
    n = sum(data.weights)
    nbins = length(data.grid)

    val = zero(eltype(data.grid))

    @inbounds @simd for i in 1:(nbins - 1)
        for j in (i + 1):nbins
            val += @views 2 *
                data.weights[i] *
                data.weights[j] *
                ϕ4(α^(-1) * (data.grid[i] - data.grid[j]))
        end
    end

    val += sum(data.weights .^ 2) * ϕ4(zero(eltype(data.grid)))

    return (n * (n - 1))^(-1) * α^(-5) * val
end

#=
S. J. Sheather, M. C. Jones, A Reliable Data-Based Bandwidth Selection Method for Kernel Density Estimation, Journal of the Royal Statistical Society: Series B (Methodological), Volume 53, Issue 3, July 1991, Pages 683–690, https://doi.org/10.1111/j.2517-6161.1991.tb01857.x
=#
function sheather_jones_bandwidth(x::AbstractVector, nbins::Integer=0)
    λ = iqr(x)
    n = length(x)

    data = if nbins > 0
        linear_binning(x, nbins)
    else
        x
    end

    a = 0.920λ * n^(-1 / 7)
    b = 0.912λ * n^(-1 / 9)

    α₂ = 1.357 * (SD(a, data) / TD(b, data))^(1 / 7)

    function f(h)
        α₂_h = α₂ * h^(5 / 7)

        return ((1 / (2 * sqrt(π))) / (3 * SD(α₂_h, data)))^(1 / 5) * n^(-1 / 5) - h
    end

    return newtonraphson(1.0, f, 1e-3, 1e-10, 10^3), data
end

function kde(h::Real, x::Real, X::AbstractVector{<:Real})
    return length(X)^(-1) * sum([h^(-1) * ϕ(h^(-1) * (x - xⱼ)) for xⱼ in X])
end

function kde(h::Real, x::Real, X::BinnedData)
    return sum(X.weights)^(-1) *
           sum([h^(-1) * wⱼ * ϕ(h^(-1) * (x - xⱼ)) for (xⱼ, wⱼ) in zip(X.grid, X.weights)])
end
