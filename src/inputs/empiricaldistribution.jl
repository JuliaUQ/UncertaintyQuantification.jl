"""
    EmpiricalDistribution(x::Vector{<:Real}, n::Integer=10000)

    Creates an empirical distribution from the data given in `x` using kernel density estimation.
    The kernel used is Gaussian and the bandwidth is obtained through the Sheather-Jones method.
    The support is inferred from the kde using numerical root finding.
    The `cdf` and `quantile` functions are linearly interpolated using `n` data points.

    For large datasets linear binning can be used by passing the keyword `nbins`.
"""
struct EmpiricalDistribution <: ContinuousUnivariateDistribution
    data::Union{AbstractVector{<:Real},BinnedData}
    lb::Real
    ub::Real
    h::Real
    c::Spline1D
    q::Spline1D

    function EmpiricalDistribution(
        X::AbstractVector{<:Real}, n::Integer=10000; nbins::Integer=0
    )
        h, data = sheather_jones_bandwidth(X, nbins)

        f = u -> kde(h, u, data) - eps(eltype(X))

        xmin = minimum(X)
        xmax = maximum(X)

        lb = if f(xmin - 10 * h) < 0 && f(xmin) > 0
            find_zero(f, (xmin - 10 * h, xmin); maxevals=10^3)
        else
            @warn "EmpiricalDistribution: Unable to compute compute accurate lower bound."
            xmin - 10 * h # conservative estimate for the lower bound
        end

        ub = if f(xmax) > 0 && f(xmax + 10 * h) < 0
            find_zero(f, (xmax, xmax + 10 * h); maxevals=10^3)
        else
            @warn "EmpiricalDistribution: Unable to compute compute accurate lower bound."
            xmax + 10 * h # conservative estimate for the upper bound
        end

        x = collect(range(lb, ub, n))

        f = u -> kde(h, u, data)

        y = zeros(eltype(x), size(x))
        for i in eachindex(x)
            y[i] = if i == 1
                quadgk(f, lb, x[i])[1]
            else
                y[i - 1] + quadgk(f, x[i - 1], x[i])[1]
            end
        end

        clamp!(y, 0.0, 1.0)

        y[1] = 0.0
        y[end] = 1.0

        c = Spline1D(x, y; k=1, s=0.0)

        unique_idx = findlast.(isequal.(unique(y)), [y])

        q = Spline1D(y[unique_idx], x[unique_idx]; k=1, s=0.0)

        return new(data, lb, ub, h, c, q)
    end
end

function cdf(d::EmpiricalDistribution, x::Real)
    return d.c(x)
end

function quantile(d::EmpiricalDistribution, u::Real)
    return d.q(u)
end

function pdf(d::EmpiricalDistribution, x::Real)
    return insupport(d, x) ? kde(d.h, x, d.data) : zero(x)
end

function logpdf(d::EmpiricalDistribution, x::Real)
    return log(pdf(d, x))
end

function rand(rng::AbstractRNG, d::EmpiricalDistribution)
    return quantile(d, rand(rng))
end

insupport(d::EmpiricalDistribution, x::Real) = d.lb <= x <= d.ub
minimum(d::EmpiricalDistribution) = d.lb
maximum(d::EmpiricalDistribution) = d.ub
mean(d::EmpiricalDistribution) = quadgk(x -> x * pdf(d, x), d.lb, d.ub)[1]
var(d::EmpiricalDistribution) = quadgk(x -> x^2 * pdf(d, x), d.lb, d.ub)[1] - mean(d)^2
