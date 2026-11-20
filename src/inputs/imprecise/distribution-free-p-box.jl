"""
    DistributionFreeProbabilityBox(lower_cdf::Function, upper_cdf::Function, lb::Real, ub::Real; n::Integer = 10^6, atol::Real = 1.0e-8)
    DistributionFreeProbabilityBox(lower::EmpiricalDistribution, upper::EmpiricalDistribution)

Represents a distribution-free probability box (p-box): all the cdfs between a lower cdf and an upper cdf, with no assumption on their distribution family. Unlike a [`ProbabilityBox`](@ref), it has no parameters.

To use the `DistributionFreeProbabilityBox` in an analysis it has to be wrapped in a [`RandomVariable`](@ref). In [`probability_of_failure`](@ref) only [`RandomSlicing`](@ref) is supported: [`DoubleLoop`](@ref) optimizes over the parameters of the p-boxes, and a distribution-free p-box has none.

# Constructors

- `DistributionFreeProbabilityBox(lower_cdf, upper_cdf, lb, ub; n = 10^6, atol = 1e-8)`: from two cdf functions `x -> F(x)` and the support `[lb, ub]` of the p-box, where `lb` can be `-Inf` and `ub` can be `Inf`. The lower cdf is the one of the distribution furthest to the right. With no tolerance, the constructor checks that:
    - both cdfs are exactly 0 at `lb` and 1 at `ub` (at an infinite end, that they go to 0 or 1),
    - both cdfs are in [0, 1] and non-decreasing,
    - `lower_cdf` is never above `upper_cdf`.
  These are checked at `n` evenly spaced points over the region where each cdf rises: the support, or at an infinite end up to where the cdf is within `atol` of 0 or 1 (beyond it only the limit at `-Inf` or `Inf` is checked). The cdfs are stored as given: their tails are never cut.
- `DistributionFreeProbabilityBox(lower, upper)`: from two [`EmpiricalDistribution`](@ref)s, which are cdfs by construction, so only their order is checked, exactly, at the knots of their cdf splines. Pass the distributions, not `x -> cdf(d, x)`: rounding in the spline evaluation makes these functions fail the exact checks above.

# Methods

- `cdf(pbox, x)`: the [`Interval`](@ref) from `lower_cdf(x)` to `upper_cdf(x)`.
- `quantile(pbox, u)`: the [`Interval`](@ref) from the quantile of the upper cdf to the quantile of the lower cdf, found by inverting the cdfs exactly (for empirical bounds, their own quantiles). At `u = 0` and `u = 1` it is the end of the support.
- `rand(pbox, n)`: `n` random quantile intervals.

# Examples

```jldoctest
julia> pbox = DistributionFreeProbabilityBox(x -> cdf(Normal(1, 1), x), x -> cdf(Normal(0, 1), x), -Inf, Inf)
DistributionFreeProbabilityBox(median = [0.0, 1.0], -Inf, Inf)

julia> cdf(pbox, 0.5)
[0.3085375387259869, 0.6914624612740131]

julia> DistributionFreeProbabilityBox(x -> cdf(Uniform(1, 3), x), x -> cdf(Uniform(0, 2), x), 0, 3)
DistributionFreeProbabilityBox(median = [1.0, 2.0], 0, 3)
```
"""
struct DistributionFreeProbabilityBox
    lower_cdf::Function
    upper_cdf::Function
    lower_quantile::Function
    upper_quantile::Function

    function DistributionFreeProbabilityBox(
            lower_cdf::Function,
            upper_cdf::Function,
            lb::Real,
            ub::Real;
            n::Integer = 10^6,
            atol::Real = 1.0e-8
        )
        if !(lb < ub)
            throw(ArgumentError("the support must have lb < ub, got [$lb, $ub]"))
        end

        # points where the cdfs are checked: n/2 evenly spaced points over the region where each cdf rises (the support [lb, ub] if it's finite, and at an infinite end up to where the cdf is within atol of 0 (left) or 1 (right)), merged. Both cdfs are checked on all of them, so each one has at least n/2 points where it rises, even when the two regions are far apart or very different in width
        lower_a, lower_b = _rising_region(lower_cdf, "lower_cdf", lb, ub; atol = atol)
        upper_a, upper_b = _rising_region(upper_cdf, "upper_cdf", lb, ub; atol = atol)
        lower_x = range(lower_a, lower_b, n ÷ 2)
        upper_x = range(upper_a, upper_b, n ÷ 2)
        x = sort(unique(vcat(lower_x, upper_x)))

        F_lower = lower_cdf.(x)
        F_upper = upper_cdf.(x)

        _check_isa_cdf(lower_cdf, F_lower, x, lb, ub, "lower_cdf")
        _check_isa_cdf(upper_cdf, F_upper, x, lb, ub, "upper_cdf")

        _check_bounds_order(F_lower, F_upper, x)

        # quantiles: each cdf is inverted exactly at every call
        return new(
            lower_cdf,
            upper_cdf,
            u -> _invert_cdf(lower_cdf, u, lb, ub),
            u -> _invert_cdf(upper_cdf, u, lb, ub)
        )
    end

    # Empirical distributions are already valid cdfs, so only the order check is needed
    function DistributionFreeProbabilityBox(
            lower::EmpiricalDistribution,
            upper::EmpiricalDistribution
        )
        # evaluate the splines on all the knots at once
        x = sort(unique(vcat(get_knots(lower.c), get_knots(upper.c))))
        _check_bounds_order(clamp.(lower.c(x), 0.0, 1.0), clamp.(upper.c(x), 0.0, 1.0), x)

        # the quantiles are the ones the empirical distributions already have
        return new(
            x -> clamp(cdf(lower, x), 0.0, 1.0),
            x -> clamp(cdf(upper, x), 0.0, 1.0),
            u -> quantile(lower, u),
            u -> quantile(upper, u)
        )
    end
end

# Quantile of the p-box at u: the interval from the quantile of the upper cdf to the quantile of the lower cdf
function quantile(pbox::DistributionFreeProbabilityBox, u::Real)
    if !(0 <= u <= 1)
        throw(DomainError(u, "u must be in [0, 1]"))
    end
    return Interval(pbox.upper_quantile(u), pbox.lower_quantile(u))
end

# Cdf of the p-box at x: the interval from the lower cdf to the upper cdf
function cdf(pbox::DistributionFreeProbabilityBox, x::Real)
    return Interval(pbox.lower_cdf(x), pbox.upper_cdf(x))
end

# n random samples of the p-box, as for ProbabilityBox: the quantile intervals at n uniform random u
function rand(pbox::DistributionFreeProbabilityBox, n::Integer = 1)
    return quantile.(Ref(pbox), rand(n))
end

Base.broadcastable(pbox::DistributionFreeProbabilityBox) = Ref(pbox)

function Base.show(io::IO, pbox::DistributionFreeProbabilityBox)
    lb = quantile(pbox, 0).lb
    ub = quantile(pbox, 1).ub
    return print(io, "DistributionFreeProbabilityBox(median = ", quantile(pbox, 0.5), ", ", lb, ", ", ub, ")")
end

# Region [a, b] where f rises from 0 to 1. A finite end of the support is used as it is; at an infinite end walk outwards until f is within atol of 0 (left) or 1 (right) (this also checks the limit when f can't be evaluated at ±Inf)
function _rising_region(
        f::Function,
        name::String,
        lb::Real,
        ub::Real;
        atol::Real = 1.0e-8
    )
    if isfinite(lb)
        a = lb
    else
        a = _find_point(x -> abs(f(x)) <= atol, -1.0, "$name never goes to 0 for x → -∞")
    end

    if isfinite(ub)
        b = ub
    else
        b = _find_point(x -> abs(1 - f(x)) <= atol, 1.0, "$name never reaches 1 for x → +∞")
    end

    if !isfinite(lb)
        a = find_zero(x -> f(x) - atol, (a, b))
    end
    if !isfinite(ub)
        b = find_zero(x -> f(x) - (1 - atol), (a, b))
    end
    return a, b
end

# Quantile of the cdf f at u, exact: f is inverted with find_zero, as EmpiricalDistribution finds its support. The bracket is the support [lb, ub], where an infinite end is replaced by a point found walking outwards until f is below u (left) or above it (right). At u = 0 and 1 the quantile is the end of the support
function _invert_cdf(
        f::Function,
        u::Real,
        lb::Real,
        ub::Real
    )
    if u == 0
        return lb
    end
    if u == 1
        return ub
    end

    if isfinite(lb)
        a = lb
    else
        a = _find_point(x -> f(x) <= u, -1.0, "can't invert the cdf at u = $u")
    end

    if isfinite(ub)
        b = ub
    else
        b = _find_point(x -> f(x) >= u, 1.0, "can't invert the cdf at u = $u")
    end

    # f already reaches u at a (f jumps at a)
    if f(a) >= u
        return a
    end
    return find_zero(x -> f(x) - u, (a, b))
end

# Check that f is a cdf on the support [lb, ub], given its values F at the points x: exactly 0 at lb and 1 at ub, 0 ≤ f ≤ 1 and non-decreasing, all with no tolerance. `name` is only used in the error messages
function _check_isa_cdf(
        f::Function,
        F::AbstractVector{<:Real},
        x::AbstractVector{<:Real},
        lb::Real,
        ub::Real,
        name::String
    )
    # ends of the support
    _check_end(f, lb, 0.0, name)
    _check_end(f, ub, 1.0, name)

    # values in [0, 1], exactly
    i = findfirst(v -> !(0 <= v <= 1), F)
    if !isnothing(i)
        throw(
            ArgumentError("$name must be in [0, 1], got $name($(x[i])) = $(F[i])")
        )
    end

    # monotonically non-decreasing, exactly: no step may go down, by any amount
    i = findfirst(<(0), diff(F))
    if !isnothing(i)
        throw(
            ArgumentError(
                "$name must be non-decreasing, it decreases between x = $(x[i]) and x = $(x[i + 1])"
            ),
        )
    end
    return nothing
end

# Check numerically that lower_cdf ≤ upper_cdf, given their values F_lower and F_upper at the points x (only used in the error message)
function _check_bounds_order(
        F_lower::AbstractVector{<:Real},
        F_upper::AbstractVector{<:Real},
        x::AbstractVector{<:Real}
    )
    i = findfirst(F_lower .> F_upper)
    if !isnothing(i)
        throw(
            ArgumentError(
                "lower_cdf must be ≤ upper_cdf, violated at x = $(x[i]): $(F_lower[i]) > $(F_upper[i])"
            ),
        )
    end
    return nothing
end

# Check f(side) == target exactly, at an end of the support. At a finite end f must be exactly the target there (for a non-decreasing f ≥ 0, f(lb) == 0 means f = 0 for all x < lb, and the same for ub). At an infinite end f(±Inf) is only checked if it can be evaluated and isn't NaN (e.g. Inf / Inf): otherwise _rising_region checks the limit instead
function _check_end(
        f::Function,
        side::Real,
        target::Real,
        name::String
    )
    if isfinite(side)
        v = f(side)
    else
        v = try
            f(side)
        catch
            return nothing
        end
        if !(v isa Real) || isnan(v)
            return nothing
        end
    end

    if v != target
        throw(ArgumentError("$name must be $target at x = $side, got $name($side) = $v"))
    end
    return nothing
end

# Walk outwards from x0 (doubling) until condition(x) holds
function _find_point(
        condition::Function,
        x0::Real,
        msg::String;
        maxiter::Integer = 100
    )
    x = x0
    for _ in 1:maxiter
        if condition(x)
            return x
        end
        x *= 2
    end
    throw(ArgumentError(msg))
end
