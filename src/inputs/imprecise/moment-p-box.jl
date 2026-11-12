"""
    MomentProbabilityBox(; mean = nothing, var = nothing, lb = -Inf, ub = Inf, range = nothing)
    ProbabilityBox(; mean = nothing, var = nothing, lb = -Inf, ub = Inf, range = nothing)

Construct a distribution-free p-box from moment constraints and/or support
bounds. Return a [`MomentProbabilityBox`](@ref), a concrete subtype of
`AbstractProbabilityBox{MomentConstraints}` with explicit `mean`, `var` and `range` fields.
`mean` and `var` accept a `Real` or an [`Interval`](@ref); `var` is the
population variance, not the standard deviation. An omitted variance is unknown.
An omitted mean requires finite support bounds and is inferred from the support
and variance constraints. Omitting both moments gives a range-only p-box.
`range=Interval(a, b)` is shorthand for `lb=a, ub=b`.

# Fields
- `mean::Interval`: Mean values; a precise input becomes a point interval.
- `var::Interval`: Variances; a precise input becomes a point interval.
- `range::Interval`: Support bounds, which may have infinite endpoints.

# Bound construction and selection

Constructions follow Sections 6.1–6.3 of 
[Gray et al. (2022)](https://doi.org/10.1016/j.ijar.2022.04.001):

| Available information | P-box bounds | Construction |
|:--|:--|:--|
| Mean + variance, unbounded support | One-sided Chebyshev | Mean–variance tail inequalities. |
| Mean + lower or upper bound | Markov | Apply Markov to `X-lb` or `ub-X`. |
| Mean + finite range, unknown variance | Markov from both endpoints | Intersect the shifted and reflected Markov bounds. |
| Mean + variance + one support endpoint | Cantelli construction in Gray et al. | Combine one-sided Chebyshev and Markov using the known endpoint (Section 6.2). |
| Mean + variance + finite range | Min–max–mean–variance | Optimize the bounded moment constraints (Section 6.3). |
| Variance + finite range, unknown mean | Min–max–mean–variance | Envelope the bounds over all feasible means inferred from the range and variance. |
| Finite range only | Range envelope | All distributions supported within the range remain possible. |


# Feasibility and interval moments

Moment intervals describe all feasible moment values within the intervals.
They are intersected with the support's theoretical moment bounds; inconsistent
constraints throw `ArgumentError`. CDF and quantile bounds envelope all feasible
distributions, including discrete distributions. The envelopes can also contain
distributions that do not satisfy the original moment constraints.

Wrap the result in [`RandomVariable`](@ref) for sampling and [`RandomSlicing`](@ref).
Parametric [`DoubleLoop`](@ref) analysis is not supported.

Check [`check_moments_and_range`](@ref) for how feasibility constraints are computed

# References

- Chebyshev, *[Des valeurs moyennes](https://www.numdam.org/item/JMPA_1867_2_12__177_0/)*
  (1867), pp. 177–184: the classical two-sided inequality.
- Chebyshev, *[Sur les valeurs limites des intégrales](https://www.numdam.org/item/JMPA_1874_2_19__157_0/)*
  (1874), pp. 159–160: piecewise bounds from range, mean and variance; their
  unbounded-support limits give the one-sided mean–variance bounds.
- Markoff, *[Sur une question de maximum et de minimum proposée par M. Tchebycheff](https://archive.ymsc.tsinghua.edu.cn/pacm_download/117/4885-11511_2006_Article_BF02406729.pdf)*
  (1886), *Acta Mathematica* **9**, pp. 57–70: historical background on the
  general moment-bounds problem. See [`markov_cdf`](@ref) for the elementary
  inequality used here and its derivation.
- Cantelli, *[Sui confini della probabilità](https://www.mathunion.org/fileadmin/ICM/Proceedings/ICM1928.6/ICM1928.6.ocr.pdf)*,
  proceedings of the 1928 Bologna congress (published 1932), vol. 6, pp. 47–59:
  Section 7, p. 53, equations (19) and (21) give the one-sided mean–variance bounds.
- Gray, Ferson, Kreinovich and Patelli,
  *[Distribution-free risk analysis](https://doi.org/10.1016/j.ijar.2022.04.001)*
  (2022), Section 6: p-box constructions followed here, with
  CDF orientation and quantiles treated explicitly.

# Examples
```julia
ProbabilityBox(; mean = 0, var = 1)                 # Chebyshev
ProbabilityBox(; mean = 2, range = Interval(0, 5))  # Markov from both endpoints
ProbabilityBox(; mean = 2, lb = 0)                 # shifted Markov
ProbabilityBox(; mean = 2, ub = 5)                 # reflected Markov
ProbabilityBox(; mean = 0, var = 3, lb = -1)        # Cantelli
ProbabilityBox(; mean = 0, var = 1, ub = 2)         # reflected Cantelli
ProbabilityBox(; mean = Interval(1, 2), var = Interval(0.2, 1), range = Interval(0, 5))
ProbabilityBox(; var = Interval(0.2, 1), range = Interval(0, 5))
ProbabilityBox(; range = Interval(0, 5))           # range only
MomentProbabilityBox(; mean = 0, var = 1)          # direct constructor
```
"""
struct MomentProbabilityBox <: AbstractProbabilityBox{MomentConstraints}
    mean::Interval
    var::Interval
    range::Interval

    function MomentProbabilityBox(;
            mean::Union{Nothing, Real, Interval} = nothing,
            var::Union{Nothing, Real, Interval} = nothing,
            lb::Real = -Inf, ub::Real = Inf, range::Union{Nothing, Interval} = nothing,
        )
        if range !== nothing
            (lb == -Inf && ub == Inf) ||
                throw(ArgumentError("Specify either range or lb/ub, not both."))
            lb, ub = bounds(range)
        end
        mean, var, lb, ub = check_moments_and_range(; mean, var, lb, ub)
        return new(mean, var, Interval(lb, ub))
    end
end

"""
    check_moments_and_range(; mean = nothing, var = nothing, lb = -Inf, ub = Inf)

Check consistency and tighten moment information using theoretical bounds implied
by the support. Return `(mean_interval, variance_interval, lb, ub)`. 

For example, for a range = [a, b], is impossible to have:

`mean ∩ [a, b] = ∅`
`var ∩ [0, (b-a)^2/4] = ∅`

We may thus check that these intersections are non-empty, but also tightened intervals.
If the mean and variance are known, their theoretical bounds are returned. An empty intersection is impossible and throws `ArgumentError`; for example, `mean=Interval(3, 4)` cannot be consistent with `lb=0, ub=2`.

An omitted mean requires finite support bounds and starts as the full range.
An omitted variance starts as `[0, Inf]` and requires at least one finite support
bound.

For finite support `[a, b]`, feasible moments satisfy `v ≤ (m-a)*(b-m)`.
With no mean information this gives `[0, (b-a)^2/4]`, whose upper bound is
attained by equal probability masses at the two endpoints (the maximum variance of an interval). 
A specified mean can give a smaller upper bound, attained by suitable endpoint probabilities.
Evaluate this quadratic at both mean endpoints and at the range midpoint
`a + (b-a)/2` when it belongs to the mean interval. Intersect the supplied variance
with `[0, max(v1, v2, v3)]`. As an extra tightening step, solving
`(m-a)*(b-m) ≥ variance_interval.lb` tightens the mean interval. An exactly known
mean at a finite support endpoint forces zero variance. Inconsistent joint
constraints also throw `ArgumentError`.
"""
function check_moments_and_range(;
        mean::Union{Nothing, Real, Interval} = nothing,
        var::Union{Nothing, Real, Interval} = nothing,
        lb::Real = -Inf, ub::Real = Inf,
    )
    (lb < Inf && ub > -Inf && lb <= ub) ||
        throw(ArgumentError("Support bounds must be ordered and contain finite values. Provided lb=$lb, ub=$ub."))
    support = Interval(lb, ub)
    if mean === nothing
        mean = support
    end
    provided_mean = mean
    mean = mean ∩ support
    isempty(mean) && throw(
        ArgumentError(
            "Mean ∩ range = ∅.\nProvided mean: $provided_mean\nTheoretical mean bounds: $support",
        )
    )
    # Intersecting with unbounded support can still leave infinite mean bounds.
    (isfinite(lo(mean)) && isfinite(hi(mean))) ||
        throw(ArgumentError("Mean bounds must be finite after intersecting the support."))

    if var === nothing
        (isfinite(lb) || isfinite(ub)) ||
            throw(ArgumentError("Supply a variance or at least one finite support bound."))
        var = Interval(0.0, Inf)
    end
    if isfinite(lb) && isfinite(ub)
        ml, mh = bounds(mean)
        v1 = (ml - lb) * (ub - ml)
        v2 = (mh - lb) * (ub - mh)
        v3 = 0

        midpoint = lb + (ub - lb) / 2
        if midpoint ∈ mean
            v3 = (midpoint - lb) * (ub - midpoint)
        end

        max_var = Interval(0, max(v1, v2, v3))
    elseif (lo(mean) == hi(mean) == lb || lo(mean) == hi(mean) == ub)
        max_var = Interval(0, 0)
    else
        max_var = Interval(0, Inf)
    end
    provided_var = var
    var = var ∩ max_var
    isempty(var) && throw(
        ArgumentError(
            "Variance ∩ theoretical variance bounds = ∅.\nProvided variance: $provided_var\nTheoretical variance bounds: $max_var\nMean used for variance bounds: $mean; range: $support",
        )
    )
    isfinite(lo(var)) || throw(ArgumentError("Variance must have a finite lower bound."))

    # Variance feasibility above guarantees this mean intersection is nonempty.
    if isfinite(lb) && isfinite(ub) && lo(var) > 0
        mean = mean ∩ _mean_bounds_from_variance(mean, lo(var), lb, ub)
    end
    return mean, var, lb, ub
end

# Solve the quadratic analytically: mean ∈ midpoint ± sqrt(halfwidth²-v_min).
# Use the equivalent rationalized form near the endpoints to avoid cancellation.
function _mean_bounds_from_variance(mean::Interval, v_min::Real, lb::Real, ub::Real)
    halfwidth = (ub - lb) / 2
    midpoint = lb + halfwidth
    radius = sqrt(max(0, halfwidth^2 - v_min))
    offset = v_min / (halfwidth + radius)
    lower = v_min > halfwidth^2 / 2 ? midpoint - radius : lb + offset
    upper = v_min > halfwidth^2 / 2 ? midpoint + radius : ub - offset

    # Preserve input endpoints already certified as feasible by the product
    # bound, even if rounding in the analytic roots would exclude them.
    if (mean.lb - lb) * (ub - mean.lb) >= v_min
        lower = min(lower, mean.lb)
        upper = max(upper, mean.lb)
    end
    if (mean.ub - lb) * (ub - mean.ub) >= v_min
        lower = min(lower, mean.ub)
        upper = max(upper, mean.ub)
    end
    return Interval(lower, upper)
end

ProbabilityBox(; kwargs...) = MomentProbabilityBox(; kwargs...)

mean(p::MomentProbabilityBox) = p.mean
var(p::MomentProbabilityBox) = p.var
minimum(p::MomentProbabilityBox) = iszero(var(p).ub) ? mean(p).lb : p.range.lb
maximum(p::MomentProbabilityBox) = iszero(var(p).ub) ? mean(p).ub : p.range.ub
insupport(p::MomentProbabilityBox, x::Real) = minimum(p) <= x <= maximum(p)

function bounds(::MomentProbabilityBox)
    throw(ArgumentError("Moment p-boxes have no distribution parameters for DoubleLoop; use RandomSlicing."))
end

"""
    chebyshev_cdf(x, mean, var)

Return the interval CDF bound from the Chebyshev inequality using a precise 
or interval mean and variance on unbounded support.
For `t > 0`, use `P(X-mean ≥ t) ≤ var/(var+t^2)` and its reflection.
Only the mean and variance are required. This is the one-sided refinement of
the classical two-sided inequality `P(abs(X-mean) ≥ t) ≤ var/t^2`.

The one-sided formula follows by taking unbounded-support limits in
[Chebyshev (1874)](https://www.numdam.org/item/JMPA_1874_2_19__157_0/), pp. 159–160.
It is also known as Chebyshev–Cantelli or Cantelli's inequality; an explicit later
statement is in Cantelli's
*[Sui confini della probabilità](https://www.mathunion.org/fileadmin/ICM/Proceedings/ICM1928.6/ICM1928.6.ocr.pdf)*,
Section 7, p. 53, equations (19) and (21), in the 1928 Bologna congress proceedings
(vol. 6, published 1932). 

Moment constraints must be consistent; see [`check_moments_and_range`](@ref).
"""
function chebyshev_cdf(x::Real, m::Union{Real, Interval}, v::Union{Real, Interval})
    isnan(x) && throw(ArgumentError("CDF argument must not be NaN."))
    x == -Inf && return Interval(0.0, 0.0)
    x == Inf && return Interval(1.0, 1.0)
    if iszero(hi(v))
        return Interval(x >= hi(m) ? 1.0 : 0.0, x >= lo(m) ? 1.0 : 0.0)
    end
    upper = _chebyshev_cdf_upper(x, lo(m), hi(v))
    lower = 1 - _chebyshev_cdf_upper(-x, -hi(m), hi(v))
    return Interval(lower, upper)
end

function _chebyshev_cdf_upper(x, m, v)
    return x < m ? (isinf(v) ? 1.0 : v / (v + (m - x)^2)) : 1.0
end

"""
    chebyshev_quantile(u, mean, var)

Return the explicit inverse Chebyshev bounds at probability `u ∈ [0, 1]` for a
precise or interval mean and variance. Moment constraints must be consistent.
These invert [`chebyshev_cdf`](@ref); see that function for the sources and naming.
"""
function chebyshev_quantile(u::Real, m::Union{Real, Interval}, v::Union{Real, Interval})
    0 <= u <= 1 || throw(ArgumentError("Quantile probability must be in [0, 1]."))
    ml, mh, vh = lo(m), hi(m), hi(v)
    iszero(vh) && return Interval(ml, mh)
    iszero(u) && return Interval(-Inf, mh)
    u == 1 && return Interval(ml, Inf)
    return Interval(
        ml - sqrt(vh) * (sqrt(1 - u) / sqrt(u)),
        mh + sqrt(vh) * (sqrt(u) / sqrt(1 - u)),
    )
end

"""
    markov_cdf(x, mean, lb, ub)

Return shifted/reflected Markov CDF bounds from a precise or interval mean and
support endpoints. Either endpoint may be infinite. Moment and support
constraints must be consistent; see [`check_moments_and_range`](@ref).

Apply `P(Y ≥ y) ≤ E[Y]/y` for `y > 0` to the nonnegative variable `Y=X-lb`
or `Y=ub-X`. The inequality follows directly from
`y*1{Y ≥ y} ≤ Y` by taking expectations. For background on the
general moment-bounds problem, see
[Markoff (1886)](https://archive.ymsc.tsinghua.edu.cn/pacm_download/117/4885-11511_2006_Article_BF02406729.pdf),
*Acta Mathematica* **9**, pp. 57–70.
"""
function markov_cdf(x::Real, m::Union{Real, Interval}, a::Real, b::Real)
    isnan(x) && throw(ArgumentError("CDF argument must not be NaN."))
    x == -Inf && return Interval(0.0, 0.0)
    x >= b && return Interval(1.0, 1.0)
    x < a && return Interval(0.0, 0.0)
    # A mean at the lower endpoint forces a point mass, including F(a)=1.
    hi(m) == a && return Interval(1.0, 1.0)
    upper = isfinite(b) ? clamp((b - lo(m)) / (b - x), 0, 1) : 1.0
    lower = isfinite(a) && x > a ? clamp(1 - (hi(m) - a) / (x - a), 0, 1) : 0.0
    return Interval(lower, upper)
end

"""
    markov_quantile(u, mean, lb, ub)

Return explicit inverse Markov bounds at probability `u ∈ [0, 1]` from a precise
or interval mean and support endpoints. Moment and support constraints must be
consistent. Either endpoint may be infinite.
These invert [`markov_cdf`](@ref); see that function for the construction and source.
"""
function markov_quantile(u::Real, m::Union{Real, Interval}, a::Real, b::Real)
    0 <= u <= 1 || throw(ArgumentError("Quantile probability must be in [0, 1]."))
    ml, mh = lo(m), hi(m)
    if ml == mh == a || ml == mh == b
        return Interval(ml, mh)
    end
    iszero(u) && return Interval(a, mh)
    u == 1 && return Interval(ml, b)
    lower = isfinite(b) ? max(a, b - (b - ml) / u) : a
    upper = isfinite(a) ? min(b, a + (mh - a) / (1 - u)) : b
    return Interval(lower, upper)
end

"""
    cantelli_cdf(x, mean, var, lb, ub)

Return the combined p-box construction called Cantelli bounds in Section 6.2,
equations (37)–(38),
of [Gray et al. (2022)](https://doi.org/10.1016/j.ijar.2022.04.001):
combine the [`chebyshev_cdf`](@ref) mean–variance bounds with shifted/reflected
[`markov_cdf`](@ref) bounds. One support endpoint is known; use `-Inf` or `Inf`
for the other. Moment and support constraints must be consistent. With two finite
endpoints, [`meanvarminmax_cdf`](@ref) uses their full joint moment constraints.
The construction's name follows Gray et al. The one-sided Chebyshev inequality
it uses is itself also called Cantelli's inequality and needs only mean and
variance; see [`chebyshev_cdf`](@ref) for its formula and historical sources.
"""
function cantelli_cdf(
        x::Real, m::Union{Real, Interval}, v::Union{Real, Interval}, a::Real, b::Real,
    )
    moment = chebyshev_cdf(x, m, v)
    support = markov_cdf(x, m, a, b)
    return Interval(max(moment.lb, support.lb), min(moment.ub, support.ub))
end

"""
    cantelli_quantile(u, mean, var, lb, ub)

Return the explicit inverse Cantelli bounds of Section 6.2 of Gray et al. (2022)
for consistent mean, variance and one known support endpoint. Use `-Inf` or `Inf`
for the unknown endpoint. Combine the Chebyshev and Markov inverses for
`0 < u < 1`; at endpoint probabilities, also use the variance lower bound to
constrain the possible essential infimum/supremum.
See [`cantelli_cdf`](@ref) for the source and naming convention.
"""
function cantelli_quantile(
        u::Real, m::Union{Real, Interval}, v::Union{Real, Interval}, a::Real, b::Real,
    )
    moment = chebyshev_quantile(u, m, v)
    iszero(hi(v)) && return moment
    if iszero(u)
        return Interval(a, -_moment_least_upper(-hi(m), -lo(m), lo(v), -b))
    elseif u == 1
        return Interval(_moment_least_upper(lo(m), hi(m), lo(v), a), b)
    end
    support = markov_quantile(u, m, a, b)
    return Interval(max(moment.lb, support.lb), min(moment.ub, support.ub))
end

# Maximal variance forces a unique two-point law at the support endpoints.
function _moment_endpoint_distribution(m, v, a, b)
    return lo(m) == hi(m) && isfinite(a) && isfinite(b) &&
        lo(v) == hi(v) == (lo(m) - a) * (b - lo(m))
end

"""
    meanvarminmax_cdf(x, mean, var, lb, ub)

Return the mean–variance–minimum–maximum CDF bounds described in Section 6.3 of
[Gray et al. (2022)](https://doi.org/10.1016/j.ijar.2022.04.001).
Precise and interval moments are supported, and with an infinite
endpoint, reduce to [`cantelli_cdf`](@ref). Inputs must satisfy [`check_moments_and_range`](@ref).
"""
function meanvarminmax_cdf(
        x::Real, m::Union{Real, Interval}, v::Union{Real, Interval}, a::Real, b::Real,
    )
    (!isfinite(a) || !isfinite(b)) && return cantelli_cdf(x, m, v, a, b)
    isnan(x) && throw(ArgumentError("CDF argument must not be NaN."))
    x == -Inf && return Interval(0.0, 0.0)
    x >= b && return Interval(1.0, 1.0)
    x < a && return Interval(0.0, 0.0)
    iszero(hi(v)) && return chebyshev_cdf(x, m, v)
    if _moment_endpoint_distribution(m, v, a, b)
        f = (b - lo(m)) / (b - a)
        return Interval(f, f)
    end
    ml, mh, vl, vh = lo(m), hi(m), lo(v), hi(v)
    upper = _meanvarminmax_cdf_upper(x, ml, mh, vl, vh, a, b)
    lower = 1 - _meanvarminmax_cdf_upper(-x, -mh, -ml, vl, vh, -b, -a)
    return Interval(clamp(lower, 0, upper), upper)
end

# For fixed m, the optimal variance is nearest to (m-x)(b-m) when x<m,
# and is the smallest feasible variance otherwise. The three-point branch
# is concave in m with vertex (a+x)/2; other branches decrease or equal one.
function _meanvarminmax_cdf_upper(x, ml, mh, vl, vh, a, b)
    x < a && return 0.0
    x >= b && return 1.0
    iszero(vh) && return x >= ml ? 1.0 : 0.0
    m = clamp(a + (x - a) / 2, ml, mh)
    return max(
        _meanvarminmax_cdf_upper_at_mean(x, ml, vl, vh, a, b),
        _meanvarminmax_cdf_upper_at_mean(x, m, vl, vh, a, b),
    )
end

function _meanvarminmax_cdf_upper_at_mean(x, m, vl, vh, a, b)
    vmax = min(vh, (m - a) * (b - m))
    v = x < m ? clamp((m - x) * (b - m), vl, max(vl, vmax)) : vl
    iszero(v) && return x >= m ? 1.0 : 0.0
    if x < m && v < (m - x) * (b - m)
        return _chebyshev_cdf_upper(x, m, v)
    elseif x >= m && v <= (x - m) * (m - a)
        return 1.0
    end
    return clamp(1 - (v + (m - a) * (m - x)) / ((b - a) * (b - x)), 0, 1)
end

# Least possible essential supremum; reflection gives the greatest essential
# infimum. These give the limits at probabilities one and zero, respectively.
function _moment_least_upper(ml, mh, vl, a)
    (!isfinite(a) || iszero(vl)) && return ml
    m = clamp(a + sqrt(vl), ml, mh)
    return m + vl / (m - a)
end

"""
    meanvarminmax_quantile(u, mean, var, lb, ub)

Return explicit quantile bounds at probability `u ∈ [0, 1]` for checked moment
and support constraints.

For interval moments, envelope over the feasible mean endpoints, their one-sided
limits and an analytic interior extremum. With an infinite endpoint, reduce to
[`cantelli_quantile`](@ref).
Endpoint probabilities, zero variance and forced endpoint atoms have explicit
handling, including variance lower bounds. See [`meanvarminmax_cdf`](@ref) for
the corresponding CDF bounds and their sources.
"""
function meanvarminmax_quantile(
        u::Real, m::Union{Real, Interval}, v::Union{Real, Interval}, a::Real, b::Real,
    )
    (!isfinite(a) || !isfinite(b)) && return cantelli_quantile(u, m, v, a, b)
    0 <= u <= 1 || throw(ArgumentError("Quantile probability must be in [0, 1]."))
    ml, mh, vl, vh = lo(m), hi(m), lo(v), hi(v)
    iszero(vh) && return Interval(ml, mh)
    if iszero(u)
        return Interval(a, -_moment_least_upper(-mh, -ml, vl, -b))
    elseif u == 1
        return Interval(_moment_least_upper(ml, mh, vl, a), b)
    elseif _moment_endpoint_distribution(m, v, a, b)
        q = u <= (b - ml) / (b - a) ? a : b
        return Interval(q, q)
    end
    # Pass both complementary probabilities to preserve tiny tails when 1-u
    # rounds to one. Reflection gives the upper quantile bound.
    lower = _meanvarminmax_quantile_lower(u, 1 - u, ml, mh, vl, vh, a, b)
    if ml < mh && vl == (mh - a) * (b - mh) && u == (b - mh) / (b - a)
        # At mh the endpoint law has F(a)=u. Use the limit as m approaches
        # mh from below, rather than reflecting the other end of its plateau.
        upper = mh - (b - mh)
    else
        upper = -_meanvarminmax_quantile_lower(1 - u, u, -mh, -ml, vl, vh, -b, -a)
    end
    return Interval(lower, upper)
end

# The optimum is at a mean endpoint or at the central branch's stationary
# mean: (m-a)/(b-a) = q + sqrt(v_min/(b-a)^2 - p*q), where q=1-p.
function _meanvarminmax_quantile_lower(p, q, ml, mh, vl, vh, a, b)
    lower = _meanvarminmax_quantile_lower_at_mean(p, q, ml, vl, vh, a, b)
    ml == mh && return lower
    lower = min(lower, _meanvarminmax_quantile_lower_at_mean(p, q, mh, vl, vh, a, b))
    width = b - a
    # Avoid squaring a rounded square root and preserve tiny differences at
    # the endpoint-law plateau, including when the complement rounds to one.
    tail = min(p, q)
    discriminant = muladd(tail, tail, (vl / width) / width - tail)
    if discriminant >= 0
        m = a + width * (q + sqrt(discriminant))
        if ml <= m <= mh
            # At the stationary mean the central inverse simplifies to 2m-a;
            # evaluating its quotient directly loses precision near the plateau.
            lower = min(lower, m + (m - a))
        end
    end
    return lower
end

# Intersect the support, Chebyshev, Markov and minimum-variance bounds for a
# fixed mean. Work in original units to retain small deviations on wide supports.
function _meanvarminmax_quantile_lower_at_mean(p, q, m, vl, vh, a, b)
    width = b - a
    left, right = m - a, b - m
    vmax = left * right
    iszero(vmax) && return m
    lower = max(a, m - sqrt(min(vh, vmax)) * (sqrt(q) / sqrt(p)), m - (right / p) * q)
    above_endpoint_mass = p <= q ? p > right / width : q < left / width
    if above_endpoint_mass
        gap = max(0, vmax - vl)
        iszero(gap) && return b
        denominator = p <= q ? muladd(p, width, -right) : muladd(-q, width, left)
        if denominator > 0
            central = m + muladd(-q * width, right, vl) / denominator
            lower = max(lower, central)
        end
    end
    return clamp(lower, a, b)
end

# A variance interval is redundant when it contains all variances permitted by
# the checked mean and support. This also recognizes an omitted variance after
# check_moments_and_range has tightened its original [0, Inf] interval.
function _has_informative_variance(m, v, a, b)
    lo(v) > 0 && return true
    (lo(m) == hi(m) == a || lo(m) == hi(m) == b) && return false
    if isfinite(a) && isfinite(b)
        peak = clamp(a + (b - a) / 2, lo(m), hi(m))
        return hi(v) < (peak - a) * (b - peak)
    end
    return isfinite(hi(v))
end

function cdf(p::MomentProbabilityBox, x::Real)
    m, v = mean(p), var(p)
    a, b = bounds(p.range)
    if !isfinite(a) && !isfinite(b)
        return chebyshev_cdf(x, m, v)
    elseif !_has_informative_variance(m, v, a, b)
        return markov_cdf(x, m, a, b)
    elseif !isfinite(a) || !isfinite(b)
        return cantelli_cdf(x, m, v, a, b)
    end
    return meanvarminmax_cdf(x, m, v, a, b)
end

function quantile(p::MomentProbabilityBox, u::Real)
    m, v = mean(p), var(p)
    a, b = bounds(p.range)
    if !isfinite(a) && !isfinite(b)
        return chebyshev_quantile(u, m, v)
    elseif !_has_informative_variance(m, v, a, b)
        return markov_quantile(u, m, a, b)
    elseif !isfinite(a) || !isfinite(b)
        return cantelli_quantile(u, m, v, a, b)
    end
    return meanvarminmax_quantile(u, m, v, a, b)
end

function reverse_quantile(p::MomentProbabilityBox, x::Interval)
    # Atoms and constant quantile branches can correspond to a whole probability
    # interval; choose its midpoint when the original probability is not unique.
    lower = max(cdf(p, prevfloat(float(x.lb))).ub, cdf(p, prevfloat(float(x.ub))).lb)
    upper = min(cdf(p, x.lb).ub, cdf(p, x.ub).lb)
    lower <= upper + 1.0e-6 ||
        throw(ArgumentError("Interval is not a quantile slice of this moment p-box."))
    return (lower + upper) / 2
end
