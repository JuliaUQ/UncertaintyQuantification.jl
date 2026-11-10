@testitem "Moment p-box: constructors and feasibility" begin
    p = ProbabilityBox(; mean = 0, var = 1)
    @test isabstracttype(MomentConstraints)
    @test isconcretetype(typeof(p))
    @test p isa MomentProbabilityBox
    @test p isa AbstractProbabilityBox
    @test p isa AbstractProbabilityBox{MomentConstraints}
    @test isabstracttype(AbstractProbabilityBox)
    @test !(p isa ProbabilityBox)
    @test MomentProbabilityBox <: AbstractProbabilityBox{MomentConstraints}
    @test AbstractProbabilityBox{MomentConstraints} <: AbstractProbabilityBox
    @test p.mean === mean(p)
    @test p.var === var(p)
    @test p.mean isa Interval
    @test p.var isa Interval
    @test p.range isa Interval
    @test !hasproperty(p, :parameters)
    @test MomentProbabilityBox(; mean = 0, var = 1) isa AbstractProbabilityBox{MomentConstraints}
    normal = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
    @test normal isa ProbabilityBox
    @test normal isa ProbabilityBox{Normal}
    @test isconcretetype(typeof(normal))
    @test normal isa AbstractProbabilityBox{Normal}
    @test eltype([p, normal]) <: AbstractProbabilityBox
    @test all(x -> x isa Interval, cdf.([p, normal], 0))
    @test UncertaintyQuantification.bounds(mean(p)) == (0, 0)
    @test UncertaintyQuantification.bounds(var(p)) == (1, 1)
    @test minimum(p) == -Inf
    @test maximum(p) == Inf
    @test length(p) == 1

    for p in (ProbabilityBox(; mean = 2, lb = 0), ProbabilityBox(; mean = 2, ub = 5))
        @test UncertaintyQuantification.bounds(var(p)) == (0, Inf)
    end
    p = ProbabilityBox(; mean = 2, range = Interval(0, 5))
    @test UncertaintyQuantification.bounds(var(p)) == (0, 6)
    @test UncertaintyQuantification.bounds(p.range) == (0, 5)
    p = ProbabilityBox(; mean = Interval(-1, 3), var = Interval(0, 10), lb = 0, ub = 2)
    @test UncertaintyQuantification.bounds(mean(p)) == (0, 2)
    @test UncertaintyQuantification.bounds(var(p)) == (0, 1)
    p = ProbabilityBox(; mean = Interval(0, 1), var = 0.24, lb = 0, ub = 1)
    @test mean(p).lb ≈ 0.4
    @test mean(p).ub ≈ 0.6

    @test_throws ArgumentError ProbabilityBox(; mean = 0)
    @test_throws ArgumentError ProbabilityBox(; mean = Inf, var = 1)
    @test_throws ArgumentError("Mean bounds must be finite after intersecting the support.") ProbabilityBox(; mean = Interval(0, Inf), var = 1)
    @test_throws ArgumentError("Variance must have a finite lower bound.") ProbabilityBox(; mean = 0, var = Inf)
    @test_throws ArgumentError ProbabilityBox(; mean = NaN, var = 1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, var = -1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, var = NaN)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, var = 1, lb = NaN)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, var = 1, lb = 2, ub = 1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, lb = 1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, ub = -1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, var = 1, lb = 0)
    @test_throws ArgumentError ProbabilityBox(; mean = 0, var = 1, ub = 0)
    @test_throws ArgumentError ProbabilityBox(; mean = 0.1, var = 0.2, lb = 0, ub = 1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0.5, var = 0.3, lb = 0, ub = 1)
    @test_throws ArgumentError ProbabilityBox(; mean = 0.5, range = Interval(0, 1), lb = 0)
    @test_throws ArgumentError quantile(p, -0.1)
    @test_throws ArgumentError quantile(p, 1.1)
    @test_throws ArgumentError quantile(p, NaN)
    @test_throws ArgumentError cdf(p, NaN)
end

@testitem "Moment p-box: inconsistency diagnostics" begin
    using UncertaintyQuantification: check_moments_and_range

    @test_throws [
        "Mean ∩ range = ∅.", "Provided mean: [3, 4]", "Theoretical mean bounds: [0, 2]",
    ] check_moments_and_range(; mean = Interval(3, 4), lb = 0, ub = 2)
    @test_throws [
        "Provided variance: [10, 12]", "Theoretical variance bounds: [0, 9]",
        "Mean used for variance bounds: [11, 11]", "range: [10, 20]",
    ] check_moments_and_range(; mean = 11, var = Interval(10, 12), lb = 10, ub = 20)
    @test_throws [
        "Provided variance: 10", "Theoretical variance bounds: [0, 9]",
    ] check_moments_and_range(; mean = 11, var = 10, lb = 10, ub = 20)
    @test_throws [
        "Provided variance: [-2, -1]", "Theoretical variance bounds: [0, Inf]",
    ] check_moments_and_range(; mean = 0, var = Interval(-2, -1))
    @test_throws [
        "Provided lb=2, ub=1",
    ] check_moments_and_range(; mean = 0, var = 1, lb = 2, ub = 1)
end

@testitem "Moment p-box: named bound evaluations" begin
    using UncertaintyQuantification:
        chebyshev_cdf, chebyshev_quantile, cantelli_cdf, cantelli_quantile,
        markov_cdf, markov_quantile,
        meanvarminmax_cdf, meanvarminmax_quantile

    @test UncertaintyQuantification.bounds(chebyshev_cdf(-1, 0, 1)) == (0, 0.5)
    @test UncertaintyQuantification.bounds(chebyshev_quantile(0.5, 0, 1)) == (-1, 1)
    @test UncertaintyQuantification.bounds(markov_cdf(4, 2, 0, Inf)) == (0.5, 1)
    @test UncertaintyQuantification.bounds(markov_quantile(0.75, 2, 0, Inf)) == (0, 8)
    @test markov_cdf(1, 2, 0, 5).ub ≈ 0.75
    @test markov_quantile(0.9, 2, 0, 5).lb ≈ 5 - 3 / 0.9
    @test cantelli_cdf(1, 0, 3, -1, Inf).lb ≈ 0.5
    @test cantelli_quantile(0.5, 0, 3, -1, Inf).ub ≈ 1
    @test cantelli_cdf(-1, 0, 3, -Inf, 1).ub ≈ 0.5
    @test cantelli_quantile(0.5, 0, 3, -Inf, 1).lb ≈ -1
    @test meanvarminmax_cdf(0.5, 0.5, 0.1, 0, 1).lb ≈ 0.2
    @test meanvarminmax_cdf(0.5, 0.5, 0.1, 0, 1).ub ≈ 0.8
    q = meanvarminmax_quantile(0.5, 0.5, 0.1, 0, 1)
    @test q.lb ≈ 0.5 - sqrt(0.1)
    @test q.ub ≈ 0.5 + sqrt(0.1)

    # At endpoint probabilities a variance lower bound adds information
    # beyond the intersection of the separate Chebyshev and Markov inverses.
    @test UncertaintyQuantification.bounds(meanvarminmax_quantile(1, 0, 3, -1, Inf)) == (3, Inf)
    @test UncertaintyQuantification.bounds(meanvarminmax_quantile(0, 0, 1, -Inf, 2)) == (-Inf, -0.5)
    @test UncertaintyQuantification.bounds(cantelli_quantile(1, 0, 3, -1, Inf)) == (3, Inf)
    @test UncertaintyQuantification.bounds(cantelli_quantile(0, 0, 1, -Inf, 2)) == (-Inf, -0.5)
    @test UncertaintyQuantification.bounds(markov_cdf(0, 0, 0, Inf)) == (1, 1)
    @test UncertaintyQuantification.bounds(markov_quantile(0, 0, 0, Inf)) == (0, 0)
    @test UncertaintyQuantification.bounds(meanvarminmax_cdf(0, 0.25, 0.1875, 0, 1)) == (0.75, 0.75)
    @test UncertaintyQuantification.bounds(meanvarminmax_quantile(0.75, 0.25, 0.1875, 0, 1)) == (0, 0)
end

@testitem "Moment p-box: variance and range" begin
    p = ProbabilityBox(; var = 0.16, range = Interval(0, 1))
    @test mean(p).lb ≈ 0.2
    @test mean(p).ub ≈ 0.8
    @test UncertaintyQuantification.bounds(var(p)) == (0.16, 0.16)
    @test cdf(p, 0.5).lb ≈ 0.195
    @test cdf(p, 0.5).ub ≈ 0.805

    # Maximum variance fixes the previously unknown mean and the endpoint law.
    p = ProbabilityBox(; var = 0.25, range = Interval(0, 1))
    @test UncertaintyQuantification.bounds(mean(p)) == (0.5, 0.5)
    @test UncertaintyQuantification.bounds(cdf(p, 0)) == (0.5, 0.5)
    @test UncertaintyQuantification.bounds(cdf(p, 0.5)) == (0.5, 0.5)
    p = ProbabilityBox(; var = Interval(0.16, 1), range = Interval(0, 1))
    @test UncertaintyQuantification.bounds(var(p)) == (0.16, 0.25)
    @test mean(p).lb ≈ 0.2
    @test mean(p).ub ≈ 0.8

    # No positive variance lower bound gives the vacuous range envelope:
    # point masses anywhere within the support remain feasible.
    for p in (
            ProbabilityBox(; range = Interval(0, 1)),
            ProbabilityBox(; var = 0, range = Interval(0, 1)),
            ProbabilityBox(; var = Interval(0, 0.1), range = Interval(0, 1)),
        )
        @test UncertaintyQuantification.bounds(cdf(p, 0.5)) == (0, 1)
        @test UncertaintyQuantification.bounds(quantile(p, 0.5)) == (0, 1)
    end
    @test_throws ArgumentError ProbabilityBox(; var = 0.3, range = Interval(0, 1))
    @test_throws ArgumentError ProbabilityBox(; var = Interval(0.3, 1), range = Interval(0, 1))
    @test_throws ArgumentError ProbabilityBox(; var = 1)
    @test_throws ArgumentError ProbabilityBox(; var = 1, lb = 0)
end

@testitem "Moment p-box: intersecting moment constraints" begin
    p = ProbabilityBox(; mean = 0.5, range = Interval(0, 2))
    @test UncertaintyQuantification.bounds(var(p)) == (0, 0.75)
    p = ProbabilityBox(; mean = Interval(1, 4), range = Interval(0, 2))
    @test UncertaintyQuantification.bounds(mean(p)) == (1, 2)
    p = ProbabilityBox(; mean = 0.5, var = Interval(0.5, 10), range = Interval(0, 2))
    @test UncertaintyQuantification.bounds(var(p)) == (0.5, 0.75)
    p = ProbabilityBox(; mean = Interval(10, 12), range = Interval(10, 12))
    @test UncertaintyQuantification.bounds(var(p)) == (0, 1)
    p = ProbabilityBox(; mean = Interval(-Inf, Inf), var = Interval(-1, 0.5), range = Interval(0, 2))
    @test UncertaintyQuantification.bounds(mean(p)) == (0, 2)
    @test UncertaintyQuantification.bounds(var(p)) == (0, 0.5)
    @test_throws ArgumentError ProbabilityBox(; mean = 0.5, var = Interval(-2, -1), range = Interval(0, 2))
    @test_throws ArgumentError ProbabilityBox(; mean = Interval(0, 0.1), var = Interval(0.2, 0.3), range = Interval(0, 1))

    # A feasible endpoint law must not be rejected because of rounding in
    # the roots used to project variance information onto the mean interval.
    using Random
    rng = MersenneTwister(273)
    for _ in 1:1000
        a = 10rand(rng) - 5
        b = a + 10rand(rng)
        m = a + (b - a) * rand(rng)
        v = (m - a) * (b - m)
        local p = ProbabilityBox(; mean = m, var = v, range = Interval(a, b))
        @test UncertaintyQuantification.bounds(mean(p)) == (m, m)
        @test UncertaintyQuantification.bounds(var(p)) == (v, v)
        @test cdf(p, a).lb ≈ (b - m) / (b - a)
        # The same mean remains feasible when supplied in a wider interval.
        p = ProbabilityBox(; mean = Interval(m, b), var = v, range = Interval(a, b))
        @test mean(p).lb <= m <= mean(p).ub
    end
end

@testitem "Moment p-box: theoretical moment bounds" begin
    using UncertaintyQuantification: check_moments_and_range

    # The midpoint is 15 on [10,20], including when the mean is omitted.
    m, v, a, b = check_moments_and_range(; lb = 10, ub = 20)
    @test UncertaintyQuantification.bounds(m) == (10, 20)
    @test UncertaintyQuantification.bounds(v) == (0, 25)
    @test (a, b) == (10, 20)
    _, v, _, _ = check_moments_and_range(; mean = 11, lb = 10, ub = 20)
    @test UncertaintyQuantification.bounds(v) == (0, 9)
    _, v, _, _ = check_moments_and_range(; mean = Interval(10, 12), lb = 10, ub = 20)
    @test UncertaintyQuantification.bounds(v) == (0, 16)
    m, v, _, _ = check_moments_and_range(; var = 16, lb = 10, ub = 20)
    @test UncertaintyQuantification.bounds(m) == (12, 18)
    @test UncertaintyQuantification.bounds(v) == (16, 16)
    m, v, _, _ = check_moments_and_range(; mean = Interval(1, 5), var = 16, lb = 0, ub = 10)
    @test UncertaintyQuantification.bounds(m) == (2, 5)
    @test UncertaintyQuantification.bounds(v) == (16, 16)
    m, _, _, _ = check_moments_and_range(; var = 1.0e-16, lb = 0, ub = 1)
    @test m.lb ≈ 1.0e-16 rtol = 1.0e-12
    @test_throws ArgumentError check_moments_and_range(; mean = Interval(), lb = 0, ub = 1)
    @test_throws ArgumentError check_moments_and_range(; mean = 0.5, var = Interval(), lb = 0, ub = 1)
    @test_throws ArgumentError check_moments_and_range(; mean = Interval(0, 1), lb = 10, ub = 20)
end

@testitem "Moment p-box: classical bounds" begin
    p = ProbabilityBox(; mean = 0, var = 1)
    @test UncertaintyQuantification.bounds(cdf(p, -1)) == (0, 0.5)
    @test UncertaintyQuantification.bounds(cdf(p, 0)) == (0, 1)
    @test UncertaintyQuantification.bounds(cdf(p, 1)) == (0.5, 1)
    @test UncertaintyQuantification.bounds(cdf(p, -Inf)) == (0, 0)
    @test UncertaintyQuantification.bounds(cdf(p, Inf)) == (1, 1)
    for u in (0.01, 0.25, 0.5, 0.75, 0.99)
        q = quantile(p, u)
        @test q.lb ≈ -sqrt((1 - u) / u) atol = 1.0e-12
        @test q.ub ≈ sqrt(u / (1 - u)) atol = 1.0e-12
    end
    @test UncertaintyQuantification.bounds(quantile(p, 0)) == (-Inf, 0)
    @test UncertaintyQuantification.bounds(quantile(p, 1)) == (0, Inf)

    p = ProbabilityBox(; mean = 2, lb = 0)
    @test UncertaintyQuantification.bounds(cdf(p, 1)) == (0, 1)
    @test UncertaintyQuantification.bounds(cdf(p, 4)) == (0.5, 1)
    @test quantile(p, 0.75).lb ≈ 0 atol = 1.0e-12
    @test quantile(p, 0.75).ub ≈ 8
    p = ProbabilityBox(; mean = 2, ub = 5)
    @test UncertaintyQuantification.bounds(cdf(p, -1)) == (0, 0.5)
    @test quantile(p, 0.25).lb ≈ -7
    @test quantile(p, 0.25).ub ≈ 5

    p = ProbabilityBox(; mean = 2, range = Interval(0, 5))
    @test cdf(p, 1).ub ≈ 0.75
    @test cdf(p, 4).lb ≈ 0.5
    @test quantile(p, 0.9).lb ≈ 5 - 3 / 0.9
    @test quantile(p, 0.1).ub ≈ 2 / 0.9

    # Adding a one-sided support improves the Chebyshev moment envelope.
    p = ProbabilityBox(; mean = 0, var = 3, lb = -1)
    @test cdf(p, 1).lb ≈ 0.5
    @test quantile(p, 0.5).ub ≈ 1
    for u in (0.1, 0.5, 0.9)
        @test quantile(p, u).lb ≈ max(-1, -sqrt(3 * (1 - u) / u)) atol = 1.0e-12
        @test quantile(p, u).ub ≈ min(-1 + 1 / (1 - u), sqrt(3 * u / (1 - u))) atol = 1.0e-12
    end
end

@testitem "Moment p-box: bounded moments and interval extrema" begin
    p = ProbabilityBox(; mean = 0.5, var = 0.1, lb = 0, ub = 1)
    @test cdf(p, 0.1).ub ≈ 0.1 / (0.1 + 0.4^2)
    @test cdf(p, 0.5).lb ≈ 0.2
    @test cdf(p, 0.5).ub ≈ 0.8
    @test cdf(p, 0.9).lb ≈ 0.4^2 / (0.1 + 0.4^2)
    @test UncertaintyQuantification.bounds(quantile(p, 0)) == (0, 0.3)
    @test UncertaintyQuantification.bounds(quantile(p, 1)) == (0.7, 1)

    # Upper-CDF optimum at variance 0.15, inside [0.01, 0.2].
    p = ProbabilityBox(; mean = 0.5, var = Interval(0.01, 0.2), lb = 0, ub = 1)
    @test cdf(p, 0.2).ub ≈ 0.625
    @test cdf(p, 0.8).lb ≈ 0.375
    @test quantile(p, 0.625).lb ≈ 0.2
    @test quantile(p, 0.375).ub ≈ 0.8
    # Upper-CDF optimum at mean 0.4, inside [0.3, 0.7]. A witness is
    # P(X=0)=0.55, P(X=0.8)=0.25, P(X=1)=0.2, with variance 0.2.
    p = ProbabilityBox(; mean = Interval(0.3, 0.7), var = Interval(0.2, 0.21), lb = 0, ub = 1)
    @test cdf(p, 0.8).ub ≈ 0.8
    @test cdf(p, 0.2).lb ≈ 0.2
    @test quantile(p, 0.8).lb ≈ 0.8
    @test quantile(p, 0.2).ub ≈ 0.2

    p = ProbabilityBox(; mean = Interval(0, 1), var = Interval(1, 4))
    @test cdf(p, -2).ub ≈ 0.5
    @test cdf(p, 3).lb ≈ 0.5
    @test quantile(p, 0.25).lb ≈ -2sqrt(3)
    @test quantile(p, 0.25).ub ≈ 1 + 2 / sqrt(3)
end

@testitem "Moment p-box: direct bounded quantiles" begin
    using Random
    rng = MersenneTwister(273)

    # Independent CDF inversion is a test oracle, not an evaluation path.
    function inverse_cdf(pbox, u, endpoint)
        left, right = UncertaintyQuantification.bounds(pbox.range)
        endpoint(cdf(pbox, left)) >= u && return left
        for _ in 1:80
            mid = left + (right - left) / 2
            if endpoint(cdf(pbox, mid)) >= u
                right = mid
            else
                left = mid
            end
        end
        return right
    end

    for i in 1:100
        a = 10rand(rng) - 5
        b = a + 10rand(rng) + 0.01
        m = a + (b - a) * rand(rng)
        vmax = (m - a) * (b - m)
        vl = vmax * rand(rng)
        vh = vl + (vmax - vl) * rand(rng)
        means = iseven(i) ? Interval(a + (m - a) * rand(rng), m + (b - m) * rand(rng)) : m
        p = MomentProbabilityBox(; mean = means, var = Interval(vl, vh), lb = a, ub = b)
        for u in (0.01, 0.2, 0.5, 0.8, 0.99)
            q = quantile(p, u)
            @test q.lb ≈ inverse_cdf(p, u, UncertaintyQuantification.hi) atol = 1.0e-11 * (b - a)
            @test q.ub ≈ inverse_cdf(p, u, UncertaintyQuantification.lo) atol = 1.0e-11 * (b - a)
        end
    end

    # Wide support must not erase mean/variance tail bounds through normalization.
    p = MomentProbabilityBox(; mean = 0, var = 1, range = Interval(-1.0e150, 1.0e150))
    @test quantile(p, 0.5).lb ≈ -1
    @test quantile(p, 0.5).ub ≈ 1
    @test quantile(p, 0.1).lb ≈ -3
    @test quantile(p, 0.1).ub ≈ 1 / 3

    # Keep both complementary probabilities when a tail is below machine epsilon.
    p = MomentProbabilityBox(; mean = 0, var = 1.0e-320, range = Interval(-1, 1))
    tiny_tail = quantile(p, 1.0e-320)
    @test tiny_tail.lb ≈ -1
    @test tiny_tail.ub ≈ 1.0e-320 rtol = 1.0e-3
    for u in (nextfloat(0.0), prevfloat(1.0))
        q = quantile(p, u)
        @test -1 <= q.lb <= q.ub <= 1
    end
end

@testitem "Moment p-box: degenerate distributions and atoms" begin
    p = ProbabilityBox(; mean = 2, var = 0)
    @test UncertaintyQuantification.bounds(cdf(p, 1.9)) == (0, 0)
    @test UncertaintyQuantification.bounds(cdf(p, 2)) == (1, 1)
    @test minimum(p) == maximum(p) == 2
    @test !insupport(p, 1)
    for u in (0, 0.5, 1)
        @test UncertaintyQuantification.bounds(quantile(p, u)) == (2, 2)
    end
    p = ProbabilityBox(; mean = Interval(1, 2), var = 0)
    @test UncertaintyQuantification.bounds(cdf(p, 1)) == (0, 1)
    @test UncertaintyQuantification.bounds(cdf(p, 2)) == (1, 1)
    @test UncertaintyQuantification.bounds(quantile(p, 0)) == (1, 2)
    @test UncertaintyQuantification.bounds(quantile(p, 1)) == (1, 2)
    for p in (ProbabilityBox(; mean = 0, lb = 0), ProbabilityBox(; mean = 1, lb = 1, ub = 1))
        @test iszero(var(p).ub)
        @test UncertaintyQuantification.bounds(cdf(p, mean(p).lb)) == (1, 1)
    end

    # Maximum variance forces an endpoint Bernoulli law. F(lb) includes its atom.
    p = ProbabilityBox(; mean = 0.25, var = 0.1875, lb = 0, ub = 1)
    @test UncertaintyQuantification.bounds(cdf(p, -0.1)) == (0, 0)
    @test UncertaintyQuantification.bounds(cdf(p, 0)) == (0.75, 0.75)
    @test UncertaintyQuantification.bounds(cdf(p, 0.9)) == (0.75, 0.75)
    @test UncertaintyQuantification.bounds(cdf(p, 1)) == (1, 1)
    @test UncertaintyQuantification.bounds(quantile(p, 0.75)) == (0, 0)
    @test UncertaintyQuantification.bounds(quantile(p, 0.8)) == (1, 1)
end

@testitem "Moment p-box: discrete distribution containment" begin
    using Random
    rng = MersenneTwister(273)
    for _ in 1:30
        xs = sort!(rand(rng, 5))
        weights = rand(rng, 5)
        weights ./= sum(weights)
        m = sum(weights .* xs)
        v = sum(weights .* (xs .- m) .^ 2)
        for p in (
                ProbabilityBox(; mean = m, var = v),
                ProbabilityBox(; mean = m, lb = 0),
                ProbabilityBox(; mean = m, ub = 1),
                ProbabilityBox(; mean = m, lb = 0, ub = 1),
                ProbabilityBox(; mean = m, var = v, lb = 0, ub = 1),
                ProbabilityBox(; var = v, range = Interval(0, 1)),
                ProbabilityBox(; var = Interval(v / 2, 2v), range = Interval(0, 1)),
                ProbabilityBox(; mean = Interval(m - 0.05, m + 0.05), var = Interval(v / 2, 2v), lb = 0, ub = 1),
            )
            for x in (xs..., 0.0, 0.25, 0.5, 0.75, 1.0)
                f = sum(weights[xs .<= x])
                b = cdf(p, x)
                @test b.lb - 1.0e-12 <= f <= b.ub + 1.0e-12
            end
            for u in (0.1, 0.4, 0.9)
                q = xs[findfirst(>=(u), cumsum(weights))]
                b = quantile(p, u)
                @test b.lb - 1.0e-12 <= q <= b.ub + 1.0e-12
            end
        end
    end
end

@testitem "Moment p-box: sampling and random slicing" begin
    using Copulas
    using DataFrames
    using Random
    p = ProbabilityBox(; mean = 0.5, var = 0.1, lb = 0, ub = 1)
    rv = RandomVariable(p, :x)
    @test isimprecise(rv)
    @test isimprecise([rv])
    @test size(sample(rv, 10)) == (10, 1)
    @test all(x -> x isa Interval, rand(p, 10))
    @test length(cdf.(p, [0, 0.5, 1])) == 3
    u = [0.1, 0.3, 0.5, 0.7, 0.9]
    slices = quantile.(p, u)
    @test UncertaintyQuantification.reverse_quantile.(p, slices) ≈ u
    sns = DataFrame(; x = quantile.(Normal(), u))
    original = copy(sns.x)
    to_physical_space!(rv, sns)
    @test all(x -> x isa Interval, sns.x)
    to_standard_normal_space!(rv, sns)
    @test sns.x ≈ original
    @test_throws ArgumentError UncertaintyQuantification.bounds(rv)
    @test_throws ArgumentError UncertaintyQuantification.map_to_precise_inputs(Float64[], [rv])

    jd = JointDistribution(GaussianCopula([1.0 0.0; 0.0 1.0]), [rv, RandomVariable(Normal(), :z)])
    @test isimprecise(jd)
    joint_sns = DataFrame(; x = original, z = zeros(length(original)))
    to_physical_space!(jd, joint_sns)
    to_standard_normal_space!(jd, joint_sns)
    @test joint_sns.x ≈ original
    @test joint_sns.z ≈ zeros(length(original)) atol = 1.0e-12

    Random.seed!(273)
    model = Model(df -> df.x, :y)
    @test_throws ArgumentError probability_of_failure(model, df -> df.y .- 0.5, [rv], DoubleLoop(MonteCarlo(10)))
    pf, _, _ = probability_of_failure(model, df -> df.y .- 0.5, [rv], RandomSlicing(MonteCarlo(2000)))
    @test pf.lb ≈ 0.2 atol = 0.04
    @test pf.ub ≈ 0.8 atol = 0.04
end
