@testitem "Distribution-free p-box: from CDF function" begin
    lower_cdf = x -> cdf(Normal(1, 1), x)
    upper_cdf = x -> cdf(Normal(0, 1), x)
    pbox = DistributionFreeProbabilityBox(lower_cdf, upper_cdf, -Inf, +Inf)

    # the cdfs are stored as given
    @test pbox.lower_cdf === lower_cdf
    @test pbox.upper_cdf === upper_cdf

    # cdf: from the lower cdf to the upper cdf
    c = cdf(pbox, 0.5)
    @test c.lb == cdf(Normal(1, 1), 0.5)
    @test c.ub == cdf(Normal(0, 1), 0.5)

    # quantile: from the quantile of the upper cdf to the quantile of the lower cdf, also in the tails. Not closer to 1 than 1 - 1e-9: near 1 the cdf only changes in steps of eps(1.0) ≈ 1e-16, so its inverse is less precise there
    for u in [1.0e-12, 0.1, 0.5, 0.9, 1 - 1.0e-9]
        q = quantile(pbox, u)
        @test q.lb ≈ quantile(Normal(0, 1), u)
        @test q.ub ≈ quantile(Normal(1, 1), u)
    end

    # rand: n quantile intervals
    samples = rand(pbox, 100)
    @test length(samples) == 100
    @test all(s -> s.lb <= s.ub, samples)

    # finite support
    pbox = DistributionFreeProbabilityBox(x -> cdf(Uniform(1, 3), x), x -> cdf(Uniform(0, 2), x), 0, 3)
    q = quantile(pbox, 0.5)
    @test q.lb ≈ 1.0
    @test q.ub ≈ 2.0
    @test quantile(pbox, 0).lb == 0
    @test quantile(pbox, 1).ub == 3

    # printed like ProbabilityBox: the interval of the median, then the ends of the support
    pbox = DistributionFreeProbabilityBox(
        x -> cdf(Uniform(1, 3), x),
        x -> cdf(Uniform(0, 2), x),
        0,
        3
    )
    @test sprint(show, pbox) == "DistributionFreeProbabilityBox(median = [1.0, 2.0], 0, 3)"
end

@testitem "Distribution-free p-box: invalid cdfs" begin
    N0 = x -> cdf(Normal(0, 1), x)
    N1 = x -> cdf(Normal(1, 1), x)
    U02 = x -> cdf(Uniform(0, 2), x)
    U13 = x -> cdf(Uniform(1, 3), x)

    # the support must have lb < ub
    @test_throws ArgumentError("the support must have lb < ub, got [1, 0]") DistributionFreeProbabilityBox(N1, N0, 1, 0)

    # at a finite end of the support the cdfs must be exactly 0 (lb) and 1 (ub)
    @test_throws ["lower_cdf must be 0.0 at x = -10"] DistributionFreeProbabilityBox(N1, N0, -10, Inf)
    @test_throws ["lower_cdf must be 1.0 at x = 2.5"] DistributionFreeProbabilityBox(U13, U02, 0, 2.5)
    @test_throws ["upper_cdf must be 0.0 at x = 0.5"] DistributionFreeProbabilityBox(U13, U02, 0.5, 3)

    # at an infinite end too, when the cdf can be evaluated there: even a tail of 5e-9 is not 0
    @test_throws ["lower_cdf must be 0.0 at x = -Inf"] DistributionFreeProbabilityBox(
        x -> N0(x) + (x < -10 ? 5.0e-9 : 0.0),
        N0,
        -Inf,
        Inf
    )

    # at an infinite end the cdf must go to 0 (left) and 1 (right)
    @test_throws ["lower_cdf never reaches 1"] DistributionFreeProbabilityBox(
        x -> 0.5 * N1(x),
        N0,
        -Inf,
        Inf
    )
    @test_throws ["upper_cdf never goes to 0"] DistributionFreeProbabilityBox(
        N1,
        x -> 0.1 + 0.9 * N0(x),
        -Inf,
        Inf
    )

    # values must be in [0, 1]: 0 at lb = 0 and 1 at ub = 2, but rising to 1.5 in between
    above_one = x -> x < 2 ? min(1.5 * x, 1.5) : 1.0
    @test_throws ["lower_cdf must be in [0, 1]"] DistributionFreeProbabilityBox(above_one, U02, 0, 2)

    # the cdf must be non-decreasing: a dip of 0.1 on (1.4, 1.6)
    @test_throws ["lower_cdf must be non-decreasing"] DistributionFreeProbabilityBox(
        x -> U02(x) - (1.4 < x < 1.6 ? 0.1 : 0.0),
        U02,
        0,
        2
    )
end

@testitem "Distribution-free p-box: order of the bounds" begin
    N0 = x -> cdf(Normal(0, 1), x)
    N1 = x -> cdf(Normal(1, 1), x)
    U01 = x -> cdf(Uniform(0, 1), x)
    U56 = x -> cdf(Uniform(5, 6), x)

    # accepted: the lower cdf is never above the upper one, also when they are equal
    @test DistributionFreeProbabilityBox(N1, N0, -Inf, Inf) isa DistributionFreeProbabilityBox
    @test DistributionFreeProbabilityBox(N0, N0, -Inf, Inf) isa DistributionFreeProbabilityBox
    @test DistributionFreeProbabilityBox(U56, U01, 0, 6) isa DistributionFreeProbabilityBox

    # accepted: bounds that are equal on part of the support (same first component)
    mixture_lower = x -> cdf(MixtureModel([Normal(0, 1), Normal(10, 1)]), x)
    mixture_upper = x -> cdf(MixtureModel([Normal(0, 1), Normal(5, 1)]), x)
    @test DistributionFreeProbabilityBox(mixture_lower, mixture_upper, -Inf, Inf) isa DistributionFreeProbabilityBox

    # rejected: the bounds swapped, on an infinite and on a finite support
    @test_throws ["lower_cdf must be ≤ upper_cdf"] DistributionFreeProbabilityBox(N0, N1, -Inf, Inf)
    @test_throws ["lower_cdf must be ≤ upper_cdf"] DistributionFreeProbabilityBox(U01, U56, 0, 6)

    # rejected: bounds that cross, Normal(0, 2) is above Normal(0, 1) on the left and below it on the right
    @test_throws ["lower_cdf must be ≤ upper_cdf"] DistributionFreeProbabilityBox(x -> cdf(Normal(0, 2), x), N0, -Inf, Inf)

    # rejected: no tolerance, Normal(0, 1) shifted left by 1e-11 is a valid cdf above Normal(0, 1) by only ~4e-12
    @test_throws ["lower_cdf must be ≤ upper_cdf"] DistributionFreeProbabilityBox(x -> N0(x + 1.0e-11), N0, -Inf, Inf)
end

@testitem "Distribution-free p-box: from EmpiricalDistributions" begin
    data = quantile.(Normal(), range(0.005, 0.995, 200))
    # the same data shifted right by 1: same bandwidth, so the lower cdf is below everywhere
    lower = EmpiricalDistribution(data .+ 1)
    upper = EmpiricalDistribution(data)
    pbox = DistributionFreeProbabilityBox(lower, upper)

    # cdf and quantile are the ones of the empirical distributions
    for x in [-2.0, 0.0, 0.5, 2.0]
        c = cdf(pbox, x)
        @test c.lb == cdf(lower, x)
        @test c.ub == cdf(upper, x)
    end
    for u in [0.0, 0.1, 0.5, 0.9, 1.0]
        q = quantile(pbox, u)
        @test q.lb == quantile(upper, u)
        @test q.ub == quantile(lower, u)
    end

    # the stored cdfs are clamped to [0, 1]: the spline evaluation can round slightly above 1
    x = range(minimum(upper) - 1, maximum(lower) + 1, 10^5)
    @test all(0 .<= pbox.lower_cdf.(x) .<= 1)
    @test all(0 .<= pbox.upper_cdf.(x) .<= 1)

    samples = rand(pbox, 100)
    @test all(s -> s.lb <= s.ub, samples)

    # the same distribution as both bounds is accepted
    @test DistributionFreeProbabilityBox(upper, upper) isa DistributionFreeProbabilityBox

    # rejected: the bounds swapped, and bounds that cross (twice as wide: above on the left, below on the right)
    @test_throws ["lower_cdf must be ≤ upper_cdf"] DistributionFreeProbabilityBox(upper, lower)
    @test_throws ["lower_cdf must be ≤ upper_cdf"] DistributionFreeProbabilityBox(EmpiricalDistribution(2 .* data), upper)
end

@testitem "Distribution-free p-box: exact quantile" begin
    # heavy tails: the quantile is found by inverting the cdf exactly, so it is accurate also where the evenly spaced check grid is coarse
    pbox = DistributionFreeProbabilityBox(
        x -> cdf(Cauchy(1, 1), x),
        x -> cdf(Cauchy(0, 1), x),
        -Inf,
        Inf
    )
    for u in [0.25, 0.5, 0.75, 0.9]
        q = quantile(pbox, u)
        @test q.lb ≈ quantile(Cauchy(0, 1), u) atol = 1.0e-12
        @test q.ub ≈ quantile(Cauchy(1, 1), u) atol = 1.0e-12
    end
    # far in the tails, checked against the cdf itself: quantile(Cauchy(), u) loses precision there
    for u in [1.0e-6, 1 - 1.0e-6]
        q = quantile(pbox, u)
        @test cdf(Cauchy(0, 1), q.lb) ≈ u
        @test cdf(Cauchy(1, 1), q.ub) ≈ u
    end

    # step cdfs: all the mass at 0 (upper) and at 1 (lower)
    pbox = DistributionFreeProbabilityBox(
        x -> x >= 1 ? 1.0 : 0.0,
        x -> x >= 0 ? 1.0 : 0.0,
        -1,
        2
    )
    q = quantile(pbox, 0.5)
    @test q.lb ≈ 0 atol = 1.0e-12
    @test q.ub ≈ 1 atol = 1.0e-12

    # a gap in the support: half of the mass on [0, 1] and half on [4, 5] (upper), the same shifted right by 1 (lower)
    gap = MixtureModel([Uniform(0, 1), Uniform(4, 5)])
    pbox = DistributionFreeProbabilityBox(
        x -> cdf(gap, x - 1),
        x -> cdf(gap, x),
        0,
        6
    )
    q = quantile(pbox, 0.25)
    @test q.lb ≈ 0.5
    @test q.ub ≈ 1.5
    q = quantile(pbox, 0.75)
    @test q.lb ≈ 4.5
    @test q.ub ≈ 5.5
    # at the level of the gap (u = 0.5) the cdf is flat, every point of the flat part is a quantile
    q = quantile(pbox, 0.5)
    @test 1 <= q.lb <= 4
    @test 2 <= q.ub <= 5
end
