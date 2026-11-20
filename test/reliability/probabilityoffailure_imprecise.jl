@testitem "Imprecise Probability of Failure: Double Loop: P-boxes only" begin
    X = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :X
    )
    Y = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :Y
    )

    pf, x_lb, x_ub = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
    )

    @test pf.lb ≈ cdf(Normal(2, sqrt(2)), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(-2, sqrt(8)), -9) atol = 1.0e-6
end

@testsnippet P_failure_imprecise begin
    X = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :X
    )
    Y = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :Y
    )
end

@testitem "Imprecise Probability of Failure: Double Loop: P-boxes with Copula: independent" setup = [
    TestSetup, P_failure_imprecise,
] begin
    c = GaussianCopula([1 0.0; 0.0 1])

    jd = JointDistribution(c, [X, Y])

    pf, x_lb, x_ub = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, jd, DoubleLoop(FORM())
    )

    @test pf.lb ≈ cdf(Normal(2, sqrt(2)), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(-2, sqrt(8)), -9) atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Double Loop: P-boxes with Copula: dependent" setup = [
    TestSetup, P_failure_imprecise,
] begin
    c = GaussianCopula([1 0.71; 0.71 1])

    jd = JointDistribution(c, [X, Y])

    pf, x_lb, x_ub = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, jd, DoubleLoop(FORM())
    )

    @test pf.lb ≈ cdf(Normal(2, sqrt(1^2 + 1^2 + 2 * (0.71 * 1 * 1))), -9)
    @test pf.ub ≈ cdf(Normal(-2, sqrt(2^2 + 2^2 + 2 * (0.71 * 2 * 2))), -9)
end

@testitem "Imprecise Probability of Failure: Double Loop: Interval - Distribution" begin
    X = IntervalVariable(-1, 1, :X)
    Y = RandomVariable(Normal(0, 2), :Y)

    pf, x_lb, x_ub = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
    )

    @test pf.lb ≈ cdf(Normal(1, 2), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(-1, 2), -9) atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Double Loop: P-box - Distribution" begin
    X = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :X
    )
    Y = RandomVariable(Normal(0, 2), :Y)

    pf, x_lb, x_ub = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
    )

    @test pf.lb ≈ cdf(Normal(1, sqrt(5)), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(-1, sqrt(8)), -9) atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Double Loop: Interval - p-box" begin
    X = IntervalVariable(-1, 1, :X)
    Y = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(1, 2))), :Y)

    pf, x_lb, x_ub = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
    )

    @test pf.lb ≈ cdf(Normal(1, 1), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(-1, 2), -9) atol = 1.0e-6

    @test x_lb ≈ [1, 1]
    @test x_ub ≈ [-1, 2]
end

@testitem "Imprecise Probability of Failure: Random Slicing: P-boxes only" begin
    X = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :X
    )
    Y = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :Y
    )

    pf, _ = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
    )

    pbox_analyt = ProbabilityBox{Normal}(
        Dict(:μ => Interval(-2, 2), :σ => Interval(sqrt(2), sqrt(8)))
    )
    failure_analty = cdf(pbox_analyt, -9)

    @test pf.lb ≈ failure_analty.lb atol = 1.0e-6
    @test pf.ub ≈ failure_analty.ub atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Random Slicing: P-boxes with Copula" setup = [
    TestSetup,
] begin
    X = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :X
    )
    Y = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :Y
    )

    # independent so the solution is the same
    c = GaussianCopula([1 0.0; 0.0 1])

    jd = JointDistribution(c, [X, Y])

    pf, _ = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, jd, RandomSlicing(FORM())
    )

    pbox_analyt = ProbabilityBox{Normal}(
        Dict(:μ => Interval(-2, 2), :σ => Interval(sqrt(2), sqrt(8)))
    )
    failure_analty = cdf(pbox_analyt, -9)

    @test pf.lb ≈ failure_analty.lb atol = 1.0e-6
    @test pf.ub ≈ failure_analty.ub atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Random Slicing: Interval - Distribution" begin
    X = IntervalVariable(-1, 1, :X)
    Y = RandomVariable(Normal(0, 2), :Y)

    pf, _ = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
    )

    pbox_analyt = ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => 2))
    failure_analty = cdf(pbox_analyt, -9)

    @test pf.lb ≈ failure_analty.lb atol = 1.0e-6
    @test pf.ub ≈ failure_analty.ub atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Random Slicing: P-box - Distribution" begin
    X = RandomVariable(
        ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))), :X
    )
    Y = RandomVariable(Normal(0, 2), :Y)

    pf, _ = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
    )

    pbox_analyt = ProbabilityBox{Normal}(
        Dict(:μ => Interval(-1, 1), :σ => Interval(sqrt(5), sqrt(8)))
    )
    failure_analty = cdf(pbox_analyt, -9)

    @test pf.lb ≈ cdf(Normal(1, sqrt(5)), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(-1, sqrt(8)), -9) atol = 1.0e-6
end

@testitem "Imprecise Probability of Failure: Random Slicing: Interval - p-box" begin
    X = IntervalVariable(-1, 1, :X)
    Y = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(1, 2))), :Y)

    pf, _ = probability_of_failure(
        UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
    )

    pbox_analyt = ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2)))
    failure_analty = cdf(pbox_analyt, -9)

    @test pf.lb ≈ failure_analty.lb atol = 1.0e-6
    @test pf.ub ≈ failure_analty.ub atol = 1.0e-6
end

@testitem "IPM Reliability Double Loop" setup = [TestSetup, QMC] begin
    # y = x1 + x2 + ϵ, with bounded noise ϵ ∈ [-0.5, 0.5] of unknown distribution.
    # The IPM is fitted to noisy observations and should capture the noise as an
    # interval, so its pf bounds match those of the reference with ϵ as an interval variable.
    x1 = RandomVariable(Normal(), :x1)
    x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => 1)), :x2)
    ϵ = IntervalVariable(-0.5, 0.5, :ϵ)

    m = Model(df -> df.x1 .+ df.x2 .+ df.ϵ, :y)

    # Noisy training data; the noise is not an input of the IPM
    data = UncertaintyQuantification.sample(
        [x1, RandomVariable(Normal(), :x2), RandomVariable(Uniform(-0.5, 0.5), :ϵ)],
        QuasiMonteCarloSampling(200, HaltonSample()),
    )
    evaluate!(m, data)

    ipm = IntervalPredictorModel(data, :y, MonomialBasis(2, 1), [:x1, :x2])

    pf, x_lb, x_ub = probability_of_failure(
        ipm, df -> 3.0 .- df.y
        , [x1, x2], DoubleLoop(FORM())
    )

    # x1 + x2 ~ N(μ, √2) and failure is x1 + x2 + ϵ > 3, worst/best case at μ + ϵ = ±1.5
    pf_lb = cdf(Normal(), -4.5 / sqrt(2))
    pf_ub = cdf(Normal(), -1.5 / sqrt(2))

    @test pf.lb ≈ pf_lb atol = 1.0e-4
    @test pf.ub ≈ pf_ub atol = 1.0e-4

    @test x_lb[1] ≈ -1.0 atol = 1.0e-4
    @test x_ub[1] ≈ 1.0 atol = 1.0e-4

end

@testitem "IPM Reliability Random Slicing" setup = [TestSetup, QMC] begin
    # y = x1 + x2 + ϵ, with bounded noise ϵ ∈ [-0.5, 0.5] of unknown distribution.
    # The IPM is fitted to noisy observations and should capture the noise as an
    # interval, so its pf bounds match those of the reference with ϵ as an interval variable.
    x1 = RandomVariable(Normal(), :x1)
    x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => 1)), :x2)
    ϵ = IntervalVariable(-0.5, 0.5, :ϵ)

    m = Model(df -> df.x1 .+ df.x2 .+ df.ϵ, :y)

    # Noisy training data; the noise is not an input of the IPM
    data = UncertaintyQuantification.sample(
        [x1, RandomVariable(Normal(), :x2), RandomVariable(Uniform(-0.5, 0.5), :ϵ)],
        QuasiMonteCarloSampling(200, HaltonSample()),
    )
    evaluate!(m, data)

    ipm = IntervalPredictorModel(data, :y, MonomialBasis(2, 1), [:x1, :x2])

    pf, x_lb, x_ub = probability_of_failure(
        ipm, df -> 3.0 .- df.y
        , [x1, x2], RandomSlicing(FORM())
    )

    # x1 + x2 ~ N(μ, √2) and failure is x1 + x2 + ϵ > 3, worst/best case at μ + ϵ = ±1.5
    pf_lb = cdf(Normal(), -4.5 / sqrt(2))
    pf_ub = cdf(Normal(), -1.5 / sqrt(2))

    @test pf.lb ≈ pf_lb atol = 1.0e-4
    @test pf.ub ≈ pf_ub atol = 1.0e-4

end

@testitem "Imprecise Probability of Failure: Random Slicing: Distribution-free p-box" begin
    # the envelope of Normal(μ, 1), μ ∈ [0, 1]: lower cdf Normal(1, 1), upper cdf Normal(0, 1)
    X = RandomVariable(
        DistributionFreeProbabilityBox(
            x -> cdf(Normal(1, 1), x),
            x -> cdf(Normal(0, 1), x),
            -Inf,
            Inf
        ),
        :X,
    )
    Y = RandomVariable(Normal(), :Y)

    pf, _ = probability_of_failure(UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM()))

    # analytical: X + Y ~ Normal(μ, √2) with μ ∈ [0, 1]
    @test pf.lb ≈ cdf(Normal(1, sqrt(2)), -9) atol = 1.0e-6
    @test pf.ub ≈ cdf(Normal(0, sqrt(2)), -9) atol = 1.0e-6

    # the same result as the parametric p-box with the same envelope: random slicing only uses the envelope
    X_parametric = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1)), :X)
    pf_parametric, _ = probability_of_failure(UQModel[], df -> 9 .+ df.X .+ df.Y, [X_parametric, Y], RandomSlicing(FORM()))
    @test pf.lb ≈ pf_parametric.lb
    @test pf.ub ≈ pf_parametric.ub
end

@testitem "Imprecise Probability of Failure: Random Slicing: Distribution-free p-box contains the parametric p-box" begin
    # the double loop only searches the Normal(μ, 1), μ ∈ [0, 1], while the distribution-free p-box with the same envelope contains every cdf between Normal(1, 1) and Normal(0, 1). For a non-monotone performance function the extremes are not reached by the bounding cdfs, so the distribution-free interval is strictly wider
    X = RandomVariable(
        DistributionFreeProbabilityBox(
            x -> cdf(Normal(1, 1), x),
            x -> cdf(Normal(0, 1), x),
            -Inf,
            Inf
        ),
        :X,
    )
    X_parametric = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1)), :X)
    Y = RandomVariable(Normal(), :Y)

    performance = df -> 1.5 .- (df.X .- 0.5) .^ 2 .+ 0.3 .* df.Y

    pf, _ = probability_of_failure(UQModel[], performance, [X, Y], RandomSlicing(MonteCarlo(10^4)))
    pf_parametric, _ = probability_of_failure(
        UQModel[], performance, [X_parametric, Y], DoubleLoop(MonteCarlo(10^4))
    )

    @test pf.lb < pf_parametric.lb
    @test pf_parametric.ub < pf.ub
end

@testitem "Imprecise Probability of Failure: Random Slicing: Distribution-free p-box from EmpiricalDistributions" begin
    # deterministic data: the quantiles of Normal(30, 5) at 200 evenly spaced probabilities. The lower bound is the same data shifted right by 2: same bandwidth, so the cdfs don't cross
    data = quantile.(Normal(30, 5), range(0.005, 0.995, 200))
    lower = EmpiricalDistribution(data .+ 2)
    upper = EmpiricalDistribution(data)
    Z = RandomVariable(DistributionFreeProbabilityBox(lower, upper), :Z)

    # failure is Z > 35: its probability goes from 1 - upper_cdf(35) to 1 - lower_cdf(35). The FORM tolerance is tighter than the default 1e-3, which leaves an error of ~4e-5
    pf, _ = probability_of_failure(UQModel[], df -> 35 .- df.Z, [Z], RandomSlicing(FORM(10, 1.0e-9)))

    @test pf.lb ≈ 1 - cdf(upper, 35) atol = 1.0e-9
    @test pf.ub ≈ 1 - cdf(lower, 35) atol = 1.0e-9
end

@testitem "Imprecise Probability of Failure: Distribution-free p-box only with Random Slicing" begin
    X = RandomVariable(
        DistributionFreeProbabilityBox(
            x -> cdf(Normal(1, 1), x),
            x -> cdf(Normal(0, 1), x),
            -Inf,
            Inf
        ),
        :X,
    )
    Z = RandomVariable(
        DistributionFreeProbabilityBox(
            x -> cdf(Uniform(1, 3), x),
            x -> cdf(Uniform(0, 2), x),
            0,
            3
        ),
        :Z,
    )
    Y = RandomVariable(Normal(), :Y)
    performance = df -> 9 .+ df.X .+ df.Y

    # the double loop optimizes over the parameters of the p-boxes, and a distribution-free p-box has none
    @test_throws ArgumentError(
        "DoubleLoop doesn't support DistributionFreeProbabilityBox inputs (X), use RandomSlicing"
    ) probability_of_failure(UQModel[], performance, [X, Y], DoubleLoop(FORM()))

    # also as a marginal of a joint distribution, and all of them are listed
    @test_throws ArgumentError(
        "DoubleLoop doesn't support DistributionFreeProbabilityBox inputs (X), use RandomSlicing"
    ) probability_of_failure(UQModel[], performance, JointDistribution([X, Y]), DoubleLoop(FORM()))
    @test_throws ArgumentError(
        "DoubleLoop doesn't support DistributionFreeProbabilityBox inputs (X, Z), use RandomSlicing"
    ) probability_of_failure(UQModel[], df -> 9 .+ df.X .+ df.Y .+ df.Z, [JointDistribution([X, Y]), Z], DoubleLoop(FORM()))

    # the single loop methods reject every imprecise input
    @test_throws ErrorException(
        "You must use DoubleLoop or RandomSlicing with imprecise inputs or models."
    ) probability_of_failure(UQModel[], performance, [X, Y], FORM())
    @test_throws ErrorException(
        "You must use DoubleLoop or RandomSlicing with imprecise inputs or models."
    ) probability_of_failure(UQModel[], performance, [X, Y], MonteCarlo(10^3))
end
