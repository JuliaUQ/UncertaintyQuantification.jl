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
