@testset "Imprecise Probability of Failure" begin
    @testset "Double Loop" begin
        @testset "P-boxes only" begin
            X = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :X,
            )
            Y = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :Y,
            )

            pf, x_lb, x_ub = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
            )

            @test pf.lb ≈ cdf(Normal(2, sqrt(2)), -9) atol = 1e-6
            @test pf.ub ≈ cdf(Normal(-2, sqrt(8)), -9) atol = 1e-6
        end

        @testset "P-boxes with Copula" begin
            X = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :X,
            )
            Y = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :Y,
            )

            @testset "independent" begin
                c = GaussianCopula([1 0.0; 0.0 1])

                jd = JointDistribution(c, [X, Y])

                pf, x_lb, x_ub = probability_of_failure(
                    UQModel[], df -> 9 .+ df.X .+ df.Y, jd, DoubleLoop(FORM())
                )

                @test pf.lb ≈ cdf(Normal(2, sqrt(2)), -9) atol = 1e-6
                @test pf.ub ≈ cdf(Normal(-2, sqrt(8)), -9) atol = 1e-6
            end

            @testset "dependent" begin
                c = GaussianCopula([1 0.71; 0.71 1])

                jd = JointDistribution(c, [X, Y])

                pf, x_lb, x_ub = probability_of_failure(
                    UQModel[], df -> 9 .+ df.X .+ df.Y, jd, DoubleLoop(FORM())
                )

                @test pf.lb ≈ cdf(Normal(2, sqrt(1^2 + 1^2 + 2 * (0.71 * 1 * 1))), -9)
                @test pf.ub ≈ cdf(Normal(-2, sqrt(2^2 + 2^2 + 2 * (0.71 * 2 * 2))), -9)
            end
        end

        @testset "Interval - Distribution" begin
            X = IntervalVariable(-1, 1, :X)
            Y = RandomVariable(Normal(0, 2), :Y)

            pf, x_lb, x_ub = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
            )

            @test pf.lb ≈ cdf(Normal(1, 2), -9) atol = 1e-6
            @test pf.ub ≈ cdf(Normal(-1, 2), -9) atol = 1e-6
        end

        @testset "P-box - Distribution" begin
            X = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :X,
            )
            Y = RandomVariable(Normal(0, 2), :Y)

            pf, x_lb, x_ub = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
            )

            @test pf.lb ≈ cdf(Normal(1, sqrt(5)), -9) atol = 1e-6
            @test pf.ub ≈ cdf(Normal(-1, sqrt(8)), -9) atol = 1e-6
        end

        @testset "Interval - p-box" begin
            X = IntervalVariable(-1, 1, :X)
            Y = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(1, 2))), :Y
            )

            pf, x_lb, x_ub = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], DoubleLoop(FORM())
            )

            @test pf.lb ≈ cdf(Normal(1, 1), -9) atol = 1e-6
            @test pf.ub ≈ cdf(Normal(-1, 2), -9) atol = 1e-6

            @test x_lb ≈ [1, 1]
            @test x_ub ≈ [-1, 2]
        end
    end

    @testset "Random Slicing" begin
        @testset "P-boxes only" begin
            X = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :X,
            )
            Y = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :Y,
            )

            pf, _ = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
            )

            pbox_analyt = ProbabilityBox{Normal}(
                Dict(:μ => Interval(-2, 2), :σ => Interval(sqrt(2), sqrt(8)))
            )
            failure_analty = cdf(pbox_analyt, -9)

            @test pf.lb ≈ failure_analty.lb atol = 1e-6
            @test pf.ub ≈ failure_analty.ub atol = 1e-6
        end

        @testset "P-boxes with Copula" begin
            X = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :X,
            )
            Y = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :Y,
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

            @test pf.lb ≈ failure_analty.lb atol = 1e-6
            @test pf.ub ≈ failure_analty.ub atol = 1e-6
        end

        @testset "Interval - Distribution" begin
            X = IntervalVariable(-1, 1, :X)
            Y = RandomVariable(Normal(0, 2), :Y)

            pf, _ = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
            )

            pbox_analyt = ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => 2))
            failure_analty = cdf(pbox_analyt, -9)

            @test pf.lb ≈ failure_analty.lb atol = 1e-6
            @test pf.ub ≈ failure_analty.ub atol = 1e-6
        end

        @testset "P-box - Distribution" begin
            X = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))),
                :X,
            )
            Y = RandomVariable(Normal(0, 2), :Y)

            pf, _ = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
            )

            pbox_analyt = ProbabilityBox{Normal}(
                Dict(:μ => Interval(-1, 1), :σ => Interval(sqrt(5), sqrt(8)))
            )
            failure_analty = cdf(pbox_analyt, -9)

            @test pf.lb ≈ cdf(Normal(1, sqrt(5)), -9) atol = 1e-6
            @test pf.ub ≈ cdf(Normal(-1, sqrt(8)), -9) atol = 1e-6
        end

        @testset "Interval - p-box" begin
            X = IntervalVariable(-1, 1, :X)
            Y = RandomVariable(
                ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(1, 2))), :Y
            )

            pf, _ = probability_of_failure(
                UQModel[], df -> 9 .+ df.X .+ df.Y, [X, Y], RandomSlicing(FORM())
            )

            pbox_analyt = ProbabilityBox{Normal}(
                Dict(:μ => Interval(-1, 1), :σ => Interval(1, 2))
            )
            failure_analty = cdf(pbox_analyt, -9)

            @test pf.lb ≈ failure_analty.lb atol = 1e-6
            @test pf.ub ≈ failure_analty.ub atol = 1e-6
        end
    end
end

@testset "IPM Reliability Double Loop" begin
    x1 = RandomVariable(Normal(), :x1)
    x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-2, 1), :σ => 2)), :x2)
    x3 = IntervalVariable(-1, 2, :x3)

    # QMC Samples to fit IPM
    data = sample(
        [x1, RandomVariable(Normal(-0.5, 2), :x2), RandomVariable(Uniform(-2, 2), :x3)],
        HaltonSampling(150),
    )

    m1 = Model(df -> df.x1 + df.x3, :x4)
    m2 = Model(df -> 7 .- df.x4 .^ 2 .+ df.x2, :y)

    evaluate!([m1, m2], data)

    b = MonomialBasis(2, 3)

    ipm = IntervalPredictorModel(data, :y, b, [:x2, :x4])

    pf, x_lb, x_ub = probability_of_failure(
        [m1, m2], df -> 5.0 .- df.y, [x1, x2, x3], DoubleLoop(FORM())
    )

    pf_ipm, x_lb_ipm, x_ub_ipm = probability_of_failure(
        [m1, ipm], df -> 5.0 .- df.y, [x1, x2, x3], DoubleLoop(FORM())
    )

    @test pf.lb ≈ pf_ipm.lb atol = 1e-4
    @test pf.ub ≈ pf_ipm.ub atol = 1e-4
    @test x_lb ≈ x_lb_ipm atol = 1e-4
    @test x_ub ≈ x_ub_ipm atol = 1e-4 broken = true
end

@testset "IPM Reliability Random Slicing" begin
    x1 = RandomVariable(Normal(), :x1)
    x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-2, 1), :σ => 2)), :x2)
    x3 = IntervalVariable(-1, 2, :x3)

    # QMC Samples to fit IPM
    data = sample(
        [x1, RandomVariable(Normal(-0.5, 2), :x2), RandomVariable(Uniform(-2, 2), :x3)],
        HaltonSampling(150),
    )

    m1 = Model(df -> df.x1 + df.x3, :x4)
    m2 = Model(df -> 7 .- df.x4 .^ 2 .+ df.x2, :y)

    evaluate!([m1, m2], data)

    b = MonomialBasis(2, 3)

    ipm = IntervalPredictorModel(data, :y, b, [:x2, :x4])

    pf, _, _ = probability_of_failure(
        [m1, m2], df -> 5.0 .- df.y, [x1, x2, x3], RandomSlicing(FORM())
    )

    pf_ipm, _, _ = probability_of_failure(
        [m1, ipm], df -> 5.0 .- df.y, [x1, x2, x3], RandomSlicing(FORM())
    )

    @test pf.lb ≈ pf_ipm.lb atol = 1e-4
    @test pf.ub ≈ pf_ipm.ub atol = 1e-4
end
