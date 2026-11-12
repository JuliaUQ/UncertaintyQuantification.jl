@testitem "Laplace estimate of a general density" begin
    using ADTypes, DataFrames, LinearAlgebra, Optim

    μ = [1.5, -0.5]
    Σ = [1.0 0.4; 0.4 2.0]
    target = MvNormal(μ, Σ)
    starts = DataFrame(x = [0.0, 2.0], y = [0.0, -2.0])
    original = copy(starts)
    options = Optim.Options(; iterations = 100, g_abstol = 1.0e-9)
    backend = AutoFiniteDiff()
    estimate = LaplaceEstimate(starts; optimoptions = options, adbackend = backend)
    @test estimate.optimoptions === options
    @test estimate.adbackend === backend
    @test !hasproperty(estimate, :prior)

    # A large negative offset must not underflow the mixture weights.
    logtarget = df -> begin
        df.output = df.x .+ df.y
        [logpdf(target, [row.x, row.y]) - 1000 for row in eachrow(df)]
    end
    approximation = laplaceestimate(logtarget, estimate)
    @test names(approximation) == [:x, :y]
    @test length(approximation.d.components) == 1
    @test mean(approximation) ≈ μ atol = 1.0e-5
    @test cov(approximation.d) ≈ Σ atol = 1.0e-4
    @test starts == original

    density = df -> [pdf(target, [row.x, row.y]) for row in eachrow(df)]
    approximation = laplaceestimate(density, LaplaceEstimate(starts; islog = false))
    @test mean(approximation) ≈ μ atol = 1.0e-5
    @test cov(approximation.d) ≈ Σ atol = 1.0e-4
    unfiltered = laplaceestimate(logtarget, LaplaceEstimate(starts; filtertolerance = 0))
    @test length(unfiltered.d.components) == 2
end

@testitem "Laplace multiple modes and bounds" begin
    using DataFrames, Optim

    mixture = MixtureModel([MvNormal([-3.0], [0.1;;]), MvNormal([3.0], [0.1;;])], [0.2, 0.8])
    approximation = laplaceestimate(
        df -> [logpdf(mixture, [x]) for x in df.x],
        LaplaceEstimate(DataFrame(x = [-3.0, 3.0, 3.1]))
    )
    @test length(approximation.d.components) == 2
    @test [mean(c)[1] for c in approximation.d.components] ≈ [-3.0, 3.0] atol = 1.0e-5
    @test approximation.d.prior.p ≈ [0.2, 0.8] atol = 1.0e-5

    bounded = LaplaceEstimate(DataFrame(p = [0.5]); lowerbounds = [0.0], upperbounds = [1.0])
    approximation = laplaceestimate(df -> logpdf.(Beta(4, 3), df.p), bounded)
    @test mean(approximation)[1] ≈ 0.6 atol = 1.0e-5
    @test var(approximation)[1] ≈ 1 / (3 / 0.6^2 + 2 / 0.4^2) atol = 1.0e-5

    # The options stored on the estimator are used by the optimizer.
    stopped = LaplaceEstimate(
        DataFrame(x = [5.0]); optimizer = Optim.NelderMead(),
        optimoptions = Optim.Options(; iterations = 0)
    )
    approximation = laplaceestimate(df -> -0.5 .* df.x .^ 2, stopped)
    @test mean(approximation)[1] > 4
end

@testitem "Laplace estimate validates starts and bounds" begin
    using DataFrames

    @test_throws ArgumentError LaplaceEstimate(DataFrame())
    @test_throws ArgumentError LaplaceEstimate(DataFrame(x = Float64[]))
    @test_throws ArgumentError LaplaceEstimate(DataFrame(x = [Inf]))
    @test_throws DimensionMismatch LaplaceEstimate(DataFrame(x = [0.0], y = [0.0]); lowerbounds = [-1.0])
    @test_throws ArgumentError LaplaceEstimate(DataFrame(x = [0.0]); lowerbounds = [1.0], upperbounds = [-1.0])
    @test_throws ArgumentError LaplaceEstimate(DataFrame(x = [2.0]); upperbounds = [1.0])
    @test_throws ArgumentError LaplaceEstimate(DataFrame(x = [0.0]); filtertolerance = -1)
end
