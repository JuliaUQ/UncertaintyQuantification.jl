@testsnippet GPTestSetup begin
    import UncertaintyQuantification: sample

    using QuasiMonteCarlo
    function R²(y_true, y_pred)
        ss_res = sum((y_true .- y_pred) .^ 2)
        ss_tot = sum((y_true .- mean(y_true)) .^ 2)
        return 1.0 - ss_res / ss_tot
    end

end


@testitem "GaussianProcess Posterior Sampling" setup = [TestSetup] begin

    training = DataFrame(x = [0.0, 1.0, 2.0], y = [10.0, 11.0, 14.0])
    gp = GaussianProcess(
        copy(training), :y;
        mean = ConstMean(10.0), σ² = 1.0e-3, learn_hyperparameters = false,
    )
    locations = DataFrame(x = [0.5, 1.5])

    # The training inputs have mean 1 and standard deviation 1.
    Random.seed!(42)
    expected = permutedims(rand(gp.posterior([-0.5 0.5], gp.σ²; obsdim = 2), 10))
    Random.seed!(42)
    @test sample!(gp, locations, 10) === nothing

    @test hcat(locations.y...) ≈ expected
    @test locations.x == [0.5, 1.5]
    @test gp.data == training

end

@testitem "GaussianProcess explicit RNG and latent sampling" begin
    using DataFrames
    using Random
    using Statistics
    training = DataFrame(x = [-1.0, 1.0], y = [1.0, -1.0])
    gp = GaussianProcess(training, :y; σ² = 0.25, learn_hyperparameters = false)
    locations = DataFrame(x = [0.0, 0.0])
    global_rng = copy(Random.default_rng())
    @test sample!(MersenneTwister(17), gp, locations, 256; latent = true, jitter = 1.0e-10) === nothing
    @test rand() == rand(global_rng)
    @test maximum(abs.(locations.y[1] .- locations.y[2])) < 1.0e-3
    @test all((locations.y[1] .< 0) .== (locations.y[2] .< 0))
    @test 0.1 < mean(locations.y[1] .< 0) < 0.9
    repeated = DataFrame(x = [0.0, 0.0])
    sample!(MersenneTwister(17), gp, repeated, 256; latent = true, jitter = 1.0e-10)
    @test locations.y == repeated.y
    noisy = DataFrame(x = [0.0, 0.0])
    sample!(MersenneTwister(17), gp, noisy, 256)
    @test std(noisy.y[1] .- noisy.y[2]) > 0.3
    @test_throws ArgumentError sample!(gp, locations; jitter = -1)
    @test_throws ArgumentError sample!(gp, locations; jitter = NaN)
    @test gp.data == training
    agp = AdaptiveGaussianProcess(gp, RandomVariable(Normal(), :x), Model(df -> df.x, :y))
    wrapped = DataFrame(x = [0.0, 0.0])
    sample!(MersenneTwister(17), agp, wrapped, 256; latent = true, jitter = 1.0e-10)
    @test wrapped == locations
end

@testitem "GaussianProcess explicit RNG for hyperparameter restarts" begin
    using DataFrames
    using Random
    using Optim
    data = DataFrame(x = [-1.0, 0.0, 1.0], y = [0.8, 0.0, 1.2])
    optimizer = MaximumLikelihoodEstimation(NelderMead(), Optim.Options(; iterations = 20); restarts = 2)
    global_rng = copy(Random.default_rng())
    first = GaussianProcess(
        data, :y; kernel = 2.0 * SqExponentialKernel(),
        optimizer = optimizer, rng = MersenneTwister(41)
    )
    @test rand() == rand(global_rng)
    second = GaussianProcess(
        data, :y; kernel = 2.0 * SqExponentialKernel(),
        optimizer = optimizer, rng = MersenneTwister(41)
    )
    a, b = DataFrame(x = [-0.5, 0.5]), DataFrame(x = [-0.5, 0.5])
    evaluate!(first, a; variance = true)
    evaluate!(second, b; variance = true)
    @test a == b
end

@testitem "GaussianProcess 1D Inputs" setup = [
    GPTestSetup,
] begin

    inputs = [RandomVariable(Uniform(0, 3), :x), Parameter(0.5, :a), Parameter(2.0, :b)]

    model = Model(df -> df.a .* sin.(df.b .* df.x), :y)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(100, HaltonSample()), :y;
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    verify_true = sample(inputs, 100)
    verify_gp = verify_true[:, [:x]]

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.02
    @test :x in propertynames(gp.data)
    @test :y in propertynames(gp.data)
    @test !(:a in propertynames(gp.data))
    @test !(:b in propertynames(gp.data))
end

@testitem "GaussianProcess Non RV inputs" setup = [
    GPTestSetup,
] begin

    inputs = [RandomVariable(Uniform(0, 3), :x), Parameter(0.5, :a), Parameter(2.0, :b)]

    model = Model(df -> df.a .* sin.(df.b .* df.x), :y)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(100, HaltonSample()), :y;
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    verify_true = sample(inputs, 100)
    verify_gp = verify_true[:, [:x]]

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.02
end

@testitem "GaussianProcess
 1D Samples" setup = [
    GPTestSetup,
] begin

    inputs = [RandomVariable(Uniform(0, 3), :x), Parameter(0.5, :a), Parameter(2.0, :b)]
    model = Model(df -> df.a .* sin.(df.b .* df.x), :y)

    data = sample(inputs, (QuasiMonteCarloSampling(200, HaltonSample())))
    evaluate!(model, data)

    gp = GaussianProcess(
        data[1:100, :], :y, [:x];
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    verify_true = data[101:200, :]
    verify_gp = verify_true[:, [:x]]

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    verify_true = sample(inputs, 100)
    verify_gp = verify_true[:, [:x]]

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.02

end


@testitem "GaussianProcess
 2D Inputs" setup = [
    GPTestSetup,
] begin
    inputs = [RandomVariable(Uniform(0, 3), :x1), RandomVariable(Uniform(0, 3), :x2)]
    model = Model(df -> 0.3 .* sin.(2.0 * df.x1) .+ 0.3 .* cos.(2.0 * df.x2), :y)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(100, HaltonSample()), :y;
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    verify_true = sample(inputs, 100)
    verify_gp = copy(verify_true)

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.02
end

@testitem "GaussianProcess 2D Data" setup = [
    GPTestSetup,
] begin
    inputs = [RandomVariable(Uniform(0, 3), :x1), RandomVariable(Uniform(0, 3), :x2)]
    model = Model(df -> 0.3 .* sin.(2.0 * df.x1) .+ 0.3 .* cos.(2.0 * df.x2), :y)

    data = sample(inputs, (QuasiMonteCarloSampling(200, HaltonSample())))
    evaluate!(model, data)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(100, HaltonSample()), :y;
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    verify_true = data[101:200, :]
    verify_gp = copy(verify_true)

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.02
end

@testitem "GaussianProcess evaluate!" setup = [
    GPTestSetup,
] begin
    inputs = [RandomVariable(Uniform(0, 3), :x1), RandomVariable(Uniform(0, 3), :x2)]
    model = Model(df -> 0.3 .* sin.(2.0 * df.x1) .+ 0.3 .* cos.(2.0 * df.x2), :y)

    data = sample(inputs, (QuasiMonteCarloSampling(200, HaltonSample())))
    evaluate!(model, data)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(100, HaltonSample()), :y;
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    data = sample(inputs, 10)
    evaluate!(gp, data)
    @test hasproperty(data, "y_mean")
    @test !hasproperty(data, "y_var")

    data = sample(inputs, 10)
    evaluate!(gp, data; mean = false, variance = true)
    @test !hasproperty(data, "y_mean")
    @test hasproperty(data, "y_var")

    data = sample(inputs, 10)
    evaluate!(gp, data; variance = true)
    @test hasproperty(data, "y_mean")
    @test hasproperty(data, "y_var")

end


@testitem "GaussianProcess Learn Noise" setup = [
    GPTestSetup,
] begin

    inputs = [RandomVariable(Uniform(0, 3), :x), Parameter(0.5, :a), Parameter(2.0, :b)]
    model = Model(df -> df.a .* sin.(df.b .* df.x) .+ randn(size(df, 1)), :y)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(200, HaltonSample()), :y;
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_noise = true
    )

    @test gp.σ² ≈ 1.0 rtol = 0.2

    gp = GaussianProcess(
        inputs, model, MonteCarlo(10), :y;
        mean = ConstMean(0.0),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_noise = false
    )

    @test gp.σ² == 1.0e-5
end
