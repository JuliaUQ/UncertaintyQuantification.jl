@testsnippet GPTestSetup begin
    import UncertaintyQuantification: sample

    using QuasiMonteCarlo
    function R²(y_true, y_pred)
        ss_res = sum((y_true .- y_pred) .^ 2)
        ss_tot = sum((y_true .- mean(y_true)) .^ 2)
        return 1.0 - ss_res / ss_tot
    end

end


@testitem "GaussianProcess Posterior Sampling" begin
    using DataFrames
    using Random

    training = DataFrame(x = [0.0, 1.0, 2.0], y = [10.0, 11.0, 14.0])
    gp = GaussianProcess(
        copy(training), :y;
        mean = ConstMean(10.0), σ² = 1.0e-3, learn_hyperparameters = false,
    )
    locations = DataFrame(x = [0.5, 1.5])

    # The training inputs have mean 1 and standard deviation 1.
    Random.seed!(42)
    expected = rand(gp.posterior([-0.5 0.5], gp.σ²; obsdim = 2), 2)
    Random.seed!(42)
    @test sample!(gp, locations, 2) === nothing
    @test propertynames(locations) == [:x, :y_sample_1, :y_sample_2]
    @test Matrix(locations[:, [:y_sample_1, :y_sample_2]]) ≈ expected
    @test locations.x == [0.5, 1.5]
    @test gp.data == training

    second_draw = copy(locations.y_sample_2)
    Random.seed!(123)
    @test sample!(gp, locations) === nothing
    @test locations.y_sample_1 != expected[:, 1]
    @test locations.y_sample_2 == second_draw
end

@testitem "GaussianProcess 1D Inputs" setup = [
    GPTestSetup,
] begin

    inputs = [RandomVariable(Uniform(0, 3), :x), Parameter(0.5, :a), Parameter(2.0, :b)]

    model = Model(df -> df.a .* sin.(df.b .* df.x), :y)

    gp = GaussianProcess(
        inputs, model, QuasiMonteCarloSampling(100, HaltonSample()), :y, [:x];
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_hyperparameters = false
    )

    verify_true = sample(inputs, 100)
    verify_gp = verify_true[:, [:x]]

    evaluate!(model, verify_true)
    evaluate!(gp, verify_gp)

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.01
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

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.01

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

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.01
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

    @test R²(verify_true.y, verify_gp.y_mean) ≈ 0.99 rtol = 0.01
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
        inputs, model, QuasiMonteCarloSampling(200, HaltonSample())
        , :y, [:x];
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5,
        learn_noise = true
    )

    @test gp.σ² ≈ 1.0 rtol = 0.2

    gp = GaussianProcess(
        inputs, model, MonteCarlo(10)
        , :y, [:x];
        mean = ZeroMean(),
        kernel = SqExponentialKernel(),
        σ² = 1.0e-5
        ,
        learn_noise = false
    )

    @test gp.σ² == 1.0e-5
end
