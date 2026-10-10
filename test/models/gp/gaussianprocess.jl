@testsnippet GPSetup begin
    import UncertaintyQuantification: sample

    function create_test_data(n_samples::Int, lower::Real, upper::Real, dim::Int)
        data = lower .+ (upper - lower) .* rand(n_samples, dim)
        df = DataFrame()
        for i in 1:dim
            name = Symbol("x$i")
            df[!, name] = data[:, i]
        end
        return df
    end
end

@testitem "1D GP from data" setup = [TestSetup, GPSetup] begin
    lower = 0
    upper = 5
    n = 10

    x = collect(range(lower, stop = upper, length = n))
    y = sin.(x)
    data = DataFrame(:x1 => x, :y => y)
    mean = ConstMean(0.0)
    kernel = SqExponentialKernel()
    gp = GaussianProcess(
        data, :y;
        σ² = 0.0,
        mean = mean,
        kernel = kernel,
        learn_noise = false
    )

    @test gp.σ² == 0.0

    gp = GaussianProcess(
        data, :y;
        σ² = 0.0,
        mean = mean,
        kernel = kernel,
        learn_noise = true
    )

    @test gp.σ² > 0.0

    df_mean = create_test_data(n, lower, upper, 1)
    df_var = create_test_data(n, lower, upper, 1)
    df_mean_var = create_test_data(n, lower, upper, 1)

    evaluate!(gp, df_mean; mode = :mean)
    evaluate!(gp, df_var; mode = :var)
    evaluate!(gp, df_mean_var; mode = :mean_and_var)
    @test :y_mean in propertynames(df_mean)
    @test !(:y_var in propertynames(df_mean))
    @test :y_var in propertynames(df_var)
    @test !(:y_mean in propertynames(df_var))
    @test :y_mean in propertynames(df_mean_var)
    @test :y_var in propertynames(df_mean_var)

    @test_throws ArgumentError evaluate!(gp, df_mean; mode = :error)

    @test_throws DomainError GaussianProcess(
        data, :y;
        normalize = true,
        σ² = -1.0,
        mean = mean,
        kernel = kernel,
        learn_noise = false
    )
end

@testitem "1D GP from inputs" setup = [TestSetup, GPSetup, QMC] begin
    lower = 0
    upper = 5
    n = 10

    xrv = [Parameter(1.5, :p), RandomVariable(Uniform(lower, upper), :x1)]
    xrv_single = RandomVariable(Uniform(lower, upper), :x1)
    model = Model(
        df -> df.p .* sin.(df.x1), :y
    )
    model_single = Model(
        df -> 1.5 .* sin.(df.x1), :y
    )

    mean = ConstMean(0.0)
    kernel = SqExponentialKernel()

    design = QuasiMonteCarloSampling(10, LatinHypercubeSample())

    gp = GaussianProcess(
        xrv, model, design, :y, [:x1];
        σ² = 0.0,
        mean = mean,
        kernel = kernel,
        learn_noise = false
    )
    @test gp.σ² == 0.0

    gp = GaussianProcess(
        xrv_single, model_single, design, :y;
        σ² = 0.0,
        mean = mean,
        kernel = kernel,
        learn_noise = false
    )

    @test gp.σ² == 0.0

    gp = GaussianProcess(
        xrv, model, design, :y, [:x1];
        σ² = 0.0,
        mean = mean,
        kernel = kernel,
        learn_noise = true
    )

    @test gp.σ² > 0.0


    gp = GaussianProcess(
        xrv_single, model_single, design, :y;
        σ² = 0.0,
        mean = mean,
        kernel = kernel,
        learn_noise = true
    )

    @test gp.σ² > 0.0

    df_mean = sample(xrv, n)
    df_var = sample(xrv, n)
    df_mean_var = sample(xrv, n)
    evaluate!(gp, df_mean; mode = :mean)
    evaluate!(gp, df_var; mode = :var)
    evaluate!(gp, df_mean_var; mode = :mean_and_var)
    @test :y_mean in propertynames(df_mean)
    @test !(:y_var in propertynames(df_mean))
    @test :y_var in propertynames(df_var)
    @test !(:y_mean in propertynames(df_var))
    @test :y_mean in propertynames(df_mean_var)
    @test :y_var in propertynames(df_mean_var)

    @test_throws DomainError GaussianProcess(
        xrv, model, design, :y;
        σ² = -1.0,
        mean = mean,
        kernel = kernel,
        learn_noise = false
    )

end

@testitem "2D GP from data" setup = [TestSetup, GPSetup] begin
    lower = 0
    upper = 5
    σ² = 1.0e-5
    n = 10
    mean = ConstMean(0.0)
    kernel = SqExponentialKernel()

    x = [collect(range(lower, stop = upper, length = n)) collect(range(lower, stop = upper, length = n))]
    y = sin.(x[:, 1]) + cos.(x[:, 2])
    data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)

    gp = GaussianProcess(
        data, :y;
        σ² = σ²,
        mean = mean,
        kernel = kernel,
        learn_noise = true
    )

    df = create_test_data(n, lower, upper, 2)
    evaluate!(gp, df; mode = :mean_and_var)
    @test :y_mean in propertynames(df)
    @test :y_var in propertynames(df)
end

@testitem "2D GP from inputs" setup = [TestSetup, GPSetup, QMC] begin
    σ² = 1.0e-5
    n = 10
    mean = ConstMean(0.0)
    kernel = SqExponentialKernel()

    xrv = [Parameter(1.5, :p), RandomVariable(Uniform(0, 5), :x1), RandomVariable(Uniform(0, 5), :x2)]
    model = Model(
        df -> df.p .* sin.(df.x1) + df.p .* cos.(df.x2), :y
    )

    gp = GaussianProcess(
        xrv, model, QuasiMonteCarloSampling(n, LatinHypercubeSample()), :y, [:x1, :x2];
        σ² = σ²,
        mean = mean,
        kernel = kernel,
        learn_noise = false
    )

    df = sample(xrv, n)
    evaluate!(gp, df; mode = :mean_and_var)
    @test :y_mean in propertynames(df)
    @test :y_var in propertynames(df)

end

@testitem "GP posterior sampling" begin
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
