@testsnippet AdaptiveGPSetup begin
    import UncertaintyQuantification: sample
    using QuasiMonteCarlo

    input = RandomVariable(Uniform(-2, 12), :x1)
    n_design_points = 8
    design = QuasiMonteCarloSampling(n_design_points, LatinHypercubeSample())
    model = Model(df -> df.x1 .^ 2 .* sin.(df.x1), :y)
    acquisition_function = MaximumVariance()
    candidate_sampling = MonteCarlo(200)
end

@testitem "Adaptive Gaussian process construction and refinement" setup = [TestSetup, AdaptiveGPSetup] begin
    agp = AdaptiveGaussianProcess(
        input, model, design, :y, acquisition_function;
        candidate_sampling = candidate_sampling,
        learn_hyperparameters = false,
    )

    @test agp isa AdaptiveGaussianProcess
    @test agp.gp isa GaussianProcess
    @test agp.acquisition_function === acquisition_function
    @test agp.candidate_sampling === candidate_sampling
    @test nrow(agp.gp.data) == n_design_points
    @test agp.gp.output == :y

    initial_gp = agp.gp
    @test refine!(agp, 0) === nothing
    @test agp.gp === initial_gp
    @test nrow(agp.gp.data) == n_design_points

    @test refine!(agp, 2) === nothing
    @test agp.gp !== initial_gp
    @test nrow(agp.gp.data) == n_design_points + 2
    @test agp.gp.data.y ≈ agp.gp.data.x1 .^ 2 .* sin.(agp.gp.data.x1)

    @test refine!(agp) === nothing
    @test nrow(agp.gp.data) == n_design_points + 3
end

@testitem "Adaptive Gaussian process configuration and input selection" setup = [TestSetup, AdaptiveGPSetup] begin
    inputs = [Parameter(2.0, :p), input]
    models = [
        Model(df -> df.p .* sin.(df.x1), :intermediate),
        Model(df -> df.intermediate .+ 1, :y),
    ]

    for normalize in (true, false)
        agp = AdaptiveGaussianProcess(
            inputs, models, design, :y, acquisition_function;
            normalize = normalize,
            mean = ConstMean(1.0),
            kernel = MaternKernel(),
            learn_hyperparameters = false,
            candidate_sampling = candidate_sampling,
        )

        @test agp.gp.inputs == [:x1]
        @test (agp.gp.transform === nothing) == !normalize
        @test nrow(agp.gp.data) == n_design_points
        # @test all(agp.gp.data.p .== 2.0)
        @test agp.gp.data.y ≈ 2 .* sin.(agp.gp.data.x1) .+ 1
        @test agp.gp.posterior.prior.mean.c == 1.0

        initial_kernel_parameter = agp.gp.posterior.prior.kernel.ν
        @test refine!(agp) === nothing
        @test nrow(agp.gp.data) == n_design_points + 1
        @test all(agp.gp.data.p .== 2.0)
        @test agp.gp.data.y ≈ 2 .* sin.(agp.gp.data.x1) .+ 1
        @test agp.gp.posterior.prior.mean.c == 1.0
        @test agp.gp.posterior.prior.kernel.ν == initial_kernel_parameter

        predictions = agp.gp.data[:, [:x1]]
        @test evaluate!(agp, predictions) === nothing
        @test predictions.y_mean ≈ agp.gp.data.y atol = 1.0e-3
    end
end

@testitem "Adaptive Gaussian process prediction wrappers" setup = [TestSetup, AdaptiveGPSetup] begin
    agp = AdaptiveGaussianProcess(
        input, model, design, :y;
        candidate_sampling = candidate_sampling,
        learn_hyperparameters = false,
    )
    locations = DataFrame(x1 = [-1.0, 0.5, 3.0])

    wrapped_predictions = copy(locations)
    direct_predictions = copy(locations)
    @test evaluate!(agp, wrapped_predictions; variance = true) === nothing
    @test evaluate!(agp.gp, direct_predictions; variance = true) === nothing
    @test wrapped_predictions.y_mean ≈ direct_predictions.y_mean
    @test wrapped_predictions.y_var ≈ direct_predictions.y_var

    wrapped_samples = copy(locations)
    direct_samples = copy(locations)
    Random.seed!(1234)
    @test sample!(agp, wrapped_samples, 3) === nothing
    Random.seed!(1234)
    @test sample!(agp.gp, direct_samples, 3) === nothing
    @test wrapped_samples.y == direct_samples.y

    observed = copy(locations)
    evaluate!(model, observed)
    @test logpdf(agp, observed) ≈ logpdf(agp.gp, observed)
end

@testitem "Adaptive Gaussian process refit preserves transform and skips duplicates" setup = [TestSetup, AdaptiveGPSetup] begin
    agp = AdaptiveGaussianProcess(
        input, model, design, :y;
        learn_hyperparameters = false,
        candidate_sampling = candidate_sampling,
    )
    initial_data = deepcopy(agp.gp.data)
    initial_transform = agp.gp.transform

    duplicate = copy(agp.gp.data[1:1, :])
    refitted_gp = UncertaintyQuantification._refit_gp(
        agp.gp, duplicate, agp.optimizer, agp.gp.σ², agp.learn_noise,
        agp.learn_hyperparameters,
    )
    @test refitted_gp.transform === initial_transform
    @test nrow(refitted_gp.data) == n_design_points
    @test refitted_gp.data == initial_data

    new_data = DataFrame(x1 = [15.0])
    evaluate!(model, new_data)
    refitted_gp = UncertaintyQuantification._refit_gp(
        refitted_gp, new_data, agp.optimizer, refitted_gp.σ², agp.learn_noise,
        agp.learn_hyperparameters,
    )
    @test refitted_gp.transform === initial_transform
    @test nrow(refitted_gp.data) == n_design_points + 1

    refitted_gp = UncertaintyQuantification._refit_gp(
        refitted_gp, new_data, agp.optimizer, refitted_gp.σ², agp.learn_noise,
        agp.learn_hyperparameters,
    )
    @test nrow(refitted_gp.data) == n_design_points + 1
end

@testitem "Adaptive Gaussian process with 2D inputs" setup = [TestSetup, AdaptiveGPSetup] begin
    inputs = [RandomVariable(Uniform(0, 3), :x1), RandomVariable(Uniform(0, 3), :x2)]
    model = Model(df -> 0.3 .* sin.(2.0 .* df.x1) .+ 0.3 .* cos.(2.0 .* df.x2), :y)
    design = QuasiMonteCarloSampling(100, HaltonSample())

    Random.seed!(42)
    agp = AdaptiveGaussianProcess(
        inputs, model, design, :y;
        candidate_sampling = MonteCarlo(1_000),
        learn_hyperparameters = false,
    )

    @test agp.gp.inputs == [:x1, :x2]
    @test nrow(agp.gp.data) == 100
    @test refine!(agp, 10) === nothing
    @test nrow(agp.gp.data) == 110

    verify_true = sample(inputs, 100)
    evaluate!(model, verify_true)
    verify_gp = copy(verify_true[:, [:x1, :x2]])
    evaluate!(agp, verify_gp)

    ss_res = sum((verify_true.y .- verify_gp.y_mean) .^ 2)
    ss_tot = sum((verify_true.y .- mean(verify_true.y)) .^ 2)
    @test 1.0 - ss_res / ss_tot ≈ 0.99 rtol = 0.01
end
