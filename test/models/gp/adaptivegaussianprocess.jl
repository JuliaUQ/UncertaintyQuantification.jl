@testset "Adaptive Gaussian Process" begin

    # Input
    input = RandomVariable(Uniform(-2, 12), :x1)
    n_design_points = 8
    n_added_points = 3
    design = LatinHypercubeSampling(n_design_points)
    model = Model(df -> df.x1 .^ 2 .* sin.(df.x1), :y)

    acquisition_function = MaximumVariance()
    # small candidate set keeps the tests fast
    candidate_sampling = MonteCarlo(200)

    prior_mean = ConstMean(0.0)
    kernel = SqExponentialKernel()

    # Initial design used by the DataFrame-based constructors
    data = sample(input, design)
    evaluate!(model, data)

    @testset "Custom prior + UQInput" begin

        gp = AdaptiveGaussianProcess(
            input, model, design, :y, acquisition_function, n_added_points;
            mean = prior_mean,
            kernel = kernel,
            candidate_sampling = candidate_sampling,
        )

        @test gp isa GaussianProcess
        @test size(gp.data, 1) == n_design_points + n_added_points
        @test gp.output == :y
    end

    @testset "Default prior + UQInput" begin

        gp_default = AdaptiveGaussianProcess(
            input, model, design, :y, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        @test gp_default isa GaussianProcess
        @test size(gp_default.data, 1) == n_design_points + n_added_points
    end

    @testset "Pre-fit GaussianProcess" begin
        gp_model = GaussianProcess(
            input, model, design, :y;
            mean = ZeroMean(),
            kernel = SqExponentialKernel(),
        )

        gp = AdaptiveGaussianProcess(
            gp_model, input, model, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        @test gp isa GaussianProcess
        @test size(gp.data, 1) == n_design_points + n_added_points

        # n_added_points = 0 should return the training data unchanged
        gp_unchanged = AdaptiveGaussianProcess(
            gp_model, input, model, acquisition_function, 0;
            candidate_sampling = candidate_sampling,
        )
        @test size(gp_unchanged.data, 1) == n_design_points + n_added_points
    end

    @testset "Custom prior + DataFrame" begin

        gp = AdaptiveGaussianProcess(
            copy(data), input, model, :y, acquisition_function, n_added_points;
            mean = prior_mean,
            kernel = kernel,
            candidate_sampling = candidate_sampling,
        )

        @test gp isa GaussianProcess
        @test size(gp.data, 1) == n_design_points + n_added_points
        @test gp.output == :y
    end

    @testset "Default prior + DataFrame" begin

        gp_default = AdaptiveGaussianProcess(
            copy(data), input, model, :y, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        gp_explicit = AdaptiveGaussianProcess(
            copy(data), input, model, :y, acquisition_function, n_added_points;
            mean = prior_mean,
            kernel = kernel,
            candidate_sampling = candidate_sampling,
        )

        @test gp_default isa GaussianProcess
        @test size(gp_default.data, 1) == n_design_points + n_added_points
        @test size(gp_explicit.data, 1) == n_design_points + n_added_points
    end

    @testset "Not learn hyperparameters" begin
        gp_model = GaussianProcess(
            input, model, design, :y;
            mean = ConstMean(0.0),
            kernel = MaternKernel()
        )

        new_data = sample(input)
        evaluate!(model, new_data)

        new_gp_model = UncertaintyQuantification._refit_gp(
            deepcopy(gp_model), new_data, MaximumLikelihoodEstimation(), 1.0e-9, false, false
        )

        trend_initial = gp_model.posterior.prior.mean.c
        trend_adaptive = new_gp_model.posterior.prior.mean.c

        kernel_initial = gp_model.posterior.prior.kernel.ν
        kernel_adaptive = new_gp_model.posterior.prior.kernel.ν

        @test trend_initial == trend_adaptive
        @test kernel_initial == kernel_adaptive

    end

    @testset "Normalization and input selection" begin
        inputs = [Parameter(2.0, :p), input]
        models = [
            Model(df -> df.p .* sin.(df.x1), :intermediate),
            Model(df -> df.intermediate .+ 1, :y),
        ]
        initial_data = sample(inputs, design)
        evaluate!(models, initial_data)

        for normalize in (true, false), from_data in (true, false)
            gp = if from_data
                AdaptiveGaussianProcess(
                    copy(initial_data), inputs, models, :y, acquisition_function,
                    n_added_points, [:x1];
                    normalize = normalize,
                    mean = ConstMean(1.0),
                    kernel = MaternKernel(),
                    learn_hyperparameters = false,
                    candidate_sampling = candidate_sampling,
                )
            else
                AdaptiveGaussianProcess(
                    inputs, models, design, :y, acquisition_function,
                    n_added_points, [:x1];
                    normalize = normalize,
                    mean = ConstMean(1.0),
                    kernel = MaternKernel(),
                    learn_hyperparameters = false,
                    candidate_sampling = candidate_sampling,
                )
            end

            @test gp.inputs == [:x1]
            @test (gp.transform === nothing) == !normalize
            @test size(gp.data, 1) == n_design_points + n_added_points
            @test all(gp.data.p .== 2.0)
            @test gp.data.y ≈ 2 .* sin.(gp.data.x1) .+ 1
            @test gp.posterior.prior.mean.c == 1.0
            predictions = copy(gp.data)
            evaluate!(gp, predictions)
            @test predictions.y_mean ≈ gp.data.y atol = 1.0e-5
        end
    end

    @testset "Refit preserves transformation and ignores duplicates" begin
        gp = GaussianProcess(copy(data), :y; learn_hyperparameters = false)
        original_transform = gp.transform
        new_data = DataFrame(x1 = [15.0])
        evaluate!(model, new_data)
        gp = UncertaintyQuantification._refit_gp(
            gp, new_data, MaximumLikelihoodEstimation(), gp.σ², false, false
        )
        @test gp.transform === original_transform
        @test size(gp.data, 1) == n_design_points + 1
        gp = UncertaintyQuantification._refit_gp(
            gp, new_data, MaximumLikelihoodEstimation(), gp.σ², false, false
        )
        @test size(gp.data, 1) == n_design_points + 1
    end

end
