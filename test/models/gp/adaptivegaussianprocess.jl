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

    prior = GP(ConstMean(0.0), SqExponentialKernel())

    # Initial design used by the DataFrame-based constructors
    data = sample(input, design)
    evaluate!(model, data)

    @testset "GP prior + UQInput" begin

        gp = AdaptiveGaussianProcess(
            prior, input, model, :y, acquisition_function, n_added_points,
            n_design_points, design;
            candidate_sampling = candidate_sampling,
        )

        @test gp isa GaussianProcess
        @test size(gp.training_data, 1) == n_design_points + n_added_points
        @test gp.output == :y
    end

    @testset "Default prior + UQInput" begin

        gp_default = AdaptiveGaussianProcess(
            input, model, :y, acquisition_function, n_added_points,
            n_design_points, design;
            candidate_sampling = candidate_sampling,
        )

        @test gp_default isa GaussianProcess
        @test size(gp_default.training_data, 1) == n_design_points + n_added_points
    end

    @testset "Pre-fit GaussianProcess" begin
        gp_model = GaussianProcess(
            input, model, :y;
            experimental_design = design,
            mean_fct = ZeroMean(),
            kernel = SqExponentialKernel(),
        )

        gp = AdaptiveGaussianProcess(
            gp_model, input, model, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        @test gp isa GaussianProcess
        @test size(gp.training_data, 1) == n_design_points + n_added_points

        # n_added_points = 0 should return the training data unchanged
        gp_unchanged = AdaptiveGaussianProcess(
            gp_model, input, model, acquisition_function, 0;
            candidate_sampling = candidate_sampling,
        )
        @test size(gp_unchanged.training_data, 1) == n_design_points + n_added_points
    end

    @testset "GP prior + DataFrame" begin

        gp = AdaptiveGaussianProcess(
            prior, copy(data), input, model, :y, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        @test gp isa GaussianProcess
        @test size(gp.training_data, 1) == n_design_points + n_added_points
        @test gp.output == :y
    end

    @testset "Default prior + DataFrame" begin

        gp_default = AdaptiveGaussianProcess(
            copy(data), input, model, :y, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        gp_explicit = AdaptiveGaussianProcess(
            prior, copy(data), input, model, :y, acquisition_function, n_added_points;
            candidate_sampling = candidate_sampling,
        )

        @test gp_default isa GaussianProcess
        @test size(gp_default.training_data, 1) == n_design_points + n_added_points
    end

    @testset "Not learn hyperparameters" begin
        gp_model = GaussianProcess(
            input, model, :y;
            experimental_design = design,
            mean_fct = ConstMean(0.0),
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

end
