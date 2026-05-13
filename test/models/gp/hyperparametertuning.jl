@testset "Hyperparameter tuning" begin

    @testset "1D Input" begin
        x = collect(range(0, stop=5, length=10))
        y = sin.(x)
        data = DataFrame(:x => x, :y => y)

        σ² = 1e-5
        kernel = SqExponentialKernel() ∘ ScaleTransform(10.0)
        prior_gp = with_gaussian_noise(GP(0.0, kernel), σ²)
        gp = GaussianProcess(prior_gp, data, :y)
        opt_gp = optimize_hyperparameters(gp, MaximumLikelihoodEstimation())
        
        likelihood_no_opt = logpdf(gp.posterior_gp(x), y)
        likelihood_opt = logpdf(opt_gp.posterior_gp(x), y)
        @test likelihood_opt > likelihood_no_opt
    end
    
    @testset "2D Input" begin
        x = [collect(range(0, stop=5, length=10)) collect(range(0, stop=5, length=10))]
        y = sin.(x[:, 1]) + cos.(x[:, 2])
        data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)

        σ² = 1e-5
        kernel = SqExponentialKernel() ∘ ARDTransform([5.0, 5.0])
        prior_gp = with_gaussian_noise(GP(0.0, kernel), σ²)
        gp = GaussianProcess(prior_gp, data, :y)
        opt_gp = optimize_hyperparameters(gp, MaximumLikelihoodEstimation())

        likelihood_no_opt = logpdf(gp.posterior_gp(RowVecs(x)), y)
        likelihood_opt = logpdf(opt_gp.posterior_gp(RowVecs(x)), y)
        @test likelihood_opt > likelihood_no_opt
    end
end